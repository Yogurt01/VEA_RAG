import argparse
import gc
import json
import os
import sys
import time
import re
import unicodedata
from glob import glob
from pathlib import Path
import subprocess
# from dotenv import load_dotenv

import cv2
import torch

import fiftyone as fo
import fiftyone.zoo as foz

from pydantic import BaseModel
from typing import Optional, Dict, List

from scenedetect import AdaptiveDetector
from segment import scene_detection, segment_with_stt_timestamp

# load_dotenv()
HUGGING_FACE_TOKEN = os.environ.get('HUGGING_FACE_TOKEN')

QWEN_PROMPT_TEMPLATE =  """You have the following transcript for this video scene:
{transcript}

Using the transcript above only as a hint for who is speaking, analyze this video clip.
Respond ONLY with a valid JSON object, no markdown, no explanation, no extra text.

{{
    "visual_description": "Describe only what you SEE: setting, people present, their appearance (clothing, expressions, posture), physical actions, camera angle/framing, AND any text PHYSICALLY VISIBLE on screen (signs, banners, lower-thirds, overlays). Include any visible text directly inside this description. Do NOT interpret meaning or infer narrative.",
    "mood": "calm|energetic|dramatic|neutral",
    "color_temperature": "warm|cool|achromatic",
    "color_saturation": "high|medium|low",
    "color_lightness": "high|low",
    "lighting": "natural|studio|low_light|vibrant_neon|flat_dull",
    "background": "minimalist|luxury_modern|urban_outdoor|nature|cluttered_home"
}}

For mood, color_temperature, color_saturation, color_lightness, lighting, background:
pick EXACTLY ONE value from the options listed.
"""

COHERENT_SCENE_SYSTEM_PROMPT = """You are a narrative analyst writing scene-level captions for short video clips.
You will receive a list of scenes from a single video. Each scene includes a transcript with speaker labels and a visual description.
Your goal is to write captions that will later be used to construct a discourse graph — specifically an RST (Rhetorical Structure Theory) Tree. Therefore, each caption must clearly encode the scene's rhetorical/narrative function so that an RST parser can produce a rich, hierarchical tree with diverse relation types (not just Sequence or simple Cause).

## RST Alignment & Diversity Goals (CRITICAL)
- Treat each caption as an Elementary Discourse Unit (EDU).
- Vary relation types across scenes to create a deep, non-flat RST tree: use Elaboration, Background, Circumstance, Cause/Result (volitional & non-volitional), Purpose, Means, Condition, Contrast, Antithesis, Concession, Motivation, Evidence, Evaluation, Sequence, Conjunction.
- Prefer nuclearity: make the core action/event the **nucleus** (main clause), supporting info the **satellite** (subordinate clause or adverbial).
- Use explicit discourse markers to cue relations:
  - Cause/Result: "causing...", "as a result...", "leading to...", "triggering..."
  - Purpose/Means: "to achieve...", "in order to...", "by..."
  - Contrast/Antithesis/Concession: "however...", "despite...", "although...", "in contrast to the previous..."
  - Background/Circumstance: "To set the context...", "Against the backdrop of..."
  - Elaboration: "specifically...", "for example...", "adding that..."
  - Motivation/Evidence: "motivating...", "providing evidence that...", "revealing why..."
- Make causal, temporal, logical, or rhetorical relationship to adjacent scenes explicit whenever possible.

## Caption writing rules
- Write exactly 1-2 sentences per scene, specific and concrete.
- Each caption must answer: WHAT happens, WHO is involved, and WHY it matters rhetorically in the flow of the video (nucleus + satellite).
- Capture turning points, emotional shifts, reactions, and consequences — not just static descriptions.
- If a scene is a reaction/response to the previous: start with "In response to...", "Following...", "As a result of the previous scene...".
- If a scene provides background or context: frame it as "To provide background...", "Establishing the context for...".
- If a scene is a close-up/insert/silent transition: describe its rhetorical function (e.g., "Emphasizing the emotional impact of the previous revelation...").

What NOT to do:
- Do NOT repeat shared background details in every caption.
- Do NOT write generic captions like "two people are talking" or "the scene continues."
- Do NOT overuse temporal markers only ("then...", "next...") — this creates flat Sequence-only trees.
- Do NOT summarize the whole video — each caption covers only its own scene.

## Speaker diarization note
- Speaker labels (SPEAKER_00, SPEAKER_01, etc.) are auto-generated and may be incorrect.
- Cross-reference speaker labels with visual descriptions, turn-taking logic, and topic continuity across scenes.
- If inconsistency is detected between the speaker label and the visual (e.g., label says SPEAKER_00 but visual shows a different person speaking), infer the correct speaker from context.
- If speaker identity cannot be determined, use neutral phrasing: "one of the speakers", "the person on screen".

## CRITICAL: Output format
- Your response MUST be ONLY a valid JSON array, starting with [ and ending with ].
- Do NOT include any text before or after the JSON array.
- Do NOT use markdown code blocks (no ```json ... ```).
- Each object must have exactly two keys: "scene_id" (integer) and "caption" (string).
- Example of correct output:
[{"scene_id": 0, "caption": "First caption here."}, {"scene_id": 1, "caption": "Second caption here."}]
"""

class SceneCaption(BaseModel):
    scene_id: int
    caption: str

class VideoCaptions(BaseModel):
    scenes: List[SceneCaption]


def normalize_path(path):
    """
    Normalize a path to use forward slashes for cross-platform compatibility.
    """
    return os.path.normpath(path).replace('\\', '/')

def normalize_vietnamese_text(text):
    """
    Normalize Vietnamese text to NFC form for consistent handling of diacritics.
    """
    return unicodedata.normalize('NFC', text.strip()) if text else ""

class Prompter:
    """
    A class to handle the entire video processing pipeline, including
    segmentation, OCR, captioning, and audio analysis.
    """
    def __init__(self, args):
        self.device = args.device
        self.root_dir = Path(normalize_path(args.root_dir))
        
        # Segmentation parameters
        self.adaptive_threshold = args.adaptive_threshold
        self.min_scene_len = args.min_scene_len
        self.window_width = args.window_width
        self.min_content_val = args.min_content_val

        # File names
        self.segmentation_filename = 'segments.json'
        self.clips_directory = 'clips'
        self.embedding_filename = 'scene_embeddings.pt'

        # Control flags for processing steps
        self.skip_segment = args.skip_segment
        self.skip_cut_video = args.skip_cut_video
        self.skip_visual_caption = args.skip_visual_caption
        self.skip_scene_caption = args.skip_scene_caption
        self.skip_audio = args.skip_audio
        self.skip_scene_embedding = args.skip_scene_embedding
        
        # Get all video directory paths, optionally limited
        all_video_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        limit = getattr(args, 'limit_videos', None)
        self.video_dirs = all_video_dirs[:limit] if limit and limit > 0 else all_video_dirs
        
        self.videos = []
        for vdir in self.video_dirs:
            mp4_files = list(vdir.glob('*.mp4'))
            if mp4_files:
                self.videos.append(mp4_files[0])

        self.base_dir = Path(__file__).resolve().parent
        self.beat_checkpoint_path = self.base_dir / "beats_checkpoints" / "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
        self.beat_config_json_path = self.base_dir / "beats" / "config.json"


    def _clean_memory(self):
        """
        Releases GPU and system memory.
        """
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("Warning: Could not clear CUDA cache due to memory error")
                else:
                    print(f"Warning: CUDA cache clear failed: {e}")

    ## --------------------
    ## 1. Video Segmentation
    ## --------------------
    def _segment_videos(self):
        detector = AdaptiveDetector(
            adaptive_threshold=self.adaptive_threshold,
            min_scene_len=self.min_scene_len,
            window_width=self.window_width,
            min_content_val=self.min_content_val
        )
        for video_dir, video_path in zip(self.video_dirs, self.videos):
            segment_file = video_dir / self.segmentation_filename
            try:
                if segment_file.exists():
                    with open(segment_file, 'r', encoding='utf-8') as f:
                        segments = json.load(f)
                    # Skip if segments are valid (not just start:0, end:0)
                    if not any(s.get('start') == 0 and s.get('end') == 0 for s in segments):
                        print(f"Skipping segmentation for {video_dir.name}, valid segments exist.")
                        continue

                # Perform scene detection
                result = scene_detection(str(video_path), detector)
                if result:
                    starts, ends = result
                    segments = segment_with_stt_timestamp(str(video_dir), starts, ends)
                    
                    # Normalize text and ensure 'ocr_text' field exists
                    for seg in segments:
                        if 'text' in seg:
                            seg['text'] = [normalize_vietnamese_text(t) for t in seg['text']] if isinstance(seg['text'], list) else normalize_vietnamese_text(seg['text'])
                else:
                    # Create a default segment covering the whole video
                    self._create_default_segment(video_path, segment_file)
                    continue

                with open(segment_file, 'w', encoding='utf-8') as f:
                    json.dump(segments, f, indent=2, ensure_ascii=False)
                print(f"Created segments for {video_dir.name}")

            except Exception as e:
                print(f"Error segmenting video {video_dir.name}: {e}. Creating default segment.")
                self._create_default_segment(video_path, segment_file)
        self._clean_memory()

    def _create_default_segment(self, video_path, segment_file_path):
        """
        Creates a single segment covering the entire video duration.
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                
                default_segment = [{"start": 0, "end": duration, "text": [], "ocr_text": []}]
                with open(segment_file_path, 'w', encoding='utf-8') as f:
                    json.dump(default_segment, f, indent=2, ensure_ascii=False)
                print(f"Created default segment for {video_path.parent.name}")
        except Exception as e:
            print(f"Error creating default segment for {video_path.parent.name}: {e}")

    ## --------------------
    ## 2. Cut Video into Clips
    ## --------------------
    def _cut_videos(self):
        for video_dir, video_path in zip(self.video_dirs, self.videos):
            segment_file = video_dir / self.segmentation_filename
            if not segment_file.exists():
                continue
            
            with open(segment_file, 'r', encoding='utf-8') as f:
                segments = json.load(f)

            clips_dir = video_dir / self.clips_directory
            clips_dir.mkdir(exist_ok=True)
            
            for i, seg in enumerate(segments):
                start_list = seg["start"]
                end_list = seg["end"]

                clip_start = start_list[0] if isinstance(start_list, list) else start_list
                clip_end = end_list[-1] if isinstance(end_list, list) else end_list

                actual_end = min(clip_start + 5.0, clip_end)

                clip_name = f"clip_{i:03d}_{clip_start:.2f}-{clip_end:.2f}.mp4"
                clip_path = clips_dir / clip_name

                ok = self._cut_clip(video_path, clip_start, actual_end, clip_path)
                if ok:
                    seg["clip_path"] = str(clip_path)

            with open(segment_file, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2, ensure_ascii=False)
                
    def _cut_clip(self, src: Path, start: float, end: float, dst: Path) -> bool:
        if dst.exists() and dst.stat().st_size > 1000:
            return True

        duration = round(end - start, 3)
        if duration <= 0:
            return False

        command = [
            "ffmpeg",
            "-ss", str(start),     
            "-t", str(duration),   
            "-i", str(src),
            "-c:v", "libx264",     
            "-crf", "18",          
            "-preset", "ultrafast",
            "-c:a", "aac",         
            "-y",
            "-loglevel", "error",
            str(dst)
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error cutting clip: {result.stderr}")
            return False
        return True

    ## --------------------
    ## 3. Generate Visual Caption Corpus
    ## --------------------
    def _generate_caption_corpus(self):
        # Register the model source
        foz.register_zoo_model_source(
            "https://github.com/harpreetsahota204/qwen3vl_video",
            overwrite=True
        )
        # Load Qwen3-VL model
        model = foz.load_zoo_model(
            "Qwen/Qwen3-VL-2B-Instruct",
            total_pixels=1*1024*32*32,
            max_frames=32,
            sample_fps=0.5
        )
        model.operation = "custom"

        for video_dir, video_path in zip(self.video_dirs, self.videos):
            segment_file = video_dir / self.segmentation_filename
            if not segment_file.exists():
                continue

            with open(segment_file, 'r', encoding='utf-8') as f:
                segments = json.load(f)

            if all("visual_description" in seg for seg in segments):
                print(f"Skipping visual caption for {video_dir.name}, already processed.")
                continue

            file_name = video_path.name
            dataset_name = f"clips-dataset-{file_name}"

            if fo.dataset_exists(dataset_name):
                fo.delete_dataset(dataset_name)
            clips_dataset = fo.Dataset(name=dataset_name)

            clip_to_idx = {}
            for i, scene in enumerate(segments):
                clip_path = scene.get("clip_path")
                if clip_path is None:
                    continue

                texts = scene.get("text", [])
                speakers = scene.get("speaker", [])
                transcript = [
                    {"speaker": sp, "text": t}
                    for sp, t in zip(speakers, texts) if sp.strip()
                ]

                s = fo.Sample(filepath=str(clip_path))
                s["transcript"] = transcript
                clips_dataset.add_sample(s)
                clip_to_idx[str(clip_path)] = i

            clips_dataset.compute_metadata()

            for sample in clips_dataset.iter_samples(autosave=True):
                transcript = sample.get_field("transcript") or ""
                sample["qwen_prompt"] = QWEN_PROMPT_TEMPLATE.format(transcript=transcript)

            clips_dataset.apply_model(
                model,
                prompt_field="qwen_prompt",
                label_field="custom_analysis",
                skip_failures=True
            )

            for sample in clips_dataset.iter_samples():
                idx = clip_to_idx.get(sample.filepath)
                if idx is None:
                    continue

                content_str = sample["custom_analysis_result"]
                if not content_str:
                    continue

                result = self._clean_and_parse_json(content_str)
                if "_raw" in result:
                    continue

                segments[idx]["visual_description"] = result.get("visual_description", "")
                segments[idx]["visual_elements"] = {
                    "mood": result.get("mood", ""),
                    "color_temperature": result.get("color_temperature", ""),
                    "color_saturation": result.get("color_saturation", ""),
                    "color_lightness": result.get("color_lightness", ""),
                    "lighting": result.get("lighting", ""),
                    "background": result.get("background", ""),
                }

            with open(segment_file, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2, ensure_ascii=False)

        del model
        self._clean_memory()

    def _clean_and_parse_json(self, text: str):
        text = re.sub(r'```(?:json)?\s*|\s*```', '', text).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            print(f"Failed to parse, saving raw: {text[:80]}")
            return {"_raw": text}

    
    ## --------------------
    ## 4. Generate Scene Caption
    ## --------------------
    def _generate_scene_captions(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils.import_utils import is_flash_attn_2_available

        model_id = "Qwen/Qwen3-4B-Instruct-2507"
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto"
        }
        if self.device == "cuda" and is_flash_attn_2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using Flash Attention 2")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, **model_kwargs
        ).eval()

        for video_dir, video_path in zip(self.video_dirs, self.videos):
            segment_file = video_dir / self.segmentation_filename
            if not segment_file.exists():
                continue

            with open(segment_file, 'r', encoding='utf-8') as f:
                segments = json.load(f)

            if all("caption" in seg for seg in segments):
                print(f"Skipping scene caption for {video_dir.name}, already processed.")
                continue

            # Build full payload for all scenes
            scenes_payload = []
            for i, seg in enumerate(segments):
                transcript_lines = [
                    f"{sp}: {t}"
                    for sp, t in zip(seg.get("speaker", []), seg.get("text", []))
                    if t.strip()
                ]
                scenes_payload.append({
                    "scene_id":           i,
                    "start":              seg["start"][0] if isinstance(seg["start"], list) else seg["start"],
                    "end":                seg["end"][-1]  if isinstance(seg["end"],   list) else seg["end"],
                    "transcript":         " / ".join(transcript_lines) or "(no speech)",
                    "visual_description": seg.get("visual_description", ""),
                })

            caption_map = self._generate_captions_batched(
                model, tokenizer, scenes_payload,
                batch_size=10
            )

            for i, seg in enumerate(segments):
                if i in caption_map:
                    seg["caption"] = caption_map[i]
                else:
                    print(f"Warning: Missing caption for scene {i} in {video_dir.name}")

            with open(segment_file, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2, ensure_ascii=False)

        del tokenizer, model
        self._clean_memory()


    def _call_qwen(self, model, tokenizer, messages: list) -> str:
        """
        Run inference on Qwen3-Instruct model using standard generation.
        """
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=8192,
            temperature=0.3,
            do_sample=True,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return content

    def _update_narrative_context(
        self,
        model,
        tokenizer,
        previous_context: str,
        batch_captioned: list,
        batch_start: int,
        batch_end: int,
    ) -> str:
        """
        Merge the previous narrative context with the new batch captions
        into a single updated summary paragraph.
        Keeps context length bounded regardless of video length.
        """
        new_captions_text = "\n".join(
            f"Scene_{item['scene_id']}: {item['caption']}"
            for item in batch_captioned
        )

        context_section = (
            f"Previous summary:\n{previous_context}\n\n"
            if previous_context else ""
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a video analyst. "
                    "You will receive a summary of earlier scenes and new scene captions. "
                    "Merge them into ONE updated summary paragraph (4-6 sentences max) "
                    "covering the entire video so far. "
                    "Be concise — this summary will be passed to future steps so it must stay short."
                )
            },
            {
                "role": "user",
                "content": (
                    f"{context_section}"
                    f"New scenes (scenes {batch_start}-{batch_end}):\n{new_captions_text}\n\n"
                    "Write the updated merged summary."
                )
            }
        ]

        return self._call_qwen(model, tokenizer, messages)

    def _parse_batch_captions(self, raw: str, scene_ids: List[int]) -> Dict[int, str]:
        """
        Parse JSON array from Qwen3-Instruct response with multiple fallback strategies.
        scene_ids: list of expected scene IDs to validate completeness.
        Returns {scene_id: caption}.
        """
        def extract_json(text: str) -> Optional[str]:
            # Strategy 1: Remove markdown code blocks
            text = re.sub(r'```(?:json)?\s*|\s*```', '', text).strip()
            
            # Strategy 2: Find outermost JSON array [...]
            match = re.search(r'(\[.*\])', text, re.DOTALL)
            if match:
                return match.group(1)
            
            # Strategy 3: Find JSON object with "scenes" or "captions" key
            match = re.search(r'(\{.*\})', text, re.DOTALL)
            if match:
                return match.group(1)
            
            # Strategy 4: Try to find line-by-line JSON objects and wrap them
            lines = text.split('\n')
            candidates = [line.strip() for line in lines if line.strip().startswith('{')]
            if candidates:
                wrapped = '[' + ','.join(candidates) + ']'
                return wrapped
            
            return None

        def validate_captions(parsed: Dict[int, str], expected_ids: List[int]) -> bool:
            """Check if all expected scene IDs have captions."""
            return all(sid in parsed for sid in expected_ids)

        # Try parsing up to 3 times with increasing leniency
        for attempt in range(3):
            try:
                json_str = extract_json(raw)
                if not json_str:
                    raise ValueError("No JSON structure found")
                
                data = json.loads(json_str)
                
                # Handle different possible structures
                if isinstance(data, list):
                    result = {item["scene_id"]: item["caption"] for item in data if "scene_id" in item and "caption" in item}
                elif isinstance(data, dict):
                    scenes = data.get("scenes", data.get("captions", []))
                    result = {item["scene_id"]: item["caption"] for item in scenes if "scene_id" in item and "caption" in item}
                else:
                    raise ValueError(f"Unexpected JSON type: {type(data)}")
                
                # Validate completeness
                if validate_captions(result, scene_ids):
                    return result
                else:
                    missing = set(scene_ids) - set(result.keys())
                    print(f"Attempt {attempt+1}: Missing captions for scenes {missing}, retrying...")
                    
            except json.JSONDecodeError as e:
                print(f"Attempt {attempt+1}: JSON decode error: {e}")
            except Exception as e:
                print(f"Attempt {attempt+1}: Parsing error: {e}")
            
            # Exponential backoff before retry (if we were to re-call the model)
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
        
        # Final fallback: partial recovery — return whatever we could parse
        print(f"Final fallback: returning partial results for {len([s for s in scene_ids if s in locals().get('result', {})])}/{len(scene_ids)} scenes")
        return locals().get('result', {})
    
    def _generate_captions_batched(
        self,
        model,
        tokenizer,
        scenes_payload: list,
        batch_size: int = 10,
        max_retries: int = 2,
    ) -> dict:
        caption_map = {}
        narrative_context = ""
        total = len(scenes_payload)
        n_batches = (total + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total)
            batch = scenes_payload[start_idx:end_idx]
            expected_scene_ids = [item["scene_id"] for item in batch]

            print(f"  Captioning scenes {start_idx}-{end_idx - 1} / {total - 1} "
                f"(batch {batch_idx + 1}/{n_batches})")

            # Build context block from accumulated narrative summary
            context_block = ""
            if narrative_context:
                context_block = (
                    "NARRATIVE CONTEXT — what has happened in the video so far "
                    "(do NOT rewrite these, use them only to understand continuity):\n"
                    f"{narrative_context}\n\n"
                )

            user_prompt = (
                context_block
                + f"Scenes to caption:\n{json.dumps(batch, ensure_ascii=False, indent=2)}\n\n"
                "Write captions that highlight what changes between scenes, not what stays the same. "
                "Use the narrative context above to maintain continuity with earlier scenes.\n\n"
                "IMPORTANT: Respond with ONLY a valid JSON array, no explanations, no markdown."
            )

            # Retry loop for model call + parsing
            for retry in range(max_retries + 1):
                messages = [
                    {"role": "system", "content": COHERENT_SCENE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]

                raw = self._call_qwen(model, tokenizer, messages)
                batch_caption_map = self._parse_batch_captions(raw, expected_scene_ids)

                if batch_caption_map and len(batch_caption_map) == len(expected_scene_ids):
                    break
                
                print(f"Retry {retry+1}/{max_retries} for batch {batch_idx+1} due to parse failure")
                # Optional: add a small delay or adjust temperature for retry
                if retry < max_retries:
                    time.sleep(3)

            caption_map.update(batch_caption_map)

            # Build list of successfully captioned scenes
            batch_captioned = [
                {"scene_id": item["scene_id"], "caption": caption_map[item["scene_id"]]}
                for item in batch
                if item["scene_id"] in caption_map
            ]

            # Summarize this batch to update narrative context for the next batch
            if batch_captioned and batch_idx < n_batches - 1:
                print(f"  Updating narrative context after batch {batch_idx + 1}...")
                narrative_context = self._update_narrative_context(
                    model, tokenizer,
                    previous_context=narrative_context,
                    batch_captioned=batch_captioned,
                    batch_start=start_idx,
                    batch_end=end_idx - 1,
                )

        return caption_map

    ## --------------------
    ## 5. Audio Tagging
    ## --------------------
    def _audio_tagging(self):
        current_dir = Path(__file__).resolve().parent
        beats_folder = current_dir / "beats"
        
        if str(beats_folder) not in sys.path:
            sys.path.insert(0, str(beats_folder))
        from infer_beats import load_model, audio_tagging
        model, config_data = load_model(self.beat_checkpoint_path, self.beat_config_json_path)

        for video_dir, video_path in zip(self.video_dirs, self.videos):
            segment_file = video_dir / self.segmentation_filename
            if not segment_file.exists():
                continue
            
            with open(segment_file, 'r', encoding='utf-8') as f:
                segments = json.load(f)

            if all("audio_tags" in seg for seg in segments):
                print(f"Skipping audio tagging for {video_dir.name}, already processed.")
                continue

            clips_dir = video_dir / self.clips_directory
            
            for i, seg in enumerate(segments):
                start_list = seg["start"]
                end_list = seg["end"]

                clip_start = start_list[0] if isinstance(start_list, list) else start_list
                clip_end = end_list[-1] if isinstance(end_list, list) else end_list

                clip_name = f"clip_{i:03d}_{clip_start:.2f}-{clip_end:.2f}.mp4"
                clip_path = clips_dir / clip_name

                if not clip_path.exists():
                    print(f"Clip not found, skipping scene {i}: {clip_path.name}")
                    continue

                # Extract audio of each clip
                audio_path = clips_dir / f"audio_{i:03d}_{clip_start:.2f}-{clip_end:.2f}.wav"
                print(f"Extracting audio from: {clip_path.name}")
                
                command = [
                    'ffmpeg', '-y', '-i', str(clip_path),
                    '-vn', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000',
                    '-loglevel', 'error', str(audio_path)
                ]

                try:
                    subprocess.run(command, check=True, capture_output=True)
                    print(f"Successfully extracted: {audio_path}")
                    results = audio_tagging(str(audio_path), model, config_data, top_k=2)
                    seg["audio_tags"] = [item["label"] for item in results.get("audio_tags", [])]
                    seg["audio_vibes"] = [item["label"] for item in results.get("audio_vibes", [])]

                except Exception as e:
                    print(f"Error at Scene {i}: {e}")
                finally:
                    if audio_path.exists():
                        audio_path.unlink()

            with open(segment_file, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2, ensure_ascii=False)
        
        del model
        self._clean_memory()

    ## --------------------
    ## 6. Generate Scene Embedding
    ## --------------------
    def _generate_scene_embeddings(self, force_recompute=False):
        from qwen3_vl_embedding import Qwen3VLEmbedder

        # Initialize the embedding model with bfloat16 for inference efficiency
        embedder = Qwen3VLEmbedder(
            model_name_or_path="Qwen/Qwen3-VL-Embedding-2B",
            torch_dtype=torch.bfloat16
        )

        for video_dir in self.video_dirs:
            segment_file = video_dir / self.segmentation_filename
            if not segment_file.exists():
                continue

            with open(segment_file, 'r', encoding='utf-8') as f:
                segments = json.load(f)

            embedding_file = video_dir / self.embedding_filename

            if embedding_file.exists() and not force_recompute:
                print(f"Skipping {video_dir.name}: Embeddings already exist.")
                continue

            scene_embeddings = {}

            for i, scene in enumerate(segments):
                clip_path = scene.get("clip_path")
                if not clip_path or not Path(clip_path).exists():
                    print(f"Scene {i} missing clip, skip")
                    continue

                # Construct the multimodal prompt incorporating all extracted metadata
                text_prompt = self._build_unified_embedding_text(scene)

                inputs = [{
                    "text": text_prompt,
                    "video": clip_path
                }]

                # Generate the 2048-dimensional embedding
                emb = embedder.process(inputs)

                # Move to CPU and cast to float32 for downstream GNN compatibility
                scene_embeddings[i] = emb[0].cpu().to(torch.float32) 

            if not scene_embeddings:
                continue

            # Sort by scene index to ensure temporal and structural alignment
            sorted_indices = sorted(scene_embeddings.keys())
            scene_embeddings = torch.stack([scene_embeddings[i] for i in sorted_indices])

            # Save the feature matrix along with metadata for reproducibility
            torch.save({
                "embeddings": scene_embeddings,
                "scene_ids": sorted_indices,
                "metadata": {
                    "model": "Qwen3-VL-Embedding-2B",
                    "embed_dim": 2048,
                    "created_at": time.ctime(),
                }
            }, embedding_file)

        del embedder
        self._clean_memory()

    def _build_unified_embedding_text(self, scene: dict) -> str:
        # Format the spoken transcript with speaker labels
        transcript_lines = []
        for spk, txt in zip(scene.get("speaker", []), scene.get("text", [])):
            if txt and str(txt).strip():  # bỏ dòng rỗng
                transcript_lines.append(f"{spk}: {txt.strip()}")
        
        transcript = "\n".join(transcript_lines) if transcript_lines else "(no spoken dialogue)"

        # Extract primary and secondary physical activities
        vis = scene.get("visual_elements", {})

        # Final prompt construction utilizing Task-Specific Instruction
        text_for_embedding = f"""Represent this video scene as a node in a multimodal discourse graph:

Spoken transcript:
{transcript}

Scene caption:
{scene.get("caption", "(no caption)")}

Mood: {vis.get("mood", "neutral")}
Lighting: {vis.get("lighting", "unknown")}
Background: {vis.get("background", "unknown")}
Color temperature: {vis.get("color_temperature", "unknown")}
Color saturation: {vis.get("color_saturation", "medium")}
Color lightness: {vis.get("color_lightness", "medium")}

Audio atmosphere:
Tags: {", ".join(scene.get("audio_tags", [])) or "none"}
Vibes: {", ".join(scene.get("audio_vibes", [])) or "none"}"""

        return text_for_embedding.strip()


    def run(self):
        """
        Executes the entire processing pipeline.
        """
        self._clean_memory()
        start_time = time.time()
        
        try:
            if not self.skip_segment:
                print("\n--- Step 1: Video Segmentation ---")
                self._segment_videos()

            if not self.skip_cut_video:
                print("\n--- Step 2: Cut Video into Clips ---")
                self._cut_videos()

            if not self.skip_visual_caption:
                print("\n--- Step 3: Generate Visual Caption Corpus ---")
                self._generate_caption_corpus()

            if not self.skip_scene_caption:
                print("\n--- Step 4: Generate Scene Caption Corpus ---")
                self._generate_scene_captions()

            if not self.skip_audio:
                print("\n--- Step 5: Audio Tagging ---")
                self._audio_tagging()

            if not self.skip_scene_embedding:
                print("\n--- Step 6: Generate Scene Embedding ---")
                self._generate_scene_embeddings()

            print(f"\nAll processing steps completed in {time.time() - start_time:.2f} seconds.")

        except Exception as e:
            print(f"\nA critical error occurred during processing: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._clean_memory()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process videos to generate segments, OCR, and captions.")
    parser.add_argument('--root_dir', type=str, default="test_videos/SnapUGC1", help="Directory containing video folders.")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use ('cuda' or 'cpu').")
    parser.add_argument('--limit_videos', type=int, default=0, help="Limit the number of videos to process (0 for all).")

    # Segmentation args
    parser.add_argument('--adaptive_threshold', type=float, default=1.5)
    parser.add_argument('--min_scene_len', type=int, default=15)
    parser.add_argument('--window_width', type=int, default=4)
    parser.add_argument('--min_content_val', type=float, default=6.0)

    # Skip flags
    parser.add_argument('--skip_segment', action='store_true', help="Skip video segmentation.")
    parser.add_argument('--skip_cut_video', action='store_true', help="Skip cut video into clips.")
    parser.add_argument('--skip_visual_caption', action='store_true', help="Skip visual caption generation.")
    parser.add_argument('--skip_scene_caption', action='store_true', help="Skip scene caption generation.")
    parser.add_argument('--skip_audio', action='store_true', help="Skip audio tagging.")
    parser.add_argument('--skip_scene_embedding', action='store_true', help="Skip scene embedding generation.")

    args = parser.parse_args()

    prompter = Prompter(args)
    prompter.run()