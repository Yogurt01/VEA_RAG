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

import cv2
import torch

from pydantic import BaseModel
from typing import Optional, Dict, List

os.environ["HF_HOME"] = "/models/huggingface_cache"
os.environ["TORCH_HOME"] = "/models/torch_cache"
os.environ["HF_HUB_OFFLINE"] = "1"


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
- Write ALL captions in English only, regardless of the language spoken in the video or transcript.

What NOT to do:
- Do NOT repeat shared background details in every caption.
- Do NOT write generic captions like "two people are talking" or "the scene continues."
- Do NOT overuse temporal markers only ("then...", "next...") — this creates flat Sequence-only trees.
- Do NOT summarize the whole video — each caption covers only its own scene.
- Do NOT write captions in any language other than English, even if the transcript is in Vietnamese or another language.
- Do NOT use any emoji or symbols in captions.
- Do NOT use possessive apostrophes or contractions (e.g., write "the girl is" not "girl's", "do not" not "don't").

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
        self.min_nodes = args.min_nodes
        self.merge_gap = args.merge_gap

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
        self.delete_clips = args.delete_clips

        # Get all video directory paths, optionally limited
        all_video_dirs = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir()
        ])
        
        pending_video_dirs = []
        skipped_count = 0

        for vdir in all_video_dirs:
            if self._is_video_fully_processed(vdir):
                skipped_count += 1
                continue

            pending_video_dirs.append(vdir)

        limit = getattr(args, "limit_videos", 0)

        if limit and limit > 0:
            self.video_dirs = pending_video_dirs[:limit]
        else:
            self.video_dirs = pending_video_dirs

        print(
            f"Found {len(all_video_dirs)} video folders | "
            f"Completed: {skipped_count} | "
            f"Pending: {len(pending_video_dirs)} | "
            f"Selected: {len(self.video_dirs)}"
        )
        
        self.videos = []
        for vdir in self.video_dirs:
            mp4_files = list(vdir.glob('*.mp4'))
            if mp4_files:
                self.videos.append(mp4_files[0])

        self.base_dir = Path(__file__).resolve().parent
        self.qwen_visual_model_path = args.qwen_visual_model_path
        self.qwen_instruct_model_path = args.qwen_instruct_model_path
        self.qwen_vl_embedding_model_path = args.qwen_vl_embedding_model_path
        self.beat_checkpoint_path = args.beat_checkpoint_path

    def _is_video_fully_processed(self, video_dir: Path) -> bool:
        segment_file = video_dir / self.segmentation_filename
        embedding_file = video_dir / self.embedding_filename

        return (
            segment_file.exists()
            and embedding_file.exists()
            and embedding_file.stat().st_size > 0
        )

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
        from scenedetect import AdaptiveDetector
        from segment import scene_detection, segment_with_stt_timestamp, fallback_boundaries

        for video_dir, video_path in zip(self.video_dirs, self.videos):
            if video_path is None:
                print(f"[SKIP] {video_dir.name}: no .mp4")
                continue

            segment_file = video_dir / self.segmentation_filename

            # Skip nếu đã có segment hợp lệ
            if segment_file.exists():
                try:
                    with open(segment_file, 'r', encoding='utf-8') as f:
                        existing = json.load(f)
                    if existing and len(existing) >= self.min_nodes:
                        print(f"[SKIP] {video_dir.name}: {len(existing)} segments exist")
                        continue
                except Exception:
                    pass

            print(f"\n[Video] {video_dir.name}")
            
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            duration = frame_count / fps
            min_sec = max(0.8, min(3.0, duration * 0.08))
            min_scene_len = int(fps * min_sec)

            sensitive_detector = AdaptiveDetector(
                adaptive_threshold=2.5,
                min_scene_len=min_scene_len,
                window_width=2,
                min_content_val=3.0,
            )

            try:
                print("Scene detection")
                starts, ends = scene_detection(str(video_path), sensitive_detector)
                scene_ok = starts is not None and len(starts) > 2

                if scene_ok:
                    print(f"Scene detection OK ({len(starts)} scenes) → STT align")
                else:
                    print(f"Scene detection insufficient → fallback")
                    starts, ends = fallback_boundaries(
                        video_path = video_path,
                        video_dir  = video_dir,
                        min_nodes  = self.min_nodes,
                        merge_gap  = self.merge_gap,
                    )

                print(f"STT alignment ({len(starts)} segments)")
                segments = segment_with_stt_timestamp(str(video_dir), starts, ends)

                # Normalize text
                for seg in segments:
                    if isinstance(seg.get('text'), list):
                        seg['text'] = [
                            normalize_vietnamese_text(t) for t in seg['text'] if t
                        ]

                with open(segment_file, 'w', encoding='utf-8') as f:
                    json.dump(segments, f, indent=2, ensure_ascii=False)

                print(f"Saved {len(segments)} segments → {segment_file.name}")

            except Exception as e:
                print(f"  [ERROR] {video_dir.name}: {e}")
                import traceback
                traceback.print_exc()
                self._create_default_segment(video_path, segment_file)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _create_default_segment(self, video_path, segment_file_path):
        """Tạo 1 segment duy nhất bao toàn bộ video (last-resort fallback)."""
        from segment import get_video_duration
        try:
            duration = get_video_duration(str(video_path))
            default  = [{
                "start":   [0.0],
                "end":     [round(duration, 2)],
                "text":    [],
                "speaker": []
            }]
            with open(segment_file_path, 'w', encoding='utf-8') as f:
                json.dump(default, f, indent=2, ensure_ascii=False)
            print(f"Default segment saved ({duration:.1f}s)")
        except Exception as e:
            print(f"  [ERROR] _create_default_segment: {e}")

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
            
            print(f"Processing: {video_dir.name} ({len(segments)} segments)")

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
        # Import the standalone model wrapper
        from qwen3_vl_visual import Qwen3VLStandalone
        
        model_path = self.qwen_visual_model_path

        # Initialize the standalone model with same parameters as original
        model = Qwen3VLStandalone(
            model_path=model_path,
            total_pixels=1*1024*32*32,
            min_pixels=64*32*32,
            max_frames=32,
            sample_fps=0.5,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )
        
        for video_dir, video_path in zip(self.video_dirs, self.videos):
            segment_file = video_dir / self.segmentation_filename
            if not segment_file.exists():
                continue

            with open(segment_file, 'r', encoding='utf-8') as f:
                segments = json.load(f)

            if all(seg.get("visual_description", "").strip() for seg in segments):
                print(f"Skipping visual caption for {video_dir.name}, already processed.")
                continue

            print(f"Processing visual captions for {video_dir.name} ({len(segments)} segments)")
            
            clips_dir = video_dir / self.clips_directory

            for i, scene in enumerate(segments):
                # Bỏ qua segment đã có visual_description
                if scene.get("visual_description", "").strip():
                    continue

                try:
                    clip_path = scene.get("clip_path")
                    if clip_path is None:
                        raise ValueError(f"clip_path missing for segment {i}")

                    # Resolve clip path
                    clip_full_path = Path(clip_path)
                    if not clip_full_path.exists():
                        clip_full_path = clips_dir / Path(clip_path).name
                        if not clip_full_path.exists():
                            raise FileNotFoundError(f"Clip not found: {clip_path}")

                    # Prepare transcript context
                    texts = scene.get("text", [])
                    speakers = scene.get("speaker", [])
                    transcript_parts = [
                        f"{sp.strip()}: {t.strip()}"
                        for sp, t in zip(speakers, texts)
                        if sp.strip() and t.strip()
                    ]
                    transcript_str = " | ".join(transcript_parts)
                    
                    # Build prompt
                    prompt = QWEN_PROMPT_TEMPLATE.format(transcript=transcript_str)
                    
                    # Run inference
                    result = model.predict(
                        video_path=str(clip_full_path),
                        prompt=prompt,
                        parse_json=True
                    )
                    
                    # Handle result
                    if isinstance(result, dict) and "_raw" not in result:
                        segments[i]["visual_description"] = result.get("visual_description", "")
                        segments[i]["visual_elements"] = {
                            "mood": result.get("mood", ""),
                            "color_temperature": result.get("color_temperature", ""),
                            "color_saturation": result.get("color_saturation", ""),
                            "color_lightness": result.get("color_lightness", ""),
                            "lighting": result.get("lighting", ""),
                            "background": result.get("background", ""),
                        }
                        print(f"  Segment {i}: OK")
                    else:
                        raw_preview = str(result)[:100] if result else "None"
                        print(f"  Segment {i}: Parse failed, raw: {raw_preview}...")
                        
                except Exception as e:
                    print(f"  Segment {i}: Error - {e}")
                    continue
            
            # Lưu toàn bộ sau khi xử lý xong video
            with open(segment_file, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2, ensure_ascii=False)
            print(f"Saved updated segments for {video_dir.name}")

        # Cleanup
        del model
        self._clean_memory()
    
    ## --------------------
    ## 4. Generate Scene Caption
    ## --------------------
    def _generate_scene_captions(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils.import_utils import is_flash_attn_2_available

        model_path = self.qwen_instruct_model_path
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto"
        }
        if self.device == "cuda" and is_flash_attn_2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using Flash Attention 2")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, **model_kwargs
        ).eval()

        try:
            for video_dir, video_path in zip(self.video_dirs, self.videos):
                segment_file = video_dir / self.segmentation_filename
                if not segment_file.exists():
                    continue

                with open(segment_file, 'r', encoding='utf-8') as f:
                    segments = json.load(f)

                if all(seg.get("caption", "").strip() for seg in segments):
                    print(f"Skipping {video_dir.name}: already processed")
                    continue

                print(f"Generating captions for: {video_dir.name}")

                # Build payload
                scenes_payload = []
                for i, seg in enumerate(segments):
                    transcript_lines = [
                        f"{sp}: {t}"
                        for sp, t in zip(seg.get("speaker", []), seg.get("text", []))
                        if t and t.strip()
                    ]
                    scenes_payload.append({
                        "scene_id": i,
                        "start": seg["start"][0] if isinstance(seg["start"], list) else seg["start"],
                        "end": seg["end"][-1] if isinstance(seg["end"], list) else seg["end"],
                        "transcript": " / ".join(transcript_lines) or "(no speech)",
                        "visual_description": seg.get("visual_description", ""),
                    })

                caption_map = self._generate_captions_batched(
                    model, tokenizer, scenes_payload, batch_size=10
                )

                # Apply captions with warning for missing ones
                missing_count = 0
                for i, seg in enumerate(segments):
                    if i in caption_map and caption_map[i]:
                        seg["caption"] = caption_map[i]
                    else:
                        missing_count += 1
                        print(f"Missing caption for scene {i}")
                
                print(f"{video_dir.name}: {len(segments) - missing_count}/{len(segments)} captions generated")

                with open(segment_file, 'w', encoding='utf-8') as f:
                    json.dump(segments, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Error in caption generation: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            del tokenizer, model
            self._clean_memory()
            print("Memory cleaned")


    def _call_qwen(self, model, tokenizer, messages: list, temperature: float = 0.3) -> str:
        """
        Run inference with adjustable temperature for retry robustness.
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
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
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
        Parse JSON with regex fallback for malformed outputs.
        """
        result = {}
        
        def extract_json(text: str) -> Optional[str]:
            text = re.sub(r'```(?:json)?\s*|\s*```', '', text).strip()
            match = re.search(r'(\[.*\])', text, re.DOTALL)
            if match:
                return match.group(1)
            match = re.search(r'(\{.*\})', text, re.DOTALL)
            if match:
                return match.group(1)
            lines = text.split('\n')
            candidates = [line.strip() for line in lines if line.strip().startswith('{')]
            if candidates:
                return '[' + ','.join(candidates) + ']'
            return None

        # Try standard JSON parsing first
        for attempt in range(2):
            try:
                json_str = extract_json(raw)
                if not json_str:
                    raise ValueError("No JSON structure found")
                
                data = json.loads(json_str)
                
                if isinstance(data, list):
                    result = {
                        item["scene_id"]: item["caption"] 
                        for item in data 
                        if isinstance(item, dict) and "scene_id" in item and "caption" in item
                    }
                elif isinstance(data, dict):
                    scenes = data.get("scenes", data.get("captions", []))
                    result = {
                        item["scene_id"]: item["caption"] 
                        for item in scenes 
                        if isinstance(item, dict) and "scene_id" in item and "caption" in item
                    }
                
                if result:
                    return result
                    
            except json.JSONDecodeError as e:
                if attempt == 0:
                    continue
                break
            except Exception:
                break

        pattern = r'"scene_id"\s*:\s*(\d+).*?"caption"\s*:\s*"((?:[^"\\]|\\.)*)"'
        matches = re.findall(pattern, raw, re.DOTALL)
        
        if matches:
            print(f"JSON parse failed, rescued {len(matches)} captions via regex")
            result = {}
            for sid, caption in matches:
                # Unescape common sequences
                clean_caption = caption.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                result[int(sid)] = clean_caption
            return result
        
        # All failed
        print(f"Failed to parse any captions. Raw preview: {raw[:200]}...")
        return {}
    
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

            context_block = ""
            if narrative_context:
                context_block = (
                    "NARRATIVE CONTEXT — what has happened in the video so far "
                    "(do NOT rewrite these, use them only to understand continuity):\n"
                    f"{narrative_context}\n\n"
                )

            base_user_prompt = (
                context_block
                + f"Scenes to caption:\n{json.dumps(batch, ensure_ascii=False, indent=2)}\n\n"
                "Write captions that highlight what changes between scenes, not what stays the same. "
                "Use the narrative context above to maintain continuity with earlier scenes.\n\n"
                "IMPORTANT: Respond with ONLY a valid JSON array, no explanations, no markdown."
            )

            # Retry loop: Re-call model with lower temperature on failure
            batch_caption_map = {}
            for retry in range(max_retries + 1):
                # Lower temperature on retry = more deterministic output
                current_temp = 0.1 if retry > 0 else 0.3
                
                # Add extra format emphasis on retry
                user_prompt = base_user_prompt
                if retry > 0:
                    user_prompt += "\n\nREMEMBER: Output MUST be valid JSON. Escape all quotes like \\\"this\\\"."

                messages = [
                    {"role": "system", "content": COHERENT_SCENE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]

                raw = self._call_qwen(model, tokenizer, messages, temperature=current_temp)
                batch_caption_map = self._parse_batch_captions(raw, expected_scene_ids)

                # Validate: check both count AND content quality
                if batch_caption_map and len(batch_caption_map) == len(expected_scene_ids):
                    # Optional: check that captions aren't empty/gibberish
                    if all(c and len(c.strip()) > 10 for c in batch_caption_map.values()):
                        print(f"Batch {batch_idx+1} parsed successfully (attempt {retry+1})")
                        break
                
                if retry < max_retries:
                    print(f"Retry {retry+1}/{max_retries} for batch {batch_idx+1} "
                        f"(got {len(batch_caption_map)}/{len(expected_scene_ids)} valid)")
                    time.sleep(2)

            caption_map.update(batch_caption_map)

            # Build list of successfully captioned scenes for context update
            batch_captioned = [
                {"scene_id": sid, "caption": caption_map[sid]}
                for sid in expected_scene_ids
                if sid in caption_map and caption_map[sid]
            ]

            # Update narrative context only if we have good captions
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
        beat_config_json_path = beats_folder / "config.json"
        model, config_data = load_model(self.beat_checkpoint_path, beat_config_json_path)

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
	
        model_path = self.qwen_vl_embedding_model_path
            
        # Initialize the embedding model with bfloat16 for inference efficiency
        embedder = Qwen3VLEmbedder(
            model_name_or_path=model_path,
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


    ## --------------------
    ## 7. Clean up clips
    ## --------------------    
    def _cleanup_clips(self):
        """
        Remove the clips directory for each video to save disk space.
        Only call this after all processing steps that need clips are done.
        """
        deleted_count = 0
        
        for video_dir in self.video_dirs:
            clips_dir = video_dir / self.clips_directory
            
            if clips_dir.exists() and clips_dir.is_dir():
                try:
                    clip_files = list(clips_dir.glob('*.mp4'))
                    audio_files = list(clips_dir.glob('*.wav'))
                    total_files = len(clip_files) + len(audio_files)
                    
                    failed_files = []
                    for file_path in clips_dir.iterdir():
                        if file_path.is_file():
                            try:
                                file_path.unlink()
                            except Exception as e:
                                failed_files.append((file_path.name, str(e)))
                    
                    remaining_files = list(clips_dir.iterdir())
                    if not remaining_files:
                        clips_dir.rmdir()
                        print(f"  [DELETED] {video_dir.name}/clips ({total_files} files removed)")
                        deleted_count += 1
                    else:
                        print(f"  [PARTIAL] {video_dir.name}/clips: deleted {total_files - len(failed_files)}/{total_files} files, {len(remaining_files)} remaining")
                        if failed_files:
                            for fname, err in failed_files[:3]:
                                print(f"    - Failed to delete {fname}: {err}")
                        
                except Exception as e:
                    print(f"  [ERROR] Failed to delete {clips_dir}: {e}")
        
        print(f"Cleanup completed: {deleted_count}/{len(self.video_dirs)} clips directories fully removed")


    ## --------------------
    ## MAIN PIPELINE RUNNING
    ## --------------------    
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

            if self.delete_clips:
                print("\n--- Step 7: Cleanup Temporary Clips ---")
                self._cleanup_clips()

            print(f"\nAll processing steps completed in {time.time() - start_time:.2f} seconds.")

        except Exception as e:
            print(f"\nA critical error occurred during processing: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._clean_memory()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process videos to generate segments, OCR, and captions.")
    parser.add_argument('--root_dir', type=str, default="/content/drive/MyDrive/KhoaLuan/EnTube/Download_2min", help="Directory containing video folders.")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use ('cuda' or 'cpu').")
    parser.add_argument('--limit_videos', type=int, default=0, help="Limit the number of videos to process (0 for all).")

    # Segmentation args
    parser.add_argument('--adaptive_threshold', type=float, default=1.5)
    parser.add_argument('--min_scene_len', type=int, default=15)
    parser.add_argument('--window_width', type=int, default=4)
    parser.add_argument('--min_content_val', type=float, default=6.0)
    parser.add_argument('--min_nodes', type=int, default=3)
    parser.add_argument('--merge_gap', type=float, default=1.0)

    # Skip flags
    parser.add_argument('--skip_segment', action='store_true', help="Skip video segmentation.")
    parser.add_argument('--skip_cut_video', action='store_true', help="Skip cut video into clips.")
    parser.add_argument('--skip_visual_caption', action='store_true', help="Skip visual caption generation.")
    parser.add_argument('--skip_scene_caption', action='store_true', help="Skip scene caption generation.")
    parser.add_argument('--skip_audio', action='store_true', help="Skip audio tagging.")
    parser.add_argument('--skip_scene_embedding', action='store_true', help="Skip scene embedding generation.")
    parser.add_argument('--delete_clips', action='store_true', help="Delete temporary clips directories after processing to save disk space.")
    parser.add_argument('--beat_checkpoint_path', type=str, default='/content/drive/MyDrive/KhoaLuan/models/beats_checkpoints/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')
    parser.add_argument('--qwen_visual_model_path', type=str, default='/content/drive/MyDrive/KhoaLuan/models/Qwen3-VL-2B-Instruct')
    parser.add_argument('--qwen_instruct_model_path', type=str, default='/content/drive/MyDrive/KhoaLuan/models/Qwen3-4B-Instruct-2507')
    parser.add_argument('--qwen_vl_embedding_model_path', type=str, default='/content/drive/MyDrive/KhoaLuan/models/Qwen3-VL-Embedding-2B')
    
    args = parser.parse_args()

    prompter = Prompter(args)
    prompter.run()
