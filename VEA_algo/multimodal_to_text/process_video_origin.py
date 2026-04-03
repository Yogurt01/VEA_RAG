import argparse
import gc
import json
import os
import sys
import time
import unicodedata
from glob import glob
from pathlib import Path

import cv2
import easyocr
import numpy as np
import torch
from decord import VideoReader, cpu
from scenedetect import AdaptiveDetector
from tqdm import tqdm

# Add InternVideo to system path for retrieval models
sys.path.append('InternVideo/Downstream/Video-Text-Retrieval')
from inference import caption_retrieval
from run_blip import caption_video, generate_captions, load_model
from segment import scene_detection, segment_with_stt_timestamp

# Global OCR reader instance to avoid reloading the model
OCR_READER = None

def initialize_ocr_reader(use_gpu=None):
    """
    Initializes the EasyOCR reader if it hasn't been already.
    Supports both Vietnamese and English.
    """
    global OCR_READER
    if OCR_READER is None:
        try:
            if use_gpu is None:
                use_gpu = torch.cuda.is_available()
            OCR_READER = easyocr.Reader(['vi', 'en'], gpu=use_gpu)
            print("EasyOCR reader initialized successfully.")
        except Exception as e:
            print(f"Error initializing EasyOCR: {e}")
            OCR_READER = None
    return OCR_READER

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

def perform_ocr_on_frame(image_path):
    """
    Extracts text from a single image frame using EasyOCR.
    """
    try:
        ocr_reader = initialize_ocr_reader()
        if not ocr_reader:
            return []

        img = cv2.imread(str(image_path))
        if img is None:
            return []

        results = ocr_reader.readtext(img)
        
        # Filter results with low confidence
        texts = [
            normalize_vietnamese_text(text)
            for _, text, prob in results
            if prob > 0.2 and text.strip()
        ]
        return texts
    except Exception as e:
        print(f"Error during OCR on frame: {e}")
        return []

def extract_frame_from_video(video_path, output_path, frame_number=0):
    """
    Extracts a specific frame from a video and saves it as an image file.
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return False

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"Cannot read frame {frame_number} from video: {video_path}")
            return False

        cv2.imwrite(str(output_path), frame)
        return True
    except Exception as e:
        print(f"Error extracting frame from video: {e}")
        return False

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
        self.caption_filename = 'caption_corpus.pt'
        self.segmentation_filename = 'segments.json'
        self.audcap_filename = 'audtag.pt'

        # Control flags for processing steps
        self.skip_segment = args.skip_segment
        self.skip_ocr = args.skip_ocr
        self.skip_diarization = args.skip_diarization
        self.skip_caption = args.skip_caption
        self.skip_vcap = args.skip_vcap
        self.skip_audio = args.skip_audio
        
        # Get all video directory paths, optionally limited
        all_video_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        limit = getattr(args, 'limit_videos', None)
        self.video_dirs = all_video_dirs[:limit] if limit and limit > 0 else all_video_dirs
        
        self.videos = []
        for vdir in self.video_dirs:
            mp4_files = list(vdir.glob('*.mp4'))
            if mp4_files:
                self.videos.append(mp4_files[0])

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
                        seg.setdefault('ocr_text', [])
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
    ## 2. OCR on Segments
    ## --------------------
    def _ocr_videos(self):
        print("Starting OCR processing...")
        ocr_reader = initialize_ocr_reader()
        if not ocr_reader:
            print("Cannot initialize EasyOCR, skipping OCR.")
            return

        for video_dir, video_path in zip(self.video_dirs, self.videos):
            segment_file = video_dir / self.segmentation_filename
            if not segment_file.exists():
                continue

            with open(segment_file, 'r', encoding='utf-8') as f:
                segments = json.load(f)
            
            # Check if OCR already done
            if all('ocr_text' in s and s['ocr_text'] for s in segments):
                continue

            print(f"Processing OCR for {video_dir.name}...")
            self._process_ocr(video_path, segments, ocr_reader)
            
            with open(segment_file, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2, ensure_ascii=False)

    def _process_ocr(self, video_path, segments, ocr_reader):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0  # Default FPS
        
        for segment in segments:
            try:
                # Safely extract start time
                start_time = segment.get('start', 0)
                if isinstance(start_time, (list, tuple)) and len(start_time) > 0:
                    start_time = start_time[0]
                elif isinstance(start_time, str):
                    start_time = float(start_time) if start_time.replace('.', '').isdigit() else 0
                else:
                    start_time = float(start_time) if start_time else 0
                    
                frame_number = int(start_time * fps)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()

                if ret:
                    results = ocr_reader.readtext(frame)
                    ocr_texts = [text for _, text, prob in results if prob > 0.2 and text.strip()]
                    segment['ocr_text'] = ocr_texts
                else:
                    segment['ocr_text'] = []
                    
            except Exception as e:
                print(f"Error processing OCR for segment: {e}")
                segment['ocr_text'] = []

        cap.release()

    ## -----------------------
    ## 3. Visual Description
    ## -----------------------
    def _generate_caption_corpus(self):
        try:
            print("Loading BLIP caption model...")
            model, vis_processors, device = load_model()
            print(f"Model loaded on {device}")
        except Exception as e:
            print(f"Failed to load captioning model: {e}")
            return
            
        videos_to_process = [
            (v_dir, v_path) for v_dir, v_path in zip(self.video_dirs, self.videos)
            if not (v_dir / self.caption_filename).exists()
        ]

        if not videos_to_process:
            print("All videos already have a caption corpus. Skipping generation.")
            return
            
        print(f"Generating captions for {len(videos_to_process)} videos...")

        for video_dir, video_path in tqdm(videos_to_process, desc="Generating Captions"):
            out_path = video_dir / self.caption_filename
            try:
                result = caption_video(str(video_path), model, vis_processors, device)
                
                if result:
                    captions = result['captions']
                    torch.save(captions, out_path)
                    print(f"Generated captions for {video_dir.name}")
                else:
                    torch.save([f"Error: Could not process video"], out_path)
                    
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) or "CUBLAS_STATUS_EXECUTION_FAILED" in str(e):
                    print(f"CUDA error for {video_dir.name}, switching to CPU...")
                    try:
                        # Move model to CPU
                        model = model.to('cpu')
                        device = 'cpu'
                        
                        # Try again with CPU
                        result = caption_video(str(video_path), model, vis_processors, device)
                        if result:
                            captions = result['captions']
                            torch.save(captions, out_path)
                            print(f"Generated captions on CPU for {video_dir.name}")
                        else:
                            torch.save([f"Error: Could not process video on CPU"], out_path)
                    except Exception as cpu_e:
                        print(f"CPU fallback also failed for {video_dir.name}: {cpu_e}")
                        torch.save([f"Error: {cpu_e}"], out_path)
                else:
                    print(f"Runtime error for {video_dir.name}: {e}")
                    torch.save([f"Error: {e}"], out_path)
            except Exception as e:
                print(f"Error generating captions for {video_dir.name}: {e}")
                torch.save([f"Error: {e}"], out_path)
            finally:
                self._clean_memory()


    ## --------------------
    ## 4. Speech Diarization
    ## --------------------
    def _speaker_diarization(self):
        try:
            from diarization import run_diarization
            print("Running speaker diarization...")
            run_diarization(str(self.root_dir), self.segmentation_filename)
            
            # Post-process to normalize text
            for video_dir in self.video_dirs:
                segment_file = video_dir / self.segmentation_filename
                if segment_file.exists():
                    with open(segment_file, 'r+', encoding='utf-8') as f:
                        segments = json.load(f)
                        modified = False
                        for seg in segments:
                            if 'text' in seg:
                                seg['text'] = [normalize_vietnamese_text(t) for t in seg['text']] if isinstance(seg['text'], list) else normalize_vietnamese_text(seg['text'])
                                modified = True
                        if modified:
                            f.seek(0)
                            json.dump(segments, f, indent=2, ensure_ascii=False)
                            f.truncate()
        except ImportError:
            print("Diarization module not found, skipping.")
        except Exception as e:
            print(f"Error during speaker diarization: {e}")
        finally:
            self._clean_memory()
            
    ## --------------------
    ## 5. Audio Tagging
    ## --------------------
    def _audio_tagging(self):
        try:
            sys.path.append('EfficientAT')
            from infer_custom import audio_tagging
            print("Running audio tagging...")
            
            audio_files_exist = any(list(self.root_dir.glob('*/audio.wav')))
            
            if not audio_files_exist:
                print("No audio.wav files found. Creating empty audio tags.")
                for vdir in self.video_dirs:
                    torch.save([], vdir / self.audcap_filename)
                return

            audio_tagging(str(self.root_dir), self.audcap_filename)

        except ImportError:
            print("Audio tagging module not found, skipping.")
        except Exception as e:
            print(f"Error during audio tagging: {e}")
        finally:
            if 'EfficientAT' in sys.path:
                sys.path.remove('EfficientAT')
            self._clean_memory()

    ## ------------------------------
    ## 6. Select Final Frame Captions
    ## ------------------------------
    def _select_frame_captions(self):
        print("Selecting final captions for segments...")
        try:
            for video_dir in self.video_dirs:
                segment_file = video_dir / self.segmentation_filename
                caption_file = video_dir / self.caption_filename
                
                if not segment_file.exists() or not caption_file.exists():
                    print(f"Missing segment or caption file for {video_dir.name}, skipping.")
                    continue
                
                try:
                    with open(segment_file, 'r', encoding='utf-8') as f:
                        segments = json.load(f)
                    
                    # Process only if 'vcap' is missing or erroneous
                    needs_processing = any('vcap' not in s or 'Error' in str(s.get('vcap')) for s in segments)
                    if not needs_processing:
                        print(f"Valid captions already exist for {video_dir.name}, skipping.")
                        continue
                        
                    print(f"Retrieving best captions for {video_dir.name}...")
                    caption_retrieval(str(video_dir), self.caption_filename, self.segmentation_filename)
                
                except Exception as e:
                    print(f"Error during caption selection for {video_dir.name}: {e}")
        
        except Exception as e:
            print(f"An error occurred in the caption selection process: {e}")
        finally:
            self._clean_memory()

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
            
            if not self.skip_ocr:
                print("\n--- Step 2: OCR on Segments ---")
                self._ocr_videos()
            
            if not self.skip_diarization:
                print("\n--- Step 3: Speaker Diarization ---")
                self._speaker_diarization()
            
            if not self.skip_caption:
                print("\n--- Step 4: Visual Caption Generation ---")
                self._generate_caption_corpus()
                
            if not self.skip_vcap:
                print("\n--- Step 5: Final Caption Selection ---")
                self._select_frame_captions()
            
            if not self.skip_audio:
                print("\n--- Step 6: Audio Tagging ---")
                self._audio_tagging()

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
    parser.add_argument('--skip_ocr', action='store_true', help="Skip OCR processing.")
    parser.add_argument('--skip_diarization', action='store_true', help="Skip speaker diarization.")
    parser.add_argument('--skip_caption', action='store_true', help="Skip visual caption generation.")
    parser.add_argument('--skip_vcap', action='store_true', help="Skip final caption selection.")
    parser.add_argument('--skip_audio', action='store_true', help="Skip audio tagging.")

    args = parser.parse_args()

    prompter = Prompter(args)
    prompter.run()