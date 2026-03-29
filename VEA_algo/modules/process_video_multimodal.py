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
import torch
from scenedetect import AdaptiveDetector

# Add InternVideo to system path for retrieval models
from segment import scene_detection, segment_with_stt_timestamp


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
        self.caption_filename = 'caption_corpus.pt'
        self.segmentation_filename = 'segments.json'

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