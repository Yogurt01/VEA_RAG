import argparse
import json
import logging
import os
import subprocess
import sys
import time
import gc
from multiprocessing import Pool
from pathlib import Path
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
from tqdm import tqdm

os.environ["HF_HOME"] = "/content/drive/MyDrive/KhoaLuan/models/huggingface_cache"
os.environ["TORCH_HOME"] = "/content/drive/MyDrive/KhoaLuan/models/torch_cache"
os.environ["HF_HUB_OFFLINE"] = "1"


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)

def extract_audio_worker(video_path: Path) -> None:
    """Extract audio from a single video file."""
    video_dir = video_path.parent
    audio_path = video_dir / "audio.wav"
    
    if not video_path.exists() or audio_path.exists():
        return

    logging.info(f"Extracting audio from: {video_path.name}")
    
    command = [
        'ffmpeg', '-y', '-i', str(video_path),
        '-vn', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000',
        '-loglevel', 'error', str(audio_path)
    ]

    try:
        subprocess.run(command, check=True, capture_output=True)
        logging.info(f"Successfully extracted: {audio_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg failed for {video_path.name}")
        if audio_path.exists():
            audio_path.unlink()

class VideoProcessor:
    """Simple video processing pipeline."""
    def __init__(self, args: argparse.Namespace):
        self.root_dir = Path(args.root_dir)
        self.num_workers = args.num_workers
        self.args = args
        self.whisper_model_path = args.whisper_model_path
        self.diarization_model_path = args.diarization_model_path
        self.device = args.device
        self.batch_size = 16
        self.compute_type = "float16"
        
        if not self.root_dir.exists():
            self.root_dir.mkdir(parents=True, exist_ok=True)

    def run_pipeline(self):
        """Execute the full processing pipeline."""
        start_time = time.time()
        logging.info("Starting video processing pipeline")

        if not self.args.skip_audio:
            self.extract_audios()

        if not self.args.skip_whisper:
            self.run_speech_to_text()

        total_time = time.time() - start_time
        logging.info(f"Pipeline completed in {total_time:.2f} seconds")

    def extract_audios(self):
        """Extract audio from all videos in parallel."""
        logging.info("Extracting audio from videos")
        video_paths = list(self.root_dir.rglob('*.mp4'))
        
        if not video_paths:
            logging.warning("No .mp4 videos found")
            return

        logging.info(f"Found {len(video_paths)} videos")
        with Pool(self.num_workers) as p:
            list(tqdm(p.imap(extract_audio_worker, video_paths),
                     total=len(video_paths), desc="Extracting Audio"))

    def run_speech_to_text(self):
        """Transcribe audio files using Whisper."""
        logging.info("Running speech-to-text")
        video_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        
        if not video_dirs:
            logging.warning("No video directories found")
            return

        model = None
        diarize_model = None

        try:
            logging.info("Loading Whisper model 'large-v2'")
            model_target = self.whisper_model_path
            
            if model_target != "large-v2" and not Path(model_target).exists():
                logging.warning(
                    f"Local path '{model_target}' was not found. "
                    f"Automatically switching to online mode to fetch/utilize 'large-v2'."
                )
                model_target = "large-v2"
            
            logging.info(f"Loading Whisper model from: {model_target}")
            model = whisperx.load_model(model_target, self.device, compute_type=self.compute_type)
            
            logging.info(f"Loading local Diarization model from: {self.diarization_model_path}")
            if not Path(self.diarization_model_path).exists():
                raise FileNotFoundError(f"Local diarization model not found at: {self.diarization_model_path}")

            diarize_model = DiarizationPipeline(
                model_name=self.diarization_model_path,
                device=self.device,
            )
            
            for video_dir in tqdm(video_dirs, desc="Transcribing Audio"):
                audio_path = video_dir / 'audio.wav'
                json_path = video_dir / 'audio.json'
    
                if json_path.exists() or not audio_path.exists():
                    continue

                try:
                    logging.info(f"Transcribing: {audio_path}")

                    # 1. Transcribe with original whisper (batched)
                    audio = whisperx.load_audio(audio_path)
                    result = model.transcribe(audio, batch_size=self.batch_size)
                    
                    # 2. Align whisper output
                    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
                    result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
                    del model_a
                    torch.cuda.empty_cache()

                    # 3. Assign speaker labels
                    diarize_segments = diarize_model(audio)
                    result = whisperx.assign_word_speakers(diarize_segments, result)

                    with json_path.open('w', encoding='utf-8') as f:
                        json.dump(result["segments"], f, indent=2, ensure_ascii=False)
                        
                except Exception as e:
                    logging.error(f"Error transcribing {audio_path}: {e}")
        finally:
            del model, diarize_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            

def main():
    """Parse arguments and run the video processing pipeline."""
    parser = argparse.ArgumentParser(description='Extract audio and transcribe videos')
    
    parser.add_argument('--root_dir', type=str, default="/content/drive/MyDrive/KhoaLuan/EnTube/Download_2min",
                       help='Root folder containing video subdirectories')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use ('cuda' or 'cpu').")
    parser.add_argument('--num_workers', type=int, default=os.cpu_count() or 1,
                       help='Number of worker processes for parallel processing')
    parser.add_argument('--whisper_model_path', type=str, default='/content/drive/MyDrive/KhoaLuan/models/faster-whisper-large-v2',
                        help='Tên mô hình Whisper hoặc Đường dẫn (Path) tới mô hình lưu tại local')
    parser.add_argument('--diarization_model_path', type=str, default='/content/drive/MyDrive/KhoaLuan/models/speaker-diarization-community-1',
                        help='Tên mô hình Pyannote Community 1 hoặc Đường dẫn (Path) tới mô hình lưu tại local')
    parser.add_argument('--skip_audio', action='store_true',
                       help='Skip audio extraction step')
    parser.add_argument('--skip_whisper', action='store_true',
                       help='Skip speech-to-text step')

    args = parser.parse_args()
    
    try:
        processor = VideoProcessor(args)
        processor.run_pipeline()
    except Exception as e:
        logging.critical(f"Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
