import os
import json
import subprocess
from pathlib import Path
import sys

# Get the directory of the script
script_dir = Path(os.path.abspath(__file__)).parent
base_dir = script_dir.parent

# Add 'modules' to sys.path so we can import internal components
sys.path.append(str(base_dir / "modules"))

from inference_beats import AudioFeatureExtractor

DATASET_DIR = base_dir / "dataset"
CHECKPOINT_FILE = base_dir / "modules" / "checkpoints" / "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
CONFIG_FILE = base_dir / "modules" / "beats" / "config.json"

def process_videos():
    print(f"Loading BEATs model from {CHECKPOINT_FILE}...")
    try:
        # The AudioFeatureExtractor requires the checkpoint and the config json
        extractor = AudioFeatureExtractor(str(CHECKPOINT_FILE), str(CONFIG_FILE))
    except Exception as e:
        print(f"Failed to load BEATs model: {e}")
        return

    # Iterate over directories in dataset
    for video_dir in DATASET_DIR.iterdir():
        if not video_dir.is_dir():
            continue
            
        video_id = video_dir.name
        print(f"\nProcessing {video_id}...")
        
        # Find the video file (mp4 or mkv)
        video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.mkv"))
        if not video_files:
            print(f"No video file found in {video_dir}")
            continue
            
        video_path = video_files[0]
        audio_path = video_dir / "audio_raw.wav"
        json_path = video_dir / "audio_analysis.json"
        
        # 1. Audio Extraction
        print(f"  Extracting audio to {audio_path.name}...")
        # 16kHz, Mono (ac 1), PCM 16-bit (pcm_s16le)
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(audio_path)
        ]
        
        try:
            # Run ffmpeg to extract audio
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  FFmpeg failed for {video_id}: {e}")
            continue
            
        # 2 & 3. Audio Tagging & Classification
        print(f"  Analyzing audio tags and vibes...")
        try:
            outputs = extractor.get_prediction(str(audio_path), top_k=5)
        except Exception as e:
            print(f"  Failed to analyze audio for {video_id}: {e}")
            continue
            
        # Construct output JSON
        result = {
            "video_id": video_id,
            "audio_tags": outputs.get("audio_tags", []),
            "audio_vibes": outputs.get("audio_vibes", [])
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
            
        print(f"  Saved analysis to {json_path.name}")

if __name__ == "__main__":
    process_videos()
