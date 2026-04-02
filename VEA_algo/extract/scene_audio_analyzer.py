import os
import json
import subprocess
from pathlib import Path
import sys

# Cấu trúc lại đường dẫn hệ thống để gọi mô hình BEATs
script_dir = Path(os.path.abspath(__file__)).parent
base_dir = script_dir.parent
sys.path.append(str(script_dir))
sys.path.append(str(base_dir / "modules"))

from inference_beats import AudioFeatureExtractor
# Tận dụng thuật toán phát hiện Scene của PySceneDetect mà project đang xài
from scenedetect import detect, ContentDetector

DATASET_DIR = base_dir / "dataset"
CHECKPOINT_FILE = base_dir / "modules" / "checkpoints" / "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
CONFIG_FILE = base_dir / "modules" / "beats" / "config.json"

def format_timecode(seconds):
    # Hàm con chuyển đổi theo định dạng HH:MM:SS
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def process_scenes():
    print(f"Loading BEATs model from {CHECKPOINT_FILE}...")
    try:
        # Nạp mô hình BEATs cùng config phân loại Vibe/Tag
        extractor = AudioFeatureExtractor(str(CHECKPOINT_FILE), str(CONFIG_FILE))
    except Exception as e:
        print(f"Khởi tạo BEATs model thất bại: {e}")
        return

    # Vòng lặp duyệt các thư mục video nằm trong rễ dataset/
    for video_dir in DATASET_DIR.iterdir():
        if not video_dir.is_dir():
            continue
            
        video_id = video_dir.name
        print(f"\nProcessing video: {video_id}...")
        
        # Tìm lại file nguyên thuỷ (chấp nhận đuôi mp4/mkv)
        video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.mkv"))
        if not video_files:
            print(f"Video không tồn tại ở thư mục này: {video_dir}")
            continue
            
        video_path = video_files[0]
        json_path = video_dir / "scene_audio_results.json"
        
        # Bước 1: Quét logic chuyển cảnh toàn bộ video
        print(f"  Tiến hành phân mảnh (Scene Detection) hình ảnh...")
        scene_list = detect(str(video_path), ContentDetector(threshold=27.0))
        print(f"  Đã quét được {len(scene_list)} cảnh (scenes).")
        
        scenes_data = []
        
        # Bước 2 & 3: Cuốn chiếu vòng phân đoạn - Cắt & Predict
        for i, scene in enumerate(scene_list):
            scene_index = i + 1
            start_sec = scene[0].get_seconds()
            end_sec = scene[1].get_seconds()
            duration = end_sec - start_sec
            
            # Cắt audio chính xác và xuất tạo file trung gian .wav
            temp_wav = video_dir / f"temp_scene_{scene_index}.wav"
            print(f"    - Đoạn Cảnh #{scene_index} (Thời lượng: {duration:.2f}s)...")
            
            # Khởi tạo tiến trình con phân tách Audio (16kHz / Mono)
            cmd = [
                "ffmpeg", "-y", 
                "-ss", str(start_sec), 
                "-t", str(duration), 
                "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                str(temp_wav)
            ]
            
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                
                # Inference nhận diện Tags/Vibes 
                outputs = extractor.get_prediction(str(temp_wav), top_k=3)
                
                # Đóng gói Dictionary
                scenes_data.append({
                    "scene_index": scene_index,
                    "start_time": format_timecode(start_sec),
                    "end_time": format_timecode(end_sec),
                    "audio_tags": outputs.get("audio_tags", []),
                    "audio_vibes": outputs.get("audio_vibes", [])
                })
                
            except subprocess.CalledProcessError as e:
                print(f"      [Lỗi FFMpeg tại Scene {scene_index}]: {e}")
            except Exception as e:
                print(f"      [Lỗi Truy Suất tại Scene {scene_index}]: {e}")
            finally:
                # Ràng buộc kỹ thuật: Xoá tệp tạm ngay lập tức sau khi mô hình trả JSON 
                # Nhằm tiết kiệm phân vùng ổ cứng
                if temp_wav.exists():
                    temp_wav.unlink()
        
        # Ghi Log Dữ Liệu
        result = {
            "video_id": video_id,
            "scenes": scenes_data
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
            
        print(f"  Hoàn tất xuất file: {json_path.name}")

if __name__ == "__main__":
    process_scenes()
