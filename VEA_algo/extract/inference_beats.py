import torch
import torchaudio
import torch.nn.functional as F
import json
import os

# Import các thành phần từ folder 'beats' của Microsoft
from beats.BEATs import BEATs, BEATsConfig

class AudioFeatureExtractor:
    def __init__(self, checkpoint_path, config_json_path=None):
        # Khởi tạo và nạp trọng số cho mô hình BEATs
        print(f"--- Loading model from: {checkpoint_path} ---")
        
        # 1. Tải checkpoint (hỗ trợ cả CPU và GPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        cfg = BEATsConfig(checkpoint['cfg'])
        
        # 2. Xây dựng cấu trúc mô hình từ cấu hình trong checkpoint
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(device)
        self.model.eval()
        
        # 3. Nạp danh mục nhãn AudioSet từ file JSON
        self.config_data = {}
        if config_json_path and os.path.exists(config_json_path):
            with open(config_json_path, mode='r', encoding='utf-8') as f:
                self.config_data = json.load(f)
        else:
            print("Warning: Label configuration file not found. Results will show raw indices.")

    def _preprocess(self, audio_path):
        # Tiền xử lý âm thanh: chuẩn hóa về 16kHz và chuyển sang mono
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # BEATs yêu cầu tần số lấy mẫu cố định là 16.000 Hz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Nếu là âm thanh stereo (2 kênh), lấy trung bình để chuyển về mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        return waveform

    def get_prediction(self, audio_path, top_k=5):
        # Thực hiện trích xuất nhãn âm thanh từ file đầu vào
        waveform = self._preprocess(audio_path)
        device = next(self.model.parameters()).device
        waveform = waveform.to(device)
        
        with torch.no_grad():
            # Trích xuất đặc trưng (features) và tính xác suất các nhãn
            logits = self.model.extract_features(waveform)[0]
            
            # Sử dụng Sigmoid vì AudioSet là bài toán đa nhãn (Multi-label)
            # Tính trung bình xác suất trên toàn bộ chiều thời gian của đoạn audio
            probabilities = torch.sigmoid(logits).mean(dim=0)
            
            # Lấy ra tất cả các nhãn và sắp xếp theo xác suất giảm dần
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
            
        audio_tags = []
        audio_vibes = []
        
        for i in range(len(sorted_indices)):
            # Dừng vòng lặp nếu cả hai danh sách đều đã đạt đủ top_k
            if len(audio_tags) >= top_k and len(audio_vibes) >= top_k:
                break
                
            idx = sorted_indices[i].item()
            conf_score = sorted_probs[i].item()
            str_idx = str(idx)
            
            # Ánh xạ chỉ số (index) sang tên nhãn tiếng Anh và loại (type)
            if self.config_data and str_idx in self.config_data:
                label_info = self.config_data[str_idx]
                label_name = label_info["label"]
                label_type = label_info["type"]
            else:
                label_name = f"Class_{idx}"
                label_type = "tag"  # Mặc định là tag nếu không có cấu hình
            
            result_item = {
                "label": label_name, 
                "confidence": round(conf_score, 4)
            }
            
            if label_type == "vibe":
                if len(audio_vibes) < top_k:
                    audio_vibes.append(result_item)
            else:
                if len(audio_tags) < top_k:
                    audio_tags.append(result_item)
                    
        return {
            "audio_tags": audio_tags,
            "audio_vibes": audio_vibes
        }

# --- ĐOẠN MÃ CHẠY THỬ NGHIỆM ---
if __name__ == "__main__":
    # Cấu hình đường dẫn tệp tin
    CHECKPOINT_FILE = "../modules/checkpoints/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
    CONFIG_FILE = "../modules/beats/config.json"
    INPUT_AUDIO = "sample_scene.wav"

    try:
        # Khởi tạo bộ trích xuất
        extractor = AudioFeatureExtractor(CHECKPOINT_FILE, CONFIG_FILE)
        
        # Chạy dự đoán
        outputs = extractor.get_prediction(INPUT_AUDIO, top_k=5)

        print(f"\n--- Extraction Results for: {INPUT_AUDIO} ---")
        print("Audio Tags:")
        for item in outputs["audio_tags"]:
            print(f"  - {item['label']} (Confidence: {item['confidence']})")
            
        print("\nAudio Vibes:")
        for item in outputs["audio_vibes"]:
            print(f"  - {item['label']} (Confidence: {item['confidence']})")
            
    except Exception as error:
        print(f"An error occurred during execution: {error}")