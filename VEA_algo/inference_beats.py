import torch
import torchaudio
import torch.nn.functional as F
import pandas as pd
import os

# Import các thành phần từ folder 'beats' của Microsoft
from beats.BEATs import BEATs, BEATsConfig

class AudioFeatureExtractor:
    def __init__(self, checkpoint_path, label_csv_path=None):
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
        
        # 3. Nạp danh mục nhãn AudioSet từ file CSV
        self.label_list = None
        if label_csv_path and os.path.exists(label_csv_path):
            df = pd.read_csv(label_csv_path)
            self.label_list = df['display_name'].tolist()
        else:
            print("Warning: Label mapping file not found. Results will show raw indices.")

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
            
            # Lấy ra K nhãn có xác suất cao nhất
            top_probs, top_indices = torch.topk(probabilities, k=top_k)
            
        inference_results = []
        for i in range(len(top_indices)):
            idx = top_indices[i].item()
            conf_score = top_probs[i].item()
            
            # Ánh xạ chỉ số (index) sang tên nhãn tiếng Anh
            label_name = self.label_list[idx] if self.label_list else f"Class_{idx}"
            
            inference_results.append({
                "label": label_name, 
                "confidence": round(conf_score, 4)
            })
            
        return inference_results

# --- ĐOẠN MÃ CHẠY THỬ NGHIỆM ---
if __name__ == "__main__":
    # Cấu hình đường dẫn tệp tin
    CHECKPOINT_FILE = "./checkpoints/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
    LABEL_MAPPING_FILE = "./class_labels_indices.csv"
    INPUT_AUDIO = "sample_scene.wav"

    try:
        # Khởi tạo bộ trích xuất
        extractor = AudioFeatureExtractor(CHECKPOINT_FILE, LABEL_MAPPING_FILE)
        
        # Chạy dự đoán
        outputs = extractor.get_prediction(INPUT_AUDIO, top_k=10)

        print(f"\n--- Extraction Results for: {INPUT_AUDIO} ---")
        for item in outputs:
            print(f"Tag/Vibe: {item['label']} | Confidence: {item['confidence']}")
            
    except Exception as error:
        print(f"An error occurred during execution: {error}")