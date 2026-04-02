<div align="center">

# VEA_algo: Video Extraction and Analysis (Updated Architecture)

</div>

<br/>

## 📋 Table of Contents

- [⚡ Giới thiệu sơ lược](#-giới-thiệu-sơ-lược)
- [📂 Cấu trúc thư mục hiện tại](#-cấu-trúc-thư-mục-hiện-tại)
- [🚀 Hướng dẫn thực thi: Pipeline Trích xuất Âm thanh](#-hướng-dẫn-thực-thi-pipeline-trích-xuất-âm-thanh)
- [🧩 Chi tiết về Phân tích Tag & Vibe (BEATs)](#-chi-tiết-về-phân-tích-tag--vibe-beats)

## ⚡ Giới thiệu sơ lược

**VEA_algo** đã được nâng cấp cấu trúc thư mục nhằm tăng tính mô-đun hoá (modularity). Pipeline hiện tại không chỉ phân đoạn video (scene detection), nhận diện hình ảnh (với Qwen3-VL) và bóc băng văn bản, mà còn được tích hợp thêm mô hình **BEATs** để phân tích và gán nhãn sự kiện âm thanh thực tế (Audio Tags) và bầu không khí (Audio Vibes) trực tiếp từ video.

## 📂 Cấu trúc thư mục hiện tại

Với sự sắp xếp lại, thư mục tập trung chia thành các mảng chức năng rõ ràng:

```text
VEA_algo/
├── dataset/                     # Thư mục chứa video phân theo từng folder
│   ├── video_01/
│   │   ├── video1.mp4           # File video gốc
│   │   ├── audio_raw.wav        # Audio 16kHz PCM (Tự động tách ra)
│   │   └── audio_analysis.json  # Kết quả Audio Tags và Vibes (Top 5)
│   └── video_02/                ...
├── extract/                     # Các kịch bản chạy pipeline chính
│   ├── process_audio.py         # Kịch bản chính: Tách Audio & Phân loại BEATs
│   ├── inference_beats.py       # Lớp lõi gọi mô hình BEATs
│   ├── process_video_multimodal.py
│   ├── extract_audio_transcribe_diarization.py
│   ├── scene_detector.py
│   └── segment.py
├── modules/                     # Chứa các mô hình phân tích sâu/chuyên biệt
│   ├── beats/                   # Module đánh giá Vibe & Tag bằng BEATs
│   │   ├── config.json          # Bộ từ điển phân định Vibe / Tag
│   │   └── BEATs.py             # Kiến trúc mạng chính
│   ├── checkpoints/             # Nơi lưu trữ weights của các module (BEATs pt file)
│   ├── ast/                     ...
│   ├── rst/                     ...
│   └── vlm/                     ...
└── requirements.txt
```

---

## 🚀 Hướng dẫn thực thi: Pipeline Trích xuất Âm thanh

Một module quan trọng mới được bổ sung vào workflow là **`process_audio.py`**. Kịch bản tiến hành rà soát hàng loạt các thư mục trong `dataset/` nhằm:
1. Dùng `ffmpeg` trích xuất track âm thanh chuẩn hoá (16 kHz, Mono) thành file `audio_raw.wav`.
2. Truyền Audio vào **AudioFeatureExtractor** thông qua mô hình BEATs.
3. Xuất kết quả phân loại thành file JSON độc lập `audio_analysis.json` nằm trong từng folder video tương ứng.

### Chạy tiến trình

Trước khi chạy, hãy khởi động môi trường ảo của bạn (Ví dụ: `conda activate vea-rag`) do bước này yêu cầu cài đặt đầy đủ package `torch` và `torchaudio`.

Từ thư mục gốc `VEA_algo`, bạn gọi trực tiếp file thực thi trong thư mục `extract`:
```bash
python3 extract/process_audio.py
```
> **Lưu ý**: Lệnh này tự động nhận dạng thư mục gốc của script, dẫn chính xác tới `checkpoints` và `config.json` bên trong `modules/`.

### Kết quả ví dụ (Mẫu xuất ra JSON)
Tệp `dataset/video_01/audio_analysis.json` sẽ chứa danh sách các thể loại âm thanh nổi bật nhất:
```json
{
    "video_id": "video_01",
    "audio_tags": [
        {
            "label": "Speech",
            "confidence": 0.8123
        },
        ...
    ],
    "audio_vibes": [
        {
            "label": "Inside, small room",
            "confidence": 0.5109
        },
        ...
    ]
}
```

---

## 🧩 Chi tiết về Phân tích Tag & Vibe (BEATs)

Dự án phân biệt rõ hai khái niệm thông qua tệp từ điển `modules/beats/config.json`:
- **Audio Tags**: Những nhãn đại diện cho con người, động vật, máy móc, sự kiện và hiện tượng vật lý (Ví dụ: *Speech, Bell, Bark, Knock*).
- **Audio Vibes**: Được quy ước cho thể loại âm nhạc, cảm xúc/tâm trạng của âm thanh, hoặc tiếng ồn môi trường tổng quan (Ví dụ: *Sad, Inside/Outside, Noise, Environmental, Soundtrack*).

Khâu trích xuất sẽ sắp xếp độ chắc chắn (confidence) từ cao xuống thấp và dừng vòng lặp ngay khi lấy đủ **Top 5** cho mỗi phân loại. Điều này rất hữu ích cho hệ thống RAG (Retrieval-Augmented Generation) để ngữ cảnh hoá các đoạn scene mô tả bầu không khí trong video.
