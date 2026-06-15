<div align="center">

# VEA_algo: Video Extraction and Analysis with Qwen3-VL

</div>

<br/>

## 📋 Table of Contents

- [⚡ VEA_algo Framework](#-vea_algo-framework)

## ⚡ VEA_algo Framework

VEA_algo is an advanced video processing pipeline that builds on the structural foundations of VideoRAG. It utilizes PySceneDetect to segment videos into coherent semantic scenes, then leverages **Qwen3-VL-2B-Instruct** for dense visual captioning and a robust ASR model (Whisper/SenseVoice) for accurate audio transcription. The scene-by-scene extracted data is then aggregated into a unified chronological corpus for downstream Retrieval-Augmented Generation or deep video analysis.


## Prerequisites

```bash
pip install -r requirements.txt
```

---

## 1. Environment Setup

Create a `.env` file in the project root:

```
HUGGING_FACE_TOKEN=hf_...
PYANNOTEAI_API_KEY=sk_...
OPENAI_API_KEY=sk-...
```

> `HUGGING_FACE_TOKEN` — required for Whisper alignment model and pyannote diarization  
> `PYANNOTEAI_API_KEY` — required only if using `--diarization_model precision-2`  
> `OPENAI_API_KEY` — required for scene caption generation (Step 4)

---

## 2. Dataset Structure

Prepare your dataset folder as follows:

```
dataset/
  video_01/
    video.mp4
  video_02/
    video.mp4
  video_03/
    video.mp4
  edu_dataset.json        ← will be generated in Part 2, Step 1
```

Each video must be in its **own subfolder**. The folder name becomes the `doc_id` used throughout the pipeline.

---

## 3. Clone RST Parser

```bash
cd VEA_algo/rst_tree_parsing
git clone https://github.com/thnndat236/RSTParser_EACL24.git
```

---

## Part 1 — Audio, Transcription & Visual Analysis

### Step 1: Extract audio + transcribe + speaker diarization

```bash
python multimodal_to_text/extract_audio_transcribe_diarization.py \
    --root_dir dataset/ \
    --diarization_model community-1
```

Options:
- `--diarization_model community-1` — uses `pyannote/speaker-diarization-community-1` (free, requires `HUGGING_FACE_TOKEN`)
- `--diarization_model precision-2` — uses `pyannote/speaker-diarization-precision-2` (paid, requires `PYANNOTEAI_API_KEY`)
- `--skip_audio` — skip audio extraction (if `audio.wav` already exists)
- `--skip_separate` — skip vocal separation via demucs
- `--skip_whisper` — skip transcription

Output per video folder:
```
video_01/
  audio.wav
  demucs_out/htdemucs/audio/vocals.wav
  audio.json        ← transcript with speaker labels
```

### Step 2: Scene detection + visual analysis + scene captions

```bash
python multimodal_to_text/process_video_multimodal.py \
    --root_dir dataset/
```

Options:
- `--limit_videos N` — process only the first N videos (useful for testing)
- `--skip_segment` — skip scene detection
- `--skip_cut_video` — skip cutting video into clips
- `--skip_visual_caption` — skip Qwen3-VL visual analysis
- `--skip_scene_caption` — skip OpenAI scene caption generation
- `--skip_audio_tagging` - skip BEATs audio tagging

Output per video folder:
```
video_01/
  segments.json     ← scenes with transcript, visual_description, visual_elements, caption
  clips/
    clip_000_0.03-5.50.mp4
    clip_001_6.33-7.25.mp4
    ...
```

---

## Part 2 — RST Tree Parsing

### Step 1: Prepare EDU dataset

```bash
python rst_tree_parsing/prepare_edus.py \
    --root_dir dataset/ \
    --output_json dataset/edu_dataset.json
```

Output `dataset/edu_dataset.json`:
```json
[
  {"doc_id": "video_01", "edu_strings": ["caption scene 0", "caption scene 1", ...], "video_dir": "dataset/video_01"},
  {"doc_id": "video_02", "edu_strings": [...], "video_dir": "dataset/video_02"}
]
```

> Videos with fewer than 2 captions are skipped automatically.

### Step 2: Parse RST trees

```bash
python rst_tree_parsing/rst_tree_parsing.py \
    --model_size 7b \
    --corpus rstdt \
    --parse_type bottom_up \
    --dataset_file dataset/edu_dataset.json
```

Options:
- `--parse_type bottom_up` or `--top_down`
- `--rel_type rel_with_nuc` (default) or `rel` or `nuc_rel`

Output per video folder:
```
video_01/
  rst_tree.tree     ← RST tree for this video
```

---

## Full Pipeline (single command sequence)

```bash
# Part 1
python multimodal_to_text/extract_audio_transcribe_diarization.py --root_dir dataset/
python multimodal_to_text/process_video_multimodal.py --root_dir dataset/

# Part 2
## Hướng 1: Sử dụng LLM để tạo graph với 9 rhetorical relations theo Video Discourse
python building_graph/caption_to_graph.py \
  --root_dir dataset/ \
  --force

## Hướng 2: Tạo RST Tree để convert thành Dependency Tree
python rst_tree_parsing/prepare_edus.py \
    --root_dir dataset/ \
    --output_json dataset/edu_dataset.json

python rst_tree_parsing/rst_tree_parsing.py \
    --model_size 7b \
    --corpus rstdt \
    --dataset_file dataset/edu_dataset.json
```
