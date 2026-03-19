<div align="center">

# VEA_algo: Video Extraction and Analysis with Qwen3-VL

</div>

<br/>

## 📋 Table of Contents

- [⚡ VEA_algo Framework](#-vea_algo-framework)
- [🛠️ Installation](#️-installation)
- [🚀 Quick Start](#-quick-start)

## ⚡ VEA_algo Framework

VEA_algo is an advanced video processing pipeline that builds on the structural foundations of VideoRAG. It utilizes PySceneDetect to segment videos into coherent semantic scenes, then leverages **Qwen3-VL-2B-Instruct** for dense visual captioning and a robust ASR model (Whisper/SenseVoice) for accurate audio transcription. The scene-by-scene extracted data is then aggregated into a unified chronological corpus for downstream Retrieval-Augmented Generation or deep video analysis.

## 🛠️ Installation

### 📦 Environment Setup

Create a conda environment and install the dependencies:

```bash
conda create --name vea python=3.11
conda activate vea
```

### 📚 Core Dependencies

Install standard libraries and the specific requirements for Qwen3-VL and Scene Detection:

```bash
# Core video processing and vision libraries
pip install torch torchvision torchaudio accelerate
pip install flash-attn --no-build-isolation

# Install latest transformers for Qwen3-VL support and PySceneDetect
pip install git+https://github.com/huggingface/transformers pyscenedetect

# Additional ASR and utilities
pip install faster_whisper moviepy
```

## 🚀 Quick Start

Here is a sample code snippet to initialize the **Qwen3-VL-2B-Instruct** model, incorporating Flash Attention 2 for memory optimization, and the required hyperparameters.

```python
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# 1. Initialize Qwen3-VL with Flash Attention 2
model_id = "Qwen/Qwen2-VL-2B-Instruct" # Note: Qwen3-VL path based on your local/HF state
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_id)

# 2. Hyperparameters configuration
# As provided: Greedy or Sampling (Temperature, Top_p), and specific Presence Penalties
generation_kwargs_vl = {
    "max_new_tokens": 512,
    "temperature": 0.0,        # 0.0 for Greedy decoding
    "top_p": 1.0, 
    "do_sample": False,
    "presence_penalty": 1.5,   # presence_penalty=1.5 for VL
}

generation_kwargs_text = {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "presence_penalty": 2.0,   # presence_penalty=2.0 for Text modules
}
```
