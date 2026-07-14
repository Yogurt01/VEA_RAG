import json
import sys
import os
from pathlib import Path
import gc
import torch
import torchaudio
import torch.nn.functional as F

from BEATs import BEATs, BEATsConfig


def load_model(checkpoint_path: str, config_json_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = BEATsConfig(checkpoint['cfg'])
    
    # Build model architecture from the configuration in the checkpoint
    model = BEATs(cfg)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    # Load AudioSet label categories from a JSON file
    config_data = {}
    if config_json_path and os.path.exists(config_json_path):
        with open(config_json_path, mode='r', encoding='utf-8') as f:
            config_data = json.load(f)
    else:
        raise FileNotFoundError("Warning: Label configuration file not found. Results will show raw indices.")
    
    return model, config_data

def _preprocess(audio_path):
    # Audio preprocessing: normalize to 16kHz and convert to mono
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # BEATs requires a fixed sampling rate of 16,000 Hz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # If audio is stereo (2 channels), calculate the mean to convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    return waveform

# def audio_tagging(audio_path, model, config_data, top_k=5):
#     # Extract audio labels from the input file
#     waveform = _preprocess(audio_path)
#     device = next(model.parameters()).device
#     waveform = waveform.to(device)
    
#     with torch.no_grad():
#         # Extract features and calculate label probabilities
#         logits = model.extract_features(waveform)[0]
        
#         # Use Sigmoid as AudioSet is a multi-label problem
#         # Calculate mean probability across the entire time dimension of the audio segment
#         probabilities = torch.sigmoid(logits).mean(dim=0)
        
#         # Retrieve all labels and sort by descending probability
#         sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
        
#     audio_tags = []
#     audio_vibes = []
    
#     for i in range(len(sorted_indices)):
#         # Stop the loop if both lists have reached top_k
#         if len(audio_tags) >= top_k and len(audio_vibes) >= top_k:
#             break
            
#         idx = sorted_indices[i].item()
#         conf_score = sorted_probs[i].item()
#         str_idx = str(idx)
        
#         # Map index to label name and type
#         if config_data and str_idx in config_data:
#             label_info = config_data[str_idx]
#             label_name = label_info["label"]
#             label_type = label_info["type"]
#         else:
#             label_name = f"Class_{idx}"
#             label_type = "tag"  # Default to tag if no configuration exists
        
#         result_item = {
#             "label": label_name, 
#             "confidence": round(conf_score, 4)
#         }
        
#         if label_type == "vibe":
#             if len(audio_vibes) < top_k:
#                 audio_vibes.append(result_item)
#         else:
#             if len(audio_tags) < top_k:
#                 audio_tags.append(result_item)
                
#     return {
#         "audio_tags": audio_tags,
#         "audio_vibes": audio_vibes
#     }

def audio_tagging(audio_path: str, model, config_data: dict, top_k: int = 5) -> dict:
    device = next(model.parameters()).device

    # Load audio — already 16kHz mono from ffmpeg extraction
    waveform, sample_rate = torchaudio.load(audio_path)

    # Safety check in case the file was not extracted at 16kHz
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

    # Safety check in case the file is stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.to(device)

    with torch.no_grad():
        output = model.extract_features(waveform)
        logits = output[0] if isinstance(output, tuple) else output
        probabilities = torch.sigmoid(logits).mean(dim=0)
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)

    audio_tags  = []
    audio_vibes = []

    for i in range(len(sorted_indices)):
        if len(audio_tags) >= top_k and len(audio_vibes) >= top_k:
            break

        idx       = sorted_indices[i].item()
        conf      = sorted_probs[i].item()
        str_idx   = str(idx)

        if str_idx in config_data:
            label_name = config_data[str_idx]["label"]
            label_type = config_data[str_idx]["type"]
        else:
            label_name = f"Class_{idx}"
            label_type = "tag"

        item = {"label": label_name, "confidence": round(conf, 4)}

        if label_type == "vibe" and len(audio_vibes) < top_k:
            audio_vibes.append(item)
        elif label_type != "vibe" and len(audio_tags) < top_k:
            audio_tags.append(item)

    return {"audio_tags": audio_tags, "audio_vibes": audio_vibes}