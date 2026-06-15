import json
import re
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.utils.import_utils import is_flash_attn_2_available
from qwen_vl_utils import process_vision_info
from decord import VideoReader, cpu
from PIL import Image

def get_device() -> str:
    """Return the best available device for inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Qwen3VLStandalone:
    """
    Standalone wrapper for Qwen3-VL video inference.
    """
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-VL-2B-Instruct",
        total_pixels: int = 1 * 1024 * 32 * 32,
        min_pixels: int = 64 * 32 * 32,
        max_frames: int = 32,
        sample_fps: float = 0.5,
        image_patch_size: int = 16,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
    ):
        self.model_path = model_path
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        self.total_pixels = total_pixels
        self.min_pixels = min_pixels
        self.max_frames = max_frames
        self.sample_fps = sample_fps
        self.image_patch_size = image_patch_size
        
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        
        self._processor = None
        self._model = None
    
    def load_model(self):
        """Load model and processor from HuggingFace."""
        if self._model is not None:
            return
        
        print(f"Loading model from {self.model_path}")
        
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        
        model_kwargs = {"dtype": "auto", "device_map": self.device}
        if self.device == "cuda" and is_flash_attn_2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using Flash Attention 2")
        
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_path, **model_kwargs
        ).eval()
        print("Model loaded successfully")
    
    def _build_message(self, prompt: str, video_path: str) -> List[Dict]:
        """Build message structure for Qwen3-VL inference."""
        return [{
            "role": "user",
            "content": [
                {
                    "video": video_path,
                    "total_pixels": self.total_pixels,
                    "min_pixels": self.min_pixels,
                    "max_frames": self.max_frames,
                    "sample_fps": self.sample_fps
                },
                {"type": "text", "text": prompt}
            ]
        }]
    
    def _prepare_inputs(self, messages: List[Dict]) -> Dict:
        """Prepare processor inputs from messages."""
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages],
            return_video_kwargs=True,
            image_patch_size=self.image_patch_size,
            return_video_metadata=True
        )
        
        if video_inputs:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None
        
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            **video_kwargs,
            do_resize=False,
            return_tensors="pt"
        ).to(self.device)
        
        return inputs
    
    def _extract_json(self, text: str) -> Optional[Union[Dict, List]]:
        """Extract JSON from model output text."""
        json_match = re.search(r'```json\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL)
        json_str = json_match.group(1) if json_match else text
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return None
    
    def predict(
        self,
        video_path: str,
        prompt: str,
        parse_json: bool = True
    ) -> Union[str, Dict, List]:
        """
        Run inference on a single video clip.
        
        Args:
            video_path: Path to the video file
            prompt: Text prompt for the model
            parse_json: If True, attempt to parse output as JSON
        
        Returns:
            Parsed JSON object or raw text string
        """
        if self._model is None:
            self.load_model()
        
        messages = self._build_message(prompt, video_path)
        inputs = self._prepare_inputs(messages)
        
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
                "repetition_penalty": self.repetition_penalty,
            }
            if self.do_sample:
                gen_kwargs.update({
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                })
            
            output_ids = self._model.generate(**inputs, **gen_kwargs)
        
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs["input_ids"], output_ids)
        ]
        
        output_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        
        if parse_json:
            parsed = self._extract_json(output_text)
            if parsed is not None:
                return parsed
        
        return output_text
    
    def __del__(self):
        """Cleanup resources."""
        if self._model is not None:
            del self._model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()