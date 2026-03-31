import logging
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import moviepy.editor as mp

logger = logging.getLogger(__name__)

class VLMProcessor:
    def __init__(self, model_id="Qwen/Qwen2-VL-2B-Instruct"):
        """
        Initializes the Qwen3-VL/Qwen2-VL model with flash_attention_2 
        as required for memory optimization.
        """
        logger.info(f"Loading VLM model: {model_id} with flash_attention_2...")
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
        except Exception as e:
            logger.error(f"Failed to load VLM: {e}")
            raise

    def extract_visual_caption(self, scene_path: str) -> str:
        """
        Extracts frames from the scene, passes them to Qwen-VL, 
        and extracts a visual caption using the specified generation kwargs.
        """
        try:
            clip = mp.VideoFileClip(scene_path)
            duration = clip.duration
            mid_time = duration / 2.0
            frame = clip.get_frame(mid_time)
            clip.close()
            
            image = Image.fromarray(frame).convert("RGB")
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Describe the content of this video scene concisely. Focus on the main action and visual details."},
                    ],
                }
            ]
            
            text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            # VL presence_penalty approx mapping
            generation_kwargs_vl = {
                "max_new_tokens": 256,
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
                "repetition_penalty": 1.5,
            }
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **generation_kwargs_vl)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            return output_text[0].strip()
            
        except Exception as e:
            logger.error(f"Error extracting visual caption for {scene_path}: {e}")
            return ""
