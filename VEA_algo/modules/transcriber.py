import logging
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

class ASRProcessor:
    def __init__(self, model_size="distil-large-v3"):
        logger.info(f"Loading ASR model: Faster-Whisper ({model_size})...")
        try:
            self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
        except Exception as e:
            logger.error(f"Failed to load ASR Model: {e}")
            raise

    def extract_transcript(self, scene_path: str) -> str:
        """
        Extracts the audio transcript for the specified video scene.
        """
        try:
            segments, info = self.model.transcribe(
                scene_path, 
                beam_size=5,
                repetition_penalty=2.0 
            )
            
            transcripts = [segment.text.strip() for segment in segments]
            return " ".join(transcripts)
        except Exception as e:
            logger.error(f"Error extracting transcript from {scene_path}: {e}")
            return ""
