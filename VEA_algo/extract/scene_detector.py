import logging
from pathlib import Path
from scenedetect import detect, ContentDetector, split_video_ffmpeg

logger = logging.getLogger(__name__)

class SceneExtractor:
    def __init__(self, output_dir: str = "temp_scenes"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def split_video_into_scenes(self, video_path: str) -> list:
        """
        Splits a video into scenes using PySceneDetect and ffmpeg.
        Returns a list of dictionaries containing scene metadata (start, end, file_path).
        """
        logger.info(f"Detecting scenes for {video_path}...")
        try:
            # We use ContentDetector to find cuts
            scene_list = detect(video_path, ContentDetector(threshold=27.0))
            logger.info(f"Detected {len(scene_list)} scenes.")
            
            # Using ffmpeg to actually split the files into the temporary directory
            output_pattern = str(self.output_dir / "scene_$SCENE_NUMBER.mp4")
            
            # Use PySceneDetect's built-in split_video_ffmpeg directly
            split_video_ffmpeg(video_path, scene_list, output_pattern, show_progress=False)
            
            scenes = []
            for i, scene in enumerate(scene_list):
                scene_file = self.output_dir / f"scene_{i+1:03d}.mp4"
                if scene_file.exists():
                    scenes.append({
                        "scene_id": i + 1,
                        "start_time": scene[0].get_seconds(),
                        "end_time": scene[1].get_seconds(),
                        "file_path": str(scene_file)
                    })
            return scenes
            
        except Exception as e:
            logger.error(f"Error splitting video: {e}")
            raise
