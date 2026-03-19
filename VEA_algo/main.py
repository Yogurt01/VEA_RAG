import logging
from modules.scene_detector import SceneExtractor
from modules.captioner import VLMProcessor
from modules.transcriber import ASRProcessor
from utils.data_handler import save_corpus, cleanup_temp_dir

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_video_pipeline(video_path: str, output_json: str, temp_dir: str = "./vea_temp_scenes"):
    """
    Main orchestration function for the VEA pipeline.
    """
    logger.info(f"Starting video processing pipeline for: {video_path}")
    
    scene_extractor = SceneExtractor(output_dir=temp_dir)
    try:
        scenes = scene_extractor.split_video_into_scenes(video_path)
    except Exception as e:
        logger.error(f"Scene extraction failed: {e}")
        return

    vlm_processor = VLMProcessor()
    asr_processor = ASRProcessor()
    
    corpus_data = []
    
    for scene in scenes:
        scene_id = scene["scene_id"]
        start_t = scene["start_time"]
        end_t = scene["end_time"]
        scene_file = scene["file_path"]
        
        logger.info(f"Processing Scene {scene_id} ({start_t:.2f}s - {end_t:.2f}s)...")
        
        caption = vlm_processor.extract_visual_caption(scene_file)
        transcript = asr_processor.extract_transcript(scene_file)
        
        corpus_data.append({
            "scene_id": scene_id,
            "start_time": round(start_t, 2),
            "end_time": round(end_t, 2),
            "visual_caption": caption,
            "audio_transcript": transcript
        })
        
    save_corpus(corpus_data, output_json)
    cleanup_temp_dir(temp_dir)
    
    logger.info("Pipeline execution completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VEA Video Processing Pipeline")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, default="output_corpus.json", help="Path to output JSON file")
    
    args = parser.parse_args()
    if args.video:
        process_video_pipeline(args.video, args.output)
    else:
        logger.warning("No video path provided. Please use --video to specify the input video.")
