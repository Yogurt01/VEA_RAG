import os
import json
import logging
import shutil

logger = logging.getLogger(__name__)

def save_corpus(corpus_data: list, output_json: str):
    """
    Saves the aggregated corpus data to a JSON file.
    """
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Corpus successfully saved to {output_json}.")
    except Exception as e:
        logger.error(f"Error saving corpus: {e}")

def cleanup_temp_dir(temp_dir: str):
    """
    Removes the temporary directory containing split scenes.
    """
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.error(f"Failed to clean up temp directory {temp_dir}: {e}")
