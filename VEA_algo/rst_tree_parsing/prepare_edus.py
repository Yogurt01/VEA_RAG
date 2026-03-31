import os
import json
import argparse
from pathlib import Path


def normalize_path(path):
    """Normalize a path to use forward slashes."""
    return os.path.normpath(path).replace('\\', '/')

class CaptionExtractor:
    def __init__(self, args):
        # Process arguments
        self.root_dir = Path(normalize_path(args.root_dir))
        self.output_file = Path(normalize_path(args.output_json))
        self.min_edus = args.min_edus
        self.segmentation_filename = 'segments.json'

        self.video_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
    def run(self):
        dataset  = []

        for video_dir in self.video_dirs:
            segment_file = video_dir / self.segmentation_filename
            if not segment_file.exists():
                continue

            with open(segment_file, 'r', encoding='utf-8') as f:
                segments = json.load(f)

            edus = [
                seg["caption"].strip()
                for seg in segments
                if seg.get("caption", "").strip()
            ]

            if len(edus) < args.min_edus:
                continue

            dataset.append({
                "doc_id": video_dir.name,
                "edu_strings": edus,
                "video_dir": str(video_dir),
            })

        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gom captions từ segments.json thành RST dataset")
    parser.add_argument("--root_dir", type=str, help="Thư mục gốc chứa các folder video")
    parser.add_argument("--output_json", type=str, help="Path file JSON output")
    parser.add_argument("--min_edus", type=int, default=2, help="Số EDU tối thiểu để include doc (mặc định: 2)")
    
    args = parser.parse_args()

    extractor = CaptionExtractor(args)
    extractor.run()