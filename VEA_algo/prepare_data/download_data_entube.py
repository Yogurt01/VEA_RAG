import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yt_dlp
from tqdm import tqdm
import os
import re
import glob
import logging
from pathlib import Path
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


DOWNLOAD_PATH = '/content/drive/MyDrive/KhoaLuan/EnTube/Download_2min'
EXCEPT_FILE = "/content/drive/MyDrive/KhoaLuan/EnTube/except_ids.csv"
LABEL_JSON = "/content/drive/MyDrive/KhoaLuan/EnTube/video_ids_label.json"
DATASET_PATH = '/content/drive/MyDrive/KhoaLuan/EnTube/Entube.csv'

df = pd.read_csv(DATASET_PATH)

ignore_ids = pd.read_csv(EXCEPT_FILE)
ignore_ids = ignore_ids['id'].tolist()

exi_paths = glob.glob(DOWNLOAD_PATH + "/*.mp4")
existed_ids = [os.path.basename(path).split(".")[0] for path in exi_paths]

df = df[~df['video_id'].isin(ignore_ids)]
df = df[~df['video_id'].isin(existed_ids)]

# due to available memory, we will only download 40% of the data
# df = df[df['engagement_rate_label'] != 1] # bỏ qua nhãn neutral (1)
# df = df[df['engagement_rate_label'] == 1] # lấy nhãn neutral (1)
df = df.sample(n=200, random_state=1)


LOG_FILE = "/content/drive/MyDrive/KhoaLuan/EnTube/download_log.txt"
# Setup logging
logger = logging.getLogger("yt_dlp")
logger.setLevel(logging.INFO)
logger.handlers.clear()

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(file_handler)

def download_video(url, ydl_opts):
    """Download a single video using yt-dlp with logging."""
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        logger.info(f"Successfully downloaded: {url}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")

def download_videos(urls, output_path, max_workers=2):
    """Download multiple videos in parallel, logging all events."""
    # ydl_opts = {
    #     'format': 'bestvideo[height<=480]+bestaudio/best[height<=480]',
    #     'outtmpl': f"{output_path}/%(id)s",
    #     'merge_output_format': 'mp4',
        
    #     'logger': logger,  # Use the logger instance
    #     'no_warnings': True,   # Suppress warning messages
    #     'ignoreerrors': True,  # Continue downloading on errors
        
    #     'writesubtitles': True,  # Download subtitles if available
    #     'subtitleslangs': ['vi', 'en'],  # Only download Vietnamese and English subtitles
    # }

    ydl_opts = {
        'format': 'mp4',
        # 'format': 'bestvideo[height<=360]+bestaudio/best[height<=360]',
        # 'merge_output_format': 'mp4',
        'outtmpl': f'{output_path}/%(id)s.%(ext)s',
        'quiet': True,
        
        'logger': logger,  # Use the logger instance
        'no_warnings': True,   # Suppress warning messages
        'ignoreerrors': True,  # Continue downloading on errors
        
        'noprogress': True,
        # Subtitle options
        # 'writesubtitles': True,           # Download subtitles
        # 'subtitleslangs': ['vi'],         # Vietnamese only
        # 'writeautomaticsub': True,        # Download auto-generated if manual not available
        # 'embedsubtitles': True,           # Embed subtitles into the video (for mp4/mkv)
        
        
        # 'postprocessor_args': ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23']
    }

    logger.info(f"Starting download of {len(urls)} videos...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(lambda url: download_video(url, ydl_opts), urls), total=len(urls), desc="Downloading videos", unit="video"))

    logger.info("Download process completed.")

# Example usage
sample_df = df
video_urls = sample_df['video_link'].tolist()
download_videos(video_urls, DOWNLOAD_PATH + "/tmp")

video_paths = glob.glob(os.path.join(DOWNLOAD_PATH, "tmp", f"*.mp4"))
print(f"Successfully downloaded {len(video_paths)} videos")


LOG_FILE = "/content/drive/MyDrive/KhoaLuan/EnTube/post_process_logs.txt"
logger = logging.getLogger("yt_dlp")
logger.setLevel(logging.INFO)
logger.handlers.clear()

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(file_handler)

# def post_process_video(video_path):
#     """Post-process a video by extracting a subclip."""
#     try:
#         output_path = os.path.join(DOWNLOAD_PATH, os.path.basename(video_path))
#         ffmpeg_extract_subclip(video_path, 0, 180, outputfile=output_path)
#         logger.info(f"Successfully processed: {video_path} to {output_path}")
#         os.remove(video_path)  # Remove the original video after processing
#     except Exception as e:
#         logger.error(f"Failed to process {video_path}: {e}")

def post_process_video(video_path):
    """Post-process a video by extracting a subclip."""
    try:
        v_path = Path(video_path)
        file_name = v_path.name
        stem_name = v_path.stem
        
        target_folder = Path(DOWNLOAD_PATH) / stem_name
        target_folder.mkdir(parents=True, exist_ok=True)
        
        output_path = target_folder / file_name
        
        ffmpeg_extract_subclip(str(v_path), 0, 120, outputfile=str(output_path))
        logger.info(f"Successfully processed: {video_path} to {output_path}")
        
        if v_path.exists():
            v_path.unlink()
            
    except Exception as e:
        logger.error(f"Failed to process {video_path}: {e}")


logger.info(f"Starting post processing of {len(video_paths)} videos...")

with ThreadPoolExecutor(max_workers=2) as executor:
    list(tqdm(executor.map(lambda path: post_process_video(path), video_paths), total=len(video_paths), desc="Postprocessing videos", unit="video"))

logger.info("Posprocess process completed.")


# ── 1. Lấy danh sách video ĐÃ TỒN TẠI trong thư mục Download_3min ──────────
existing_paths = glob.glob(os.path.join(DOWNLOAD_PATH, "**", "*.mp4"), recursive=True)
existing_ids   = {Path(p).stem for p in existing_paths}
print(f"Số video thực sự tồn tại trong thư mục: {len(existing_ids)}")

# ── 2. Lưu video_ids_label.json (chỉ log video có file thật) ─────────────────
# Đọc file cũ nếu đã có (để cộng dồn qua nhiều lần chạy)
if os.path.exists(LABEL_JSON):
    with open(LABEL_JSON, "r") as f:
        label_dict = json.load(f)
else:
    label_dict = {}

# df_full là DataFrame GỐC (trước khi sample/filter) chứa tất cả nhãn
df_full = pd.read_csv(DATASET_PATH)

for _, row in df_full.iterrows():
    vid   = str(row["video_id"])
    label = str(int(row["engagement_rate_label"]))   # key là string "0" / "1"
    
    if vid not in existing_ids:          # bỏ qua nếu file không tồn tại
        continue

    if label not in label_dict:
        label_dict[label] = []

    if vid not in label_dict[label]:     # tránh trùng lặp khi chạy lại
        label_dict[label].append(vid)

with open(LABEL_JSON, "w") as f:
    json.dump(label_dict, f, indent=2, ensure_ascii=False)

total_logged = sum(len(v) for v in label_dict.values())
print(f"Đã lưu {total_logged} video vào {LABEL_JSON}")
for lbl, ids in label_dict.items():
    print(f"  Label {lbl}: {len(ids)} video")

# ── 3. Cập nhật except_ids.csv (tất cả video ĐÃ THỬ download, kể cả thất bại) 
# "đã thử" = có trong sample_df (vì đó là batch vừa chạy download)
attempted_ids = set(sample_df["video_id"].astype(str).tolist())

if os.path.exists(EXCEPT_FILE):
    old_except = pd.read_csv(EXCEPT_FILE)
    old_ids    = set(old_except["id"].astype(str).tolist())
else:
    old_ids = set()

merged_ids = old_ids | attempted_ids   # hợp 2 tập, tự loại trùng

pd.DataFrame({"id": sorted(merged_ids)}).to_csv(EXCEPT_FILE, index=False)
print(f"\nĐã cập nhật {EXCEPT_FILE}: {len(merged_ids)} ID "
      f"(+{len(merged_ids - old_ids)} mới so với trước)")