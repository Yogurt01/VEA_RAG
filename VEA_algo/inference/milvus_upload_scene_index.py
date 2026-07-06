"""
milvus_upload_scene_index.py
------------------------------
Upload scene-level embeddings (Content Similarity / Cross-Modal Self-Querying
INDEX-SIDE) lên Milvus. Dùng cùng với milvus_compute_video_query.py (query-side)
và milvus_inference_pipeline.py --evidence_mode content/full.
"""

import os
import json
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from pymilvus import MilvusClient, DataType
from dotenv import load_dotenv

load_dotenv()

CLUSTER_ENDPOINT = os.getenv("MILVUS_CLUSTER_ENDPOINT")
TOKEN            = os.getenv("MILVUS_TOKEN")
BATCH_SIZE       = 1000


# ==========================================
# 1. COLLECTION SETUP
# ==========================================

def setup_milvus_collection(client, collection_name: str, force: bool = False) -> None:
    """Tạo collection Milvus với schema và HNSW index."""
    if client.has_collection(collection_name):
        if force:
            print(f"Force mode: dropping existing collection '{collection_name}'...")
            client.drop_collection(collection_name)
            print("Dropped successfully.")
        else:
            print(f"Collection '{collection_name}' already exists. Skipping creation (use --force to recreate).")
            return
    else:
        print(f"Collection '{collection_name}' does not exist. Creating...")

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)

    # Primary key
    schema.add_field(
        field_name="scene_uid",
        datatype=DataType.VARCHAR,
        max_length=200,
        is_primary=True,
    )
    # Vector field
    schema.add_field(
        field_name="vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=2048,
    )
    # Metadata fields
    schema.add_field(field_name="video_id",     datatype=DataType.VARCHAR, max_length=200)
    schema.add_field(field_name="scene_id",     datatype=DataType.INT64)
    schema.add_field(field_name="video_label",  datatype=DataType.INT64)
    schema.add_field(field_name="caption",      datatype=DataType.VARCHAR, max_length=1000)

    # HNSW index với cosine similarity
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type="COSINE",
        index_type="HNSW",
        params={"M": 16, "efConstruction": 64},
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    print(f"Collection '{collection_name}' created successfully.")


# ==========================================
# 2. DATA PREPARATION (with split filtering)
# ==========================================

def prepare_milvus_payload(data_root: Path, include_videos: set) -> list:
    """
    Quét toàn bộ video folder trong data_root.
    Chỉ xử lý các video có tên trong include_videos.
    Với mỗi video: đọc scene_embeddings.pt + segments.json,
    tạo record cho TẤT CẢ scene (không lọc important).
    """
    all_video_dirs = sorted([
        d for d in data_root.iterdir()
        if d.is_dir() and (d / "scene_embeddings.pt").exists()
    ])

    # Lọc chỉ lấy video nằm trong include_videos
    video_dirs = [d for d in all_video_dirs if d.name in include_videos]

    print(f"[INFO] Found {len(video_dirs)} video folders to process (out of {len(all_video_dirs)} total).")

    milvus_payload = []
    success_count  = 0
    failed_count   = 0

    for video_dir in tqdm(video_dirs, desc="Preparing data"):
        video_name = video_dir.name
        emb_path   = video_dir / "scene_embeddings.pt"
        seg_path   = video_dir / "segments.json"

        if not seg_path.exists():
            failed_count += 1
            continue

        try:
            pt_data = torch.load(emb_path, map_location="cpu")

            video_label    = int(pt_data['y'].item() if isinstance(pt_data['y'], torch.Tensor) else pt_data['y'])
            embeddings     = pt_data['embeddings'].tolist()
            scene_ids_list = pt_data['scene_ids']

            with open(seg_path, 'r', encoding='utf-8') as f:
                segments = json.load(f)

            # Tạo lookup map caption — hỗ trợ cả list và dict
            def get_caption(idx: int, scene_id_int: int) -> str:
                if isinstance(segments, list):
                    entry = segments[idx] if idx < len(segments) else {}
                elif isinstance(segments, dict):
                    entry = segments.get(str(scene_id_int)) or segments.get(scene_id_int, {})
                else:
                    return ''
                if isinstance(entry, dict):
                    caption = entry.get('caption', '')
                else:
                    caption = str(entry)
                # Giới hạn độ dài caption để không vượt quá max_length Milvus
                return caption[:990] + "..." if len(caption) > 990 else caption

            # Tạo record cho TẤT CẢ scene, không filter
            for idx, scene_id in enumerate(scene_ids_list):
                scene_id_int = int(scene_id)
                record = {
                    "scene_uid":   f"{video_name}_scene_{scene_id_int}",
                    "vector":      embeddings[idx],
                    "video_id":    video_name,
                    "scene_id":    scene_id_int,
                    "video_label": video_label,
                    "caption":     get_caption(idx, scene_id_int),
                }
                milvus_payload.append(record)

            success_count += 1

        except Exception as e:
            print(f"\n[ERROR] Failed to process {video_name}: {e}")
            failed_count += 1

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED")
    print("=" * 60)
    print(f" Videos processed successfully : {success_count}")
    print(f" Videos failed/skipped         : {failed_count}")
    print(f" Total scene records created   : {len(milvus_payload)}")
    print("=" * 60)

    return milvus_payload


# ==========================================
# 3. BATCH INSERT
# ==========================================

def insert_to_milvus(client, collection_name: str, payload: list, batch_size: int = BATCH_SIZE) -> None:
    """Insert data vào Milvus theo batch để tránh OOM."""
    if not payload:
        print("No data to insert.")
        return

    total     = len(payload)
    n_batches = (total + batch_size - 1) // batch_size
    print(f"\nInserting {total:,} records in {n_batches} batches...")

    for i in tqdm(range(n_batches), desc="Inserting batches"):
        batch = payload[i * batch_size: (i + 1) * batch_size]
        client.insert(collection_name=collection_name, data=batch)

    print("Flushing data to storage...")
    client.flush(collection_name=collection_name)

    stats = client.get_collection_stats(collection_name)
    print("\n" + "=" * 50)
    print("MILVUS DATABASE STATISTICS")
    print("=" * 50)
    print(f" Total vectors stored: {stats['row_count']:,}")
    print("=" * 50)


# ==========================================
# 4. MAIN
# ==========================================

def main(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    split_path = Path(args.split_json)

    if not data_root.is_dir():
        print(f"Error: Directory not found at '{data_root}'")
        return

    if not split_path.is_file():
        print(f"Error: Split JSON file not found at '{split_path}'")
        return

    if not all([CLUSTER_ENDPOINT, TOKEN]):
        raise ValueError(
            "Missing Milvus environment variables. "
            "Please set MILVUS_CLUSTER_ENDPOINT, MILVUS_TOKEN."
        )

    # Đọc split JSON
    with open(split_path, 'r') as f:
        splits = json.load(f)

    # Lấy video IDs từ train và val
    train_ids = set(splits.get('train', []))
    val_ids   = set(splits.get('val', []))
    include_videos = train_ids | val_ids  # union

    print(f"[INFO] Train: {len(train_ids)} videos, Val: {len(val_ids)} videos")
    print(f"[INFO] Total videos to include: {len(include_videos)}")

    print("Initializing Milvus upload pipeline...")
    client = MilvusClient(uri=CLUSTER_ENDPOINT, token=TOKEN)
    content_collection_name = args.content_collection_name or os.getenv("MILVUS_CONTENT_COLLECTION_NAME")

    try:
        # 1. Setup collection
        setup_milvus_collection(client, content_collection_name, force=args.force)

        # 2. Chuẩn bị dữ liệu (chỉ train + val)
        payload = prepare_milvus_payload(data_root, include_videos)

        # 3. Insert
        insert_to_milvus(client, content_collection_name, payload)

        print("\nPipeline completed successfully.")

    finally:
        print("\nClosing Milvus connection...")
        client.close()
        print("Connection closed.")


# ==========================================
# 5. ENTRY POINT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Upload FULL scene embeddings to Milvus (no important-scene filtering). "
            "Only videos from train and val splits (defined in dataset_splits.json) are uploaded."
        )
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing per-video folders (each with scene_embeddings.pt and segments.json).",
    )
    parser.add_argument(
        "--split_json",
        type=str,
        required=True,
        help="Path to dataset_splits.json file (must contain 'train' and 'val' lists of video IDs).",
    )
    parser.add_argument(
        "--content_collection_name",
        type=str,
        required=True
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop and recreate the Milvus collection before uploading.",
    )
    args = parser.parse_args()
    main(args)