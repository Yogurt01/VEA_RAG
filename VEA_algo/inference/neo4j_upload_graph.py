"""
neo4j_upload_graph.py
---------------
Upload FULL scene graphs to Neo4j (only train+val split) using UNWIND batch mode.
Tốc độ nhanh hơn nhiều so với transaction per video.

Usage:
    python neo4j_upload_graph.py \
        --data_root /path/to/All_Videos \
        --split_dataset_file /path/to/dataset_splits.json \
        [--force] \
        [--batch_size 50]

Environment variables (Neo4j):
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE
"""

import os
import json
import argparse
import time
import torch
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI      = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

if not all([URI, USERNAME, PASSWORD]):
    raise ValueError("Missing Neo4j environment variables.")


# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def load_split_ids(split_json_path: Path) -> set:
    with open(split_json_path, 'r') as f:
        splits = json.load(f)
    return set(splits.get("train", [])) | set(splits.get("val", []))


def load_video_data(video_dir: Path) -> Dict[str, Any] | None:
    """
    Load dữ liệu từ một video folder.
    Trả về dict chứa video_name, video_label, embeddings, scene_ids, captions, rst_links.
    """
    emb_path = video_dir / "scene_embeddings.pt"
    seg_path = video_dir / "segments.json"

    if not emb_path.exists() or not seg_path.exists():
        return None

    try:
        data = torch.load(emb_path, map_location="cpu")
        with open(seg_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)

        video_label = int(data['y'].item() if isinstance(data['y'], torch.Tensor) else data['y'])
        embeddings  = data['embeddings']          # Tensor (T, 2048)
        scene_ids   = [int(s) for s in data['scene_ids']]
        rst_links   = data.get('rst_links', [])

        # Trích xuất caption theo thứ tự scene_ids
        captions = []
        if isinstance(segments, list):
            for idx in range(len(scene_ids)):
                cap = segments[idx].get('caption', '') if idx < len(segments) else ''
                captions.append(cap)
        elif isinstance(segments, dict):
            for sid in scene_ids:
                entry = segments.get(str(sid)) or segments.get(sid, {})
                cap   = entry.get('caption', '') if isinstance(entry, dict) else str(entry)
                captions.append(cap)

        return {
            'video_name':  video_dir.name,
            'video_label': video_label,
            'embeddings':  embeddings,   # Tensor
            'scene_ids':   scene_ids,
            'rst_links':   rst_links,
            'captions':    captions,
        }
    except Exception as e:
        print(f"  [ERROR] Failed to load {video_dir.name}: {e}")
        return None


def clear_database(tx):
    tx.run("MATCH (n) DETACH DELETE n")
    print("Database cleared.")


# ==========================================
# 2. BATCH UPLOAD FUNCTIONS
# ==========================================

def upload_batch(tx, batch_data: List[Dict[str, Any]]) -> None:
    """
    Upload một batch video (thường 20-50 video) trong một transaction duy nhất.
    Sử dụng UNWIND để xử lý nhiều records cùng lúc.
    """
    if not batch_data:
        return

    # -------------------------------------------------------
    # 2.1 Tạo Video nodes
    # -------------------------------------------------------
    video_records = [
        {"video_name": v['video_name'], "video_label": v['video_label']}
        for v in batch_data
    ]
    if video_records:
        tx.run("""
            UNWIND $videos AS v
            MERGE (video:Video {id: v.video_name})
            SET video.video_label = v.video_label
        """, videos=video_records)

    # -------------------------------------------------------
    # 2.2 Tạo Scene nodes và HAS_SCENE relationships
    # -------------------------------------------------------
    scene_records = []
    for v in batch_data:
        video_name = v['video_name']
        video_label = v['video_label']
        embeddings_list = v['embeddings'].tolist()
        for idx, scene_id in enumerate(v['scene_ids']):
            scene_records.append({
                "video_name":   video_name,
                "scene_uid":    f"{video_name}_scene_{scene_id}",
                "scene_id":     scene_id,
                "caption":      v['captions'][idx] if idx < len(v['captions']) else '',
                "embedding":    embeddings_list[idx],
                "video_label":  video_label,
            })

    if scene_records:
        tx.run("""
            UNWIND $scenes AS s
            MATCH (v:Video {id: s.video_name})
            MERGE (scene:Scene {uid: s.scene_uid})
            SET scene.scene_id   = s.scene_id,
                scene.caption    = s.caption,
                scene.embedding  = s.embedding,
                scene.video_label = s.video_label
            MERGE (v)-[:HAS_SCENE]->(scene)
        """, scenes=scene_records)

    # -------------------------------------------------------
    # 2.3 Tạo RST relationships (group theo rel_type)
    # -------------------------------------------------------
    # Gom các edge theo rel_type để dùng UNWIND cho từng loại
    edges_by_type = defaultdict(list)
    for v in batch_data:
        video_name = v['video_name']
        for src, tgt, rel_type in v['rst_links']:
            src_int = int(src) - 1
            tgt_int = int(tgt) - 1
            rel_type_upper = str(rel_type).strip().upper().replace(" ", "_").replace("-", "_")
            edges_by_type[rel_type_upper].append({
                "src_uid": f"{video_name}_scene_{src_int}",
                "tgt_uid": f"{video_name}_scene_{tgt_int}",
            })

    # Với mỗi rel_type, tạo một UNWIND query riêng để có relationship type cố định
    for rel_type, edges in edges_by_type.items():
        if not edges:
            continue
        # Dùng f-string để đưa rel_type vào relationship type
        query = f"""
            UNWIND $edges AS edge
            MATCH (src:Scene {{uid: edge.src_uid}})
            MATCH (tgt:Scene {{uid: edge.tgt_uid}})
            MERGE (src)-[:{rel_type}]->(tgt)
        """
        tx.run(query, edges=edges)


# ==========================================
# 3. MAIN
# ==========================================

def main(args: argparse.Namespace) -> None:
    data_root  = Path(args.data_root)
    split_json = Path(args.split_dataset_file)
    batch_size = args.batch_size

    if not data_root.is_dir():
        print(f"Error: Directory not found at '{data_root}'")
        return
    if not split_json.exists():
        print(f"Error: Split file not found at '{split_json}'")
        return

    allowed_ids = load_split_ids(split_json)
    print(f"Loaded {len(allowed_ids)} video IDs from train+val split.")

    # Lấy danh sách thư mục video hợp lệ (có scene_embeddings.pt)
    video_dirs = sorted([
        d for d in data_root.iterdir()
        if d.is_dir() and d.name in allowed_ids and (d / "scene_embeddings.pt").exists()
    ])
    total_videos = len(video_dirs)
    print(f"Found {total_videos} video folders to process.\n")

    # Kết nối Neo4j
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    try:
        driver.verify_connectivity()
        print("Connected to Neo4j.\n")
    except Exception as e:
        print(f"Cannot connect to Neo4j: {e}")
        return

    # Xóa database nếu có --force
    with driver.session(database=DATABASE) as session:
        if args.force:
            print("=" * 60)
            print("WARNING: --force flag is ACTIVE. Clearing database...")
            print("=" * 60)
            session.execute_write(clear_database)

    # Biến đếm
    success_count = 0
    failed_count = 0
    start_time = time.time()

    # Xử lý theo batch
    for batch_start in range(0, total_videos, batch_size):
        batch_dirs = video_dirs[batch_start:batch_start + batch_size]
        batch_data = []

        # Load dữ liệu từng video trong batch
        for video_dir in batch_dirs:
            data = load_video_data(video_dir)
            if data is None:
                failed_count += 1
                continue
            batch_data.append(data)

        if not batch_data:
            continue

        # Upload batch
        try:
            with driver.session(database=DATABASE) as session:
                session.execute_write(upload_batch, batch_data)
                success_count += len(batch_data)
                print(f"[Batch {batch_start//batch_size + 1}] "
                      f"Uploaded {len(batch_data)} videos "
                      f"({success_count}/{total_videos})")
        except Exception as e:
            print(f"[ERROR] Batch {batch_start//batch_size + 1} failed: {e}")
            failed_count += len(batch_data)
            # Thoát hoặc continue tùy ý
            # continue

    # Thống kê
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"UPLOAD COMPLETED in {elapsed:.2f} seconds")
    print(f"  Successful: {success_count} videos")
    print(f"  Failed    : {failed_count} videos")
    print("=" * 60 + "\n")

    # In thống kê tổng quan của database
    with driver.session(database=DATABASE) as session:
        result = session.run("""
            MATCH (v:Video) WITH count(v) AS total_videos
            MATCH (s:Scene) WITH total_videos, count(s) AS total_scenes
            MATCH ()-[r]->() RETURN total_videos, total_scenes, count(r) AS total_relationships
        """).single()
        if result:
            print("DATABASE STATISTICS")
            print(f"  Total Video Nodes  : {result['total_videos']}")
            print(f"  Total Scene Nodes  : {result['total_scenes']}")
            print(f"  Total Relationships: {result['total_relationships']}")

    driver.close()
    print("\nDone.")


# ==========================================
# 4. ENTRY POINT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload FULL scene graphs to Neo4j using UNWIND batch mode."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing per-video folders."
    )
    parser.add_argument(
        "--split_dataset_file",
        type=str,
        required=True,
        help="Path to dataset_splits.json file."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear ALL existing data in Neo4j before uploading."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of videos per batch. Default: 50"
    )
    args = parser.parse_args()
    main(args)