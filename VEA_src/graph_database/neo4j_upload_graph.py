#!/usr/bin/env python3
"""
upload_neo4j.py

Usage:
    python upload_neo4j.py \
        --data_root /path/to/Videos \
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
from collections import Counter

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
        embeddings = data['embeddings']          # Tensor (T, 2048)
        scene_ids = [int(s) for s in data['scene_ids']]
        rst_links = data.get('rst_links', [])
        video_statistics = build_video_statistics(
            scene_ids,
            rst_links
        )

        # Khởi tạo danh sách kết quả
        captions = []
        visual_elements = []
        audio_tags = []
        audio_vibes = []

        # Xử lý theo kiểu của segments
        if isinstance(segments, list):
            # Giả định scene_ids là chỉ số index trong list
            for sid in scene_ids:
                # Kiểm tra sid hợp lệ
                if isinstance(sid, int) and 0 <= sid < len(segments):
                    entry = segments[sid]
                else:
                    entry = {}
                # Trích xuất
                captions.append(entry.get('caption', ''))
                visual_elements.append(
                    json.dumps(entry.get('visual_elements', {}), ensure_ascii=False)
                )
                audio_tags.append(entry.get('audio_tags', []))
                audio_vibes.append(entry.get('audio_vibes', []))
        elif isinstance(segments, dict):
            for sid in scene_ids:
                # Thử lấy entry với key là str(sid) hoặc sid (int)
                entry = segments.get(str(sid)) or segments.get(sid, {})
                captions.append(entry.get('caption', ''))
                visual_elements.append(
                    json.dumps(entry.get('visual_elements', {}), ensure_ascii=False)
                )
                audio_tags.append(entry.get('audio_tags', []))
                audio_vibes.append(entry.get('audio_vibes', []))
        else:
            # Trường hợp không mong đợi, có thể log warning hoặc để trống
            print(f"  [WARN] segments is neither list nor dict for {video_dir.name}")

        return {
            'video_name': video_dir.name,
            'video_label': video_label,
            'embeddings': embeddings,   # Tensor
            'scene_ids': scene_ids,
            'captions': captions,
            'visual_elements': visual_elements,
            'audio_tags': audio_tags,
            'audio_vibes': audio_vibes,
            "rst_links": rst_links,
            "video_statistics": video_statistics
        }
    except Exception as e:
        print(f"  [ERROR] Failed to load {video_dir.name}: {e}")
        return None


def clear_database(tx):
    tx.run("MATCH (n) DETACH DELETE n")
    print("Database cleared.")

def build_video_statistics(scene_ids, rst_links):
    """
    Build discourse-level metadata for one video.

    Parameters
    ----------
    scene_ids : List[int]
        Scene ids of this video.

    rst_links : List[Tuple[int, int, str]]
        [(src, tgt, relation_type), ...]

    Returns
    -------
    dict
        {
            relation_statistics,
            dominant_rst,
            rst_sequence,
            rst_summary
        }
    """
    total_scenes = len(scene_ids)
    total_relations = len(rst_links)
    if not rst_links:
        return {
            "num_scenes": total_scenes,
            "num_rst_relations": 0,
            "relation_statistics": json.dumps({}, ensure_ascii=False),
            "dominant_rst": "NONE",
            "rst_sequence": "",
            "rst_summary": "No discourse relations were detected in this video."
        }

    # ---------------------------------------------------------
    # 1. Count relation types
    # ---------------------------------------------------------

    relation_counter = Counter()

    for _, _, rel in rst_links:
        rel = str(rel).upper().replace("-", "_").replace(" ", "_")
        relation_counter[rel] += 1

    relation_statistics = dict(sorted(relation_counter.items()))

    # ---------------------------------------------------------
    # 2. Dominant relation
    # ---------------------------------------------------------

    dominant_rst = relation_counter.most_common(1)[0][0]

    # ---------------------------------------------------------
    # 3. Build readable RST sequence
    # ---------------------------------------------------------

    rst_sequence_lines = []

    for src, tgt, rel in sorted(rst_links, key=lambda x:(x[0],x[1])):

        rel = str(rel).upper().replace("-", "_").replace(" ", "_")

        # Dataset bắt đầu từ scene 1
        rst_sequence_lines.append(
            f"S{src} --{rel}--> S{tgt}"
        )

    rst_sequence = "\n".join(rst_sequence_lines)

    # ---------------------------------------------------------
    # 4. Build English summary
    # ---------------------------------------------------------

    total_relations = sum(relation_counter.values())

    sorted_relations = relation_counter.most_common()

    summary_parts = []

    for rel, cnt in sorted_relations:

        percent = cnt / total_relations * 100

        summary_parts.append(
            f"{cnt} {rel} ({percent:.1f}%)"
        )

    summary_text = ", ".join(summary_parts)

    rst_summary = (
        f"The discourse structure contains {total_relations} rhetorical "
        f"relations. The dominant relation is {dominant_rst}. "
        f"The relation distribution is: {summary_text}."
    )

    return {
        "num_scenes": total_scenes,

        "num_rst_relations": total_relations,

        "relation_statistics": json.dumps(
            relation_statistics,
            ensure_ascii=False
        ),

        "dominant_rst": dominant_rst,

        "rst_sequence": rst_sequence,

        "rst_summary": rst_summary
    }

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
    # video_records = [
    #     {"video_name": v['video_name'], "video_label": v['video_label']}
    #     for v in batch_data
    # ]

    video_records = []

    for v in batch_data:

        stats = v["video_statistics"]

        video_records.append({

            "video_name": v["video_name"],

            "video_label": v["video_label"],
            "num_scenes": stats["num_scenes"],

            "num_rst_relations": stats["num_rst_relations"],

            "relation_statistics": stats["relation_statistics"],

            "dominant_rst": stats["dominant_rst"],

            "rst_sequence": stats["rst_sequence"],

            "rst_summary": stats["rst_summary"],
        })

    if video_records:
        tx.run("""
            UNWIND $videos AS v

            MERGE (video:Video {id:v.video_name})

            SET
            video.video_label=v.video_label,
            video.num_scenes=v.num_scenes,
            video.num_rst_relations=v.num_rst_relations,
            video.relation_statistics=v.relation_statistics,
            video.dominant_rst=v.dominant_rst,
            video.rst_sequence=v.rst_sequence,
            video.rst_summary=v.rst_summary
        """, videos=video_records)

    # -------------------------------------------------------
    # 2.2 Tạo Scene nodes và HAS_SCENE relationships
    # -------------------------------------------------------
    scene_records = []
    for v in batch_data:
        video_name = v['video_name']
        video_label = v['video_label']
        embeddings_list = v['embeddings'].cpu().tolist()
        for idx, scene_id in enumerate(v['scene_ids']):
            scene_records.append({
                "video_name":   video_name,
                "scene_uid":    f"{video_name}_scene_{scene_id}",
                "scene_id":     scene_id,
                "caption":      v['captions'][idx] if idx < len(v['captions']) else '',
                "visual_elements":
                    v["visual_elements"][idx]
                    if idx < len(v["visual_elements"]) else "{}",
                "audio_tags":
                    v["audio_tags"][idx]
                    if idx < len(v["audio_tags"]) else [],
                "audio_vibes":
                    v["audio_vibes"][idx]
                    if idx < len(v["audio_vibes"]) else [],
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
                scene.visual_elements = s.visual_elements,
                scene.audio_tags = s.audio_tags,
                scene.audio_vibes = s.audio_vibes,
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
        scene_ids = v['scene_ids']
        for src, tgt, rel_type in v['rst_links']:
            src_idx = int(src) - 1
            tgt_idx = int(tgt) - 1
            if 0 <= src_idx < len(scene_ids) and 0 <= tgt_idx < len(scene_ids):
                src_scene_id = scene_ids[src_idx]
                tgt_scene_id = scene_ids[tgt_idx]
                rel_type_upper = str(rel_type).strip().upper().replace(" ", "_").replace("-", "_")
                edges_by_type[rel_type_upper].append({
                    "src_uid": f"{video_name}_scene_{src_scene_id}",
                    "tgt_uid": f"{video_name}_scene_{tgt_scene_id}",
                })
            else:
                print(f"[WARN] Invalid rst link in {video_name}: src={src}, tgt={tgt}")

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
