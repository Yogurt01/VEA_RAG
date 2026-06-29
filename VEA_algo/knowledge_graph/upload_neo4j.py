import os
import json
import argparse
import torch
from pathlib import Path
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
# 1. GRAPH UPLOAD FUNCTIONS
# ==========================================

def clear_database(tx):
    tx.run("MATCH (n) DETACH DELETE n")


def load_split_ids(split_json_path: Path) -> set:
    with open(split_json_path, 'r') as f:
        splits = json.load(f)
    return set(splits.get("train", [])) | set(splits.get("val", []))


def upload_single_video_full(tx, data: dict) -> None:
    video_name  = data['video_name']
    video_label = int(data['y'].item() if isinstance(data['y'], torch.Tensor) else data['y'])

    tx.run(
        "MERGE (v:Video {id: $video_name}) SET v.video_label = $video_label",
        video_name=video_name,
        video_label=video_label,
    )

    scene_query = """
    MATCH (v:Video {id: $video_name})
    MERGE (s:Scene {uid: $scene_uid})
    SET s.scene_id = $scene_id,
        s.caption = $caption,
        s.embedding = $embedding,
        s.video_label = $video_label
    MERGE (v)-[:HAS_SCENE]->(s)
    """

    embeddings_list = data['embeddings'].tolist()
    scene_ids_list  = data['scene_ids']
    segments        = data['description']

    def get_caption(idx: int, scene_id_int: int) -> str:
        if isinstance(segments, list):
            return segments[idx].get('caption', '') if idx < len(segments) else ''
        elif isinstance(segments, dict):
            entry = segments.get(str(scene_id_int)) or segments.get(scene_id_int, {})
            return entry.get('caption', '') if isinstance(entry, dict) else str(entry)
        return ''

    for idx, scene_id in enumerate(scene_ids_list):
        scene_id_int = int(scene_id)
        caption      = get_caption(idx, scene_id_int)

        tx.run(
            scene_query,
            video_name=video_name,
            scene_uid=f"{video_name}_scene_{scene_id_int}",
            scene_id=scene_id_int,
            caption=caption,
            embedding=embeddings_list[idx],
            video_label=video_label,
        )

    for src, tgt, rel_type in data.get('rst_links', []):
        src_int        = int(src)
        tgt_int        = int(tgt)
        rel_type_upper = str(rel_type).strip().upper().replace(" ", "_").replace("-", "_")

        tx.run(
            f"""
            MATCH (src:Scene {{uid: $src_uid}}), (tgt:Scene {{uid: $tgt_uid}})
            MERGE (src)-[:{rel_type_upper}]->(tgt)
            """,
            src_uid=f"{video_name}_scene_{src_int}",
            tgt_uid=f"{video_name}_scene_{tgt_int}",
        )


def print_database_statistics(driver, db_name: str) -> None:
    query = """
    MATCH (v:Video) WITH count(v) AS total_videos
    MATCH (s:Scene) WITH total_videos, count(s) AS total_scenes
    MATCH ()-[r]->() RETURN total_videos, total_scenes, count(r) AS total_relationships
    """
    with driver.session(database=db_name) as session:
        result = session.run(query).single()
        if result:
            print("\n" + "=" * 60)
            print("DATABASE STATISTICS AFTER UPLOAD")
            print("=" * 60)
            print(f" Total Video Nodes  : {result['total_videos']}")
            print(f" Total Scene Nodes  : {result['total_scenes']}")
            print(f" Total Relationships: {result['total_relationships']}")
            print("=" * 60 + "\n")


# ==========================================
# 2. MAIN
# ==========================================

def main(args: argparse.Namespace) -> None:
    data_root  = Path(args.data_root)
    split_json = Path(args.split_dataset_file)

    if not data_root.is_dir():
        print(f"Error: Directory not found at '{data_root}'")
        return

    if not split_json.exists():
        print(f"Error: Split file not found at '{split_json}'")
        return

    allowed_ids = load_split_ids(split_json)
    print(f"Loaded {len(allowed_ids)} video IDs from train+val split.")

    video_dirs = sorted([
        d for d in data_root.iterdir()
        if d.is_dir() and d.name in allowed_ids and (d / "scene_embeddings.pt").exists()
    ])

    print(f"Found {len(video_dirs)} video folders to process.\n")

    success_count = 0
    failed_count  = 0

    try:
        with GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD)) as driver:
            driver.verify_connectivity()
            print("Connected to Neo4j.\n")

            with driver.session(database=DATABASE) as session:
                if args.force:
                    print("=" * 60)
                    print("WARNING: --force flag is ACTIVE. Clearing database...")
                    print("=" * 60)
                    session.execute_write(clear_database)
                    print("Database cleared.\n")

                for idx, video_dir in enumerate(video_dirs, 1):
                    video_name = video_dir.name
                    emb_path   = video_dir / "scene_embeddings.pt"
                    seg_path   = video_dir / "segments.json"

                    print(f"[{idx}/{len(video_dirs)}] Processing: {video_name}")

                    if not seg_path.exists():
                        print(f"   [SKIPPED] segments.json not found.")
                        failed_count += 1
                        continue

                    try:
                        video_data = torch.load(emb_path, map_location="cpu")
                        with open(seg_path, 'r', encoding='utf-8') as f:
                            segments_data = json.load(f)

                        video_data['video_name']  = video_name
                        video_data['description'] = segments_data

                        session.execute_write(upload_single_video_full, video_data)
                        print(f"   [SUCCESS] Uploaded {len(video_data['scene_ids'])} scenes.")
                        success_count += 1

                    except Exception as e:
                        print(f"   [FAILED] {video_name}: {e}")
                        failed_count += 1

            print_database_statistics(driver, DATABASE)

    except Exception as e:
        print(f"Critical Error: {e}")
        return

    print("Process completed.")
    print(f" - Successful: {success_count} videos")
    print(f" - Failed    : {failed_count} videos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload FULL scene graphs to Neo4j (only train+val split)."
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
    args = parser.parse_args()
    main(args)