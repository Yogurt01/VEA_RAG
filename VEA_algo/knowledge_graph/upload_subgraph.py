import os
import glob
import json
import argparse
import torch
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

if not all([URI, USERNAME, PASSWORD]):
    raise ValueError(
        "Missing required Neo4j environment variables. "
        "Please ensure NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD are set in your .env file."
    )


# ==========================================
# 1. GRAPH UPLOAD & MANAGEMENT FUNCTIONS
# ==========================================

def clear_database(tx):
    """Delete ALL nodes and relationships in the database."""
    query = "MATCH (n) DETACH DELETE n"
    tx.run(query)


def upload_single_video(tx, data):
    """
    Execute Cypher queries to create or update Video nodes, Scene nodes, 
    and RST relationships for a single video.
    """
    video_name = data.get('video_name', 'Unknown')
    video_label = int(data['original_y'])

    # 1. Create or update the root Video node
    video_query = """
    MERGE (v:Video {id: $video_name})
    SET v.fidelity = $fidelity,
        v.sparsity = $sparsity,
        v.predicted_label = $original_y
    """
    tx.run(video_query,
           video_name=video_name,
           fidelity=data['fidelity'],
           sparsity=data['sparsity'],
           original_y=video_label)

    # 2. Filter and create important Scene nodes
    important_scenes_set = set(data['important_scenes'])
    
    # Safely convert embeddings to list
    embeddings = data['embeddings']
    embeddings_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)

    scene_query = """
    MATCH (v:Video {id: $video_name})
    MERGE (s:Scene {uid: $scene_uid})
    SET s.scene_id = $scene_id,
        s.caption = $caption,
        s.embedding = $embedding,
        s.video_label = $video_label
    MERGE (v)-[:HAS_SCENE]->(s)
    """

    for idx, scene_id in enumerate(data['scene_ids']):
        scene_id_int = int(scene_id)
        if scene_id_int not in important_scenes_set:
            continue
        
        # WARNING !!!
        desc = data['description'][idx]
        scene_uid = f"{video_name}_scene_{scene_id_int}"

        tx.run(scene_query,
               video_name=video_name,
               scene_uid=scene_uid,
               scene_id=scene_id_int,
               caption=desc.get('caption', ''),
               embedding=embeddings_list[idx],
               video_label=video_label)

    # 3. Create RST relationships between important Scene nodes
    for src, tgt, rel_type in data['rst_links']:
        src_int = int(src) - 1
        tgt_int = int(tgt) - 1
        if src_int in important_scenes_set and tgt_int in important_scenes_set:
            src_uid = f"{video_name}_scene_{src_int}"
            tgt_uid = f"{video_name}_scene_{tgt_int}"
            
            rel_type_upper = str(rel_type).strip().upper().replace(" ", "_").replace("-", "_")

            # link_query = f"""
            # MATCH (src:Scene {{uid: $src_uid}}), (tgt:Scene {{uid: $tgt_uid}})
            # MERGE (src)-[:{rel_type_upper}]->(tgt)
            # """

            link_query = """
            MATCH (src:Scene {uid: $src_uid}), (tgt:Scene {uid: $tgt_uid})
            CALL apoc.merge.relationship(src, $rel_type, {}, {}, tgt, {}) YIELD rel
            RETURN rel
            """
            tx.run(link_query, src_uid=src_uid, tgt_uid=tgt_uid, rel_type=rel_type_upper)


def print_database_statistics(driver, db_name):
    """Print an overview of the total number of nodes and relationships."""
    # query = """
    # CALL { MATCH (v:Video) RETURN count(v) as total_videos }
    # CALL { MATCH (s:Scene) RETURN count(s) as total_scenes }
    # CALL { MATCH ()-[r]->() RETURN count(r) as total_relationships }
    # RETURN total_videos, total_scenes, total_relationships
    # """
    query = """
    MATCH (v:Video) WITH count(v) as total_videos
    MATCH (s:Scene) WITH total_videos, count(s) as total_scenes
    MATCH ()-[r]->()
    RETURN total_videos, total_scenes, count(r) as total_relationships
    """
    
    with driver.session(database=db_name) as session:
        result = session.run(query).single()
        if result:
            print("\n" + "=" * 60)
            print("DATABASE OVERVIEW STATISTICS AFTER UPLOAD")
            print("=" * 60)
            print(f" Total Video Nodes (:Video)         : {result['total_videos']}")
            print(f" Total Scene Nodes (:Scene)         : {result['total_scenes']}")
            print(f" Total Relationships                : {result['total_relationships']}")
            print("=" * 60 + "\n")


# ==========================================
# 2. MAIN EXECUTION LOGIC
# ==========================================

def main(args):
    if not os.path.isdir(args.data_root):
        print(f"Error: Directory not found at '{args.data_root}'")
        return
    if not os.path.isdir(args.subgraph_dir):
        print(f"Error: Directory not found at '{args.subgraph_dir}'")
        return

    pt_pattern = os.path.join(args.subgraph_dir, '*_explanation.pt')
    pt_files = glob.glob(pt_pattern)

    print(f"Found a total of {len(pt_files)} explanation graph files to process.")
    print("Starting batch upload process to Neo4j...\n")

    success_count = 0
    failed_count = 0

    try:
        with GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD)) as driver:
            driver.verify_connectivity()
            print("Successfully connected to Neo4j.\n")

            with driver.session(database=DATABASE) as session:
                
                if args.force:
                    print("=" * 60)
                    print("WARNING: --force flag is ACTIVE.")
                    print("Deleting ALL existing nodes and relationships in the database...")
                    print("=" * 60)
                    session.execute_write(clear_database)
                    print("Database cleared successfully. Starting fresh upload.\n")

                for idx, file_path in enumerate(pt_files, 1):
                    base_filename = os.path.basename(file_path)
                    video_name = base_filename.replace('_explanation.pt', '')
                    
                    print(f"[{idx}/{len(pt_files)}] Processing Video: {video_name}")

                    json_path = os.path.join(args.data_root, video_name, 'segments.json')

                    if not os.path.exists(json_path):
                        print(f"   [SKIPPED] Description file not found at: {json_path}")
                        failed_count += 1
                        continue

                    try:
                        video_data = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
                        
                        with open(json_path, 'r', encoding='utf-8') as f:
                            segments_data = json.load(f)
                        
                        video_data['video_name'] = video_name
                        video_data['description'] = segments_data

                        session.execute_write(upload_single_video, video_data)
                        print(f"   [SUCCESS] Successfully uploaded explanation graph for {video_name}.")
                        success_count += 1

                    except Exception as e:
                        print(f"   [FAILED] Error processing video {video_name}: {e}")
                        failed_count += 1

            print_database_statistics(driver, DATABASE)

    except Exception as e:
        print(f"Critical Error: Failed to connect or process data in Neo4j: {e}")
        return

    print("\nProcess completed.")
    print(f" - Successful uploads : {success_count} videos")
    print(f" - Failed/Skipped     : {failed_count} videos")


# ==========================================
# 3. ENTRY POINT & ARGUMENT PARSING
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload SubgraphX explanation graphs and metadata to a Neo4j Database."
    )

    parser.add_argument(
        "--data_root", 
        type=str, 
        default="EnTube_Small/Download_2min",
        help="Path to the root directory containing the original segments.json files"
    )
    parser.add_argument(
        "--subgraph_dir", 
        type=str, 
        default="EnTube_Small/SubgraphX_Results",
        help="Path to the directory containing *_explanation.pt files"
    )
    
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force clear ALL existing data in the database before uploading"
    )
    # ================================

    args = parser.parse_args()
    main(args)
