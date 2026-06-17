import os
import glob
import json
import argparse
import torch
from tqdm import tqdm
from pymilvus import MilvusClient, DataType
from dotenv import load_dotenv

load_dotenv()

# Configuration
CLUSTER_ENDPOINT = os.getenv("MILVUS_CLUSTER_ENDPOINT")
TOKEN = os.getenv("MILVUS_TOKEN")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME")
BATCH_SIZE = 1000


def setup_milvus_collection(client, collection_name, force=False):
    """
    Setup Milvus collection with schema and index.
    Drops existing collection if it exists and force=True.
    """
    # Check if collection exists
    if client.has_collection(collection_name):
        if force:
            print(f"Force mode enabled. Dropping existing collection: {collection_name}")
            client.drop_collection(collection_name)
            print("Collection dropped successfully.")
        else:
            print(f"Collection {collection_name} already exists. Skipping creation (use --force to drop and recreate).")
            return  # Keep existing collection and skip schema/index creation
    else:
        print(f"Collection {collection_name} does not exist. Creating new one.")

    # Create schema
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)

    # Primary key: unique scene UID
    schema.add_field(
        field_name="scene_uid",
        datatype=DataType.VARCHAR,
        max_length=100,
        is_primary=True
    )

    # Vector field: 2048-dimensional embeddings
    schema.add_field(
        field_name="vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=2048
    )

    # Metadata fields
    schema.add_field(field_name="video_id", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="scene_id", datatype=DataType.INT64)
    schema.add_field(field_name="video_label", datatype=DataType.INT64)
    schema.add_field(field_name="caption", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="is_important", datatype=DataType.BOOL)

    # Configure HNSW index with cosine similarity
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type="COSINE",
        index_type="HNSW",
        params={"M": 16, "efConstruction": 64}
    )

    # Create collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

    print(f"Collection '{collection_name}' created successfully and ready for data.")

def extract_scene_id_from_path(clip_path):
    """
    Hàm bổ trợ trích xuất ID phân cảnh (dạng số) từ đường dẫn 'clip_path'.
    Ví dụ: '/.../clips/clip_001_17.98-23.77.mp4' -> 1
    """
    if not clip_path:
        return None
    
    # Lấy tên file gốc: "clip_001_17.98-23.77.mp4"
    filename = os.path.basename(clip_path) 
    try:
        # Tách chuỗi theo ký tự '_' và lấy phần tử thứ 2 ('001'), sau đó ép kiểu sang int
        return int(filename.split('_')[1])
    except (IndexError, ValueError):
        return None


def prepare_milvus_payload(data_root, subgraph_dir):
    """
    Scan all .pt files and prepare Milvus payload.
    Returns list of records ready for insertion.
    """
    if not os.path.isdir(data_root):
        print(f"Error: Directory not found at '{data_root}'")
        return []
    if not os.path.isdir(subgraph_dir):
        print(f"Error: Directory not found at '{subgraph_dir}'")
        return []

    # Find all explanation files
    pt_files = glob.glob(os.path.join(subgraph_dir, '*_explanation.pt'))
    print(f"[INFO] Found {len(pt_files)} explanation files to process.")

    milvus_payload = []
    success_count = 0
    failed_count = 0

    for file_path in tqdm(pt_files, desc="Preparing data"):
        base_name = os.path.basename(file_path)
        video_name = base_name.replace('_explanation.pt', '')
        json_path = os.path.join(data_root, video_name, 'segments.json')

        # Skip if segments.json not found
        if not os.path.exists(json_path):
            failed_count += 1
            continue

        try:
            # Load graph data
            pt_data = torch.load(file_path, map_location='cpu', weights_only=False)

            video_label = int(pt_data['original_y'])
            important_scenes = set(pt_data['important_scenes'])
            embeddings = pt_data['embeddings'].tolist()
            scene_ids = pt_data['scene_ids']

            # Load text descriptions
            with open(json_path, 'r', encoding='utf-8') as f:
                segments = json.load(f)

            # 3. CẢI TIẾN: Tạo bản đồ tra cứu (lookup map) bằng ID thực tế từ clip_path
            segment_map = {}
            for i, seg in enumerate(segments):
                scene_id_actual = extract_scene_id_from_path(seg.get('clip_path'))
                
                # Phương án dự phòng nếu clip_path không hợp lệ hoặc trống
                if scene_id_actual is None:
                    scene_id_actual = i 
                    
                segment_map[scene_id_actual] = seg

            # Process all scenes
            for idx, scene_id in enumerate(scene_ids):
                scene_id_int = int(scene_id)
                # desc = segments[idx]

                # Tra cứu chính xác metadata dựa trên ID thực tế đã tạo ở segment_map
                desc = segment_map.get(scene_id_int, {})

                # Get caption
                caption = desc.get('caption', '')
                if len(caption) > 990:
                    caption = caption[:990] + "..."

                record = {
                    "scene_uid": f"{video_name}_scene_{scene_id_int}",
                    "vector": embeddings[idx],
                    "video_id": video_name,
                    "scene_id": scene_id_int,
                    "video_label": video_label,
                    "caption": caption,
                    "is_important": bool(scene_id_int in important_scenes)
                }

                milvus_payload.append(record)

            success_count += 1

        except Exception as e:
            print(f"\n[ERROR] Failed to process {video_name}: {e}")
            failed_count += 1

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED")
    print("=" * 60)
    print(f" - Videos processed successfully: {success_count}")
    print(f" - Videos failed/skipped:         {failed_count}")
    print(f" - Total scene records created:   {len(milvus_payload)}")
    print("=" * 60)

    return milvus_payload


def insert_to_milvus(client, collection_name, payload, batch_size=BATCH_SIZE):
    """
    Insert data into Milvus in batches to avoid memory issues.
    """
    if not payload:
        print("No data to insert.")
        return

    total_records = len(payload)
    num_batches = (total_records + batch_size - 1) // batch_size

    print(f"\nInserting {total_records} records into Milvus in {num_batches} batches...")

    for batch_idx in tqdm(range(num_batches), desc="Inserting batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_records)
        batch = payload[start_idx:end_idx]

        client.insert(collection_name=collection_name, data=batch)

    # Flush to ensure data is persisted
    print("Flushing data to storage...")
    client.flush(collection_name=collection_name)

    # Get final statistics
    stats = client.get_collection_stats(collection_name)
    print("\n" + "=" * 50)
    print("MILVUS DATABASE STATISTICS")
    print("=" * 50)
    print(f"Total vectors stored: {stats['row_count']}")
    print("=" * 50)


def upload_embeddings(args):
    """Main pipeline: setup Milvus, prepare data, and insert."""
    print("Initializing Milvus upload pipeline...")

    # 1. Setup Milvus client
    client = MilvusClient(uri=CLUSTER_ENDPOINT, token=TOKEN)

    try:
        # 2. Create/recreate collection (pass force flag)
        setup_milvus_collection(client, COLLECTION_NAME, force=args.force)

        # 3. Prepare data
        payload = prepare_milvus_payload(args.data_root, args.subgraph_dir)

        # 4. Insert data
        insert_to_milvus(client, COLLECTION_NAME, payload)

        print("\nPipeline completed successfully.")
    finally:
        # 5. Close Milvus connection
        print("\nClosing Milvus connection...")
        client.close()
        print("Connection closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload scene embeddings to Milvus")
    parser.add_argument(
        "--data_root",
        type=str,
        default="EnTube/Download_2min",
        help="Path to directory containing segments.json files"
    )
    parser.add_argument(
        "--subgraph_dir",
        type=str,
        default="EnTube/SubgraphX_Results",
        help="Path to directory containing *_explanation.pt files"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force drop and recreate the Milvus collection if it already exists."
    )

    args = parser.parse_args()
    upload_embeddings(args)