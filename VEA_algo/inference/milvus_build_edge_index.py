"""
milvus_build_edge_index.py
---------------------------
Offline (Phase 1 mới) — Xây dựng "RST edge index" trong MỘT COLLECTION MILVUS
RIÊNG (khác collection scene-level đã có trong milvus_upload_scene_index.py),
thay thế hoàn toàn K-Means -> concept_id -> Neo4j của pipeline ConceptRAG cũ.
Đây là bước INDEX-SIDE của kênh Dense Edge Retrieval + Case-based Subgraph
Evidence (dùng cùng milvus_retrieve_edge_evidence.py / milvus_inference_pipeline.py
--evidence_mode edge/full).

Ý tưởng cốt lõi:
    Mỗi RST edge (src_scene, rst_type, tgt_scene) trong video train/val được
    biểu diễn bằng MỘT VECTOR LIÊN TỤC (không rời rạc hóa qua concept_id):

        edge_vector = concat( L2norm(emb_src) , L2norm(emb_tgt) )   # 4096-dim

    Vector này được upload lên Milvus (COSINE/HNSW) cùng các trường phụ:
        rst_type, video_id, video_label, src_caption, tgt_caption,
        depth_src, depth_tgt (DDE-lite: khoảng cách BFS tới node "gốc"
        trong RST tree của chính video đó).

    Vì đây là ANN + scalar filter trên Milvus, việc "match" không còn phụ
    thuộc vào việc hai scene có bị gán chung concept_id hay không -> giải
    quyết trực tiếp vấn đề concept collapse (57.1%) và triple sparsity
    (1.1% coverage) đã ghi nhận trong ConceptRAG_for_LLM_analysis.md.

Usage:
    python milvus_build_edge_index.py \
        --data_root  /content/drive/MyDrive/KhoaLuan/SnapUGC/All_Videos \
        --split_file /content/drive/MyDrive/KhoaLuan/SnapUGC/dataset_splits.json \
        --output_dir /content/drive/MyDrive/KhoaLuan/SnapUGC/edge_index \
        --edge_collection_name rst_edge_index \
        --force

Environment variables (Milvus — dùng CHUNG cluster với upload_milvus.py):
    MILVUS_CLUSTER_ENDPOINT, MILVUS_TOKEN
    (MILVUS_EDGE_COLLECTION_NAME có thể set thay cho --edge_collection_name)

Output:
    - Milvus collection `<edge_collection_name>` chứa toàn bộ RST edges train+val
    - <output_dir>/prior_scores.json  : prior xác suất Engaging theo rst_type
    - <output_dir>/edge_stats.json    : thống kê để kiểm tra chất lượng index
"""

import os
import json
import argparse
import collections
from pathlib import Path

import torch
from tqdm import tqdm
from pymilvus import MilvusClient, DataType
from dotenv import load_dotenv

load_dotenv()

CLUSTER_ENDPOINT = os.getenv("MILVUS_CLUSTER_ENDPOINT")
TOKEN            = os.getenv("MILVUS_TOKEN")
BATCH_SIZE       = 1000
EMB_DIM          = 2048          # dim của scene embedding Qwen3-VL-Embedding-2B
EDGE_DIM         = EMB_DIM * 2   # concat(src, tgt)


# ==========================================
# 0. RST TYPE NORMALIZATION (giữ nguyên quy ước cũ để tương thích)
# ==========================================

def normalize_rst_type(rst_type: str) -> str:
    return str(rst_type).strip().upper().replace(" ", "_").replace("-", "_")


# ==========================================
# 1. CAPTION LOADING (theo đúng index vị trí, không theo scene_id value)
# ==========================================

def load_captions_by_index(segments, scene_ids_list, max_len: int = 990):
    """
    Trả về list caption theo đúng thứ tự index (0-based) khớp với
    embeddings[i] / scene_ids_list[i]. Hỗ trợ cả segments dạng list và dict.
    """
    def _get(idx: int, scene_id_int: int) -> str:
        if isinstance(segments, list):
            entry = segments[idx] if idx < len(segments) else {}
        elif isinstance(segments, dict):
            entry = segments.get(str(scene_id_int)) or segments.get(scene_id_int, {})
        else:
            return ""
        caption = entry.get("caption", "") if isinstance(entry, dict) else str(entry)
        return caption[:max_len] + "..." if len(caption) > max_len else caption

    return [_get(i, int(sid)) for i, sid in enumerate(scene_ids_list)]


# ==========================================
# 2. STRUCTURAL DEPTH (DDE-lite) — thay cho DDE gốc cần entity linking
# ==========================================

def compute_scene_depths(rst_links, n_scenes: int) -> list:
    """
    BFS trên đồ thị VÔ HƯỚNG tạo từ toàn bộ RST edges của MỘT video, tính
    khoảng cách (số cạnh) từ một node "gốc" tới từng scene.

    Node gốc được chọn theo thứ tự ưu tiên:
        1. Node xuất hiện trong một RST edge có type == ROOT
        2. Nếu không có -> scene đầu tiên (index 0), coi như mở đầu narrative

    Trả về list depth (0-based theo index), depth = -1 nếu node không kết
    nối được tới gốc (đồ thị RST bị rời rạc — hiếm nhưng có thể xảy ra).
    """
    adj = collections.defaultdict(set)
    root_candidates = set()

    for link in rst_links:
        try:
            src, tgt, rst_type = link[0], link[1], link[2]
            s, t = int(src) - 1, int(tgt) - 1
        except Exception:
            continue
        if 0 <= s < n_scenes and 0 <= t < n_scenes:
            adj[s].add(t)
            adj[t].add(s)
            if normalize_rst_type(rst_type) == "ROOT":
                root_candidates.add(t)
                root_candidates.add(s)

    anchor = next(iter(root_candidates)) if root_candidates else 0

    depth = [-1] * n_scenes
    if n_scenes == 0:
        return depth
    depth[anchor] = 0
    queue = collections.deque([anchor])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if depth[v] == -1:
                depth[v] = depth[u] + 1
                queue.append(v)
    return depth


# ==========================================
# 3. MILVUS COLLECTION SETUP
# ==========================================

def setup_edge_collection(client, collection_name: str, force: bool = False) -> None:
    if client.has_collection(collection_name):
        if force:
            print(f"[INFO] Dropping existing collection '{collection_name}'...")
            client.drop_collection(collection_name)
        else:
            print(f"[INFO] Collection '{collection_name}' already exists. Skip (--force to recreate).")
            return

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="edge_uid",    datatype=DataType.VARCHAR, max_length=300, is_primary=True)
    schema.add_field(field_name="vector",      datatype=DataType.FLOAT_VECTOR, dim=EDGE_DIM)
    schema.add_field(field_name="video_id",    datatype=DataType.VARCHAR, max_length=200)
    schema.add_field(field_name="video_label", datatype=DataType.INT64)
    schema.add_field(field_name="rst_type",    datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="src_scene_id", datatype=DataType.INT64)
    schema.add_field(field_name="tgt_scene_id", datatype=DataType.INT64)
    schema.add_field(field_name="src_caption", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="tgt_caption", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="depth_src",   datatype=DataType.INT64)
    schema.add_field(field_name="depth_tgt",   datatype=DataType.INT64)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector", metric_type="COSINE", index_type="HNSW",
        params={"M": 16, "efConstruction": 64},
    )
    try:
        # Scalar index để filter theo rst_type nhanh hơn khi query (Tier-1 exact filter)
        index_params.add_index(field_name="rst_type", index_type="INVERTED")
    except Exception as e:
        print(f"[WARN] Could not add scalar index on rst_type (bỏ qua, filter vẫn hoạt động): {e}")

    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    print(f"[INFO] Collection '{collection_name}' created (vector dim={EDGE_DIM}).")


# ==========================================
# 4. BUILD EDGE RECORDS FROM ALL TRAIN/VAL VIDEOS
# ==========================================

def build_edge_records(data_root: Path, include_videos: set) -> tuple:
    """
    Trả về (records, prior_counts) trong đó:
        records      : list[dict] sẵn sàng insert vào Milvus
        prior_counts : dict {rst_type: {"pos": n, "total": n}} để tính prior
    """
    all_video_dirs = sorted([
        d for d in data_root.iterdir()
        if d.is_dir() and (d / "scene_embeddings.pt").exists()
    ])
    video_dirs = [d for d in all_video_dirs if d.name in include_videos]
    print(f"[INFO] Found {len(video_dirs)} video folders to process (out of {len(all_video_dirs)} total).")

    records = []
    prior_counts = collections.defaultdict(lambda: {"pos": 0, "total": 0})
    n_videos_ok, n_videos_failed, n_edges_skipped = 0, 0, 0

    for video_dir in tqdm(video_dirs, desc="Building edge vectors"):
        video_name = video_dir.name
        seg_path = video_dir / "segments.json"
        if not seg_path.exists():
            n_videos_failed += 1
            continue

        try:
            pt_data = torch.load(video_dir / "scene_embeddings.pt", map_location="cpu")
            label = int(pt_data["y"].item() if isinstance(pt_data["y"], torch.Tensor) else pt_data["y"])
            embeddings = pt_data["embeddings"].float()                     # (T, 2048)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            scene_ids_list = pt_data["scene_ids"]
            rst_links = pt_data.get("rst_links", [])
            n_scenes = embeddings.shape[0]

            with open(seg_path, "r", encoding="utf-8") as f:
                segments = json.load(f)
            captions = load_captions_by_index(segments, scene_ids_list)

            depths = compute_scene_depths(rst_links, n_scenes)

            for link in rst_links:
                try:
                    raw_src, raw_tgt, raw_rst = link[0], link[1], link[2]
                    s_idx, t_idx = int(raw_src) - 1, int(raw_tgt) - 1
                except Exception:
                    n_edges_skipped += 1
                    continue
                if not (0 <= s_idx < n_scenes and 0 <= t_idx < n_scenes):
                    n_edges_skipped += 1
                    continue

                rst_norm = normalize_rst_type(raw_rst)
                edge_vec = torch.cat([embeddings[s_idx], embeddings[t_idx]], dim=0).tolist()

                records.append({
                    "edge_uid":     f"{video_name}_e{s_idx}_{t_idx}_{rst_norm}",
                    "vector":       edge_vec,
                    "video_id":     video_name,
                    "video_label":  label,
                    "rst_type":     rst_norm,
                    "src_scene_id": int(scene_ids_list[s_idx]),
                    "tgt_scene_id": int(scene_ids_list[t_idx]),
                    "src_caption":  captions[s_idx] if s_idx < len(captions) else "",
                    "tgt_caption":  captions[t_idx] if t_idx < len(captions) else "",
                    "depth_src":    int(depths[s_idx]),
                    "depth_tgt":    int(depths[t_idx]),
                })
                prior_counts[rst_norm]["total"] += 1
                if label == 1:
                    prior_counts[rst_norm]["pos"] += 1

            n_videos_ok += 1
        except Exception as e:
            print(f"\n[ERROR] Failed to process {video_name}: {e}")
            n_videos_failed += 1

    print("\n" + "=" * 60)
    print("EDGE BUILD SUMMARY")
    print("=" * 60)
    print(f" Videos processed OK   : {n_videos_ok}")
    print(f" Videos failed/skipped : {n_videos_failed}")
    print(f" Edges skipped (bad idx): {n_edges_skipped}")
    print(f" Total edge records    : {len(records)}")
    print("=" * 60)
    return records, prior_counts


def insert_edges(client, collection_name: str, records: list, batch_size: int = BATCH_SIZE) -> None:
    if not records:
        print("[WARN] No edge records to insert.")
        return
    n_batches = (len(records) + batch_size - 1) // batch_size
    for i in tqdm(range(n_batches), desc="Inserting into Milvus"):
        batch = records[i * batch_size: (i + 1) * batch_size]
        client.insert(collection_name=collection_name, data=batch)
    client.flush(collection_name=collection_name)
    stats = client.get_collection_stats(collection_name)
    print(f"[INFO] Total edges stored in Milvus: {stats['row_count']}")


def compute_and_save_priors(prior_counts: dict, output_dir: Path) -> dict:
    """Prior Laplace-smoothed theo rst_type: P(video_label=1 | rst_type)."""
    priors = {
        rst: (c["pos"] + 1) / (c["total"] + 2)
        for rst, c in prior_counts.items()
    }
    with open(output_dir / "prior_scores.json", "w", encoding="utf-8") as f:
        json.dump(priors, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved prior_scores.json ({len(priors)} rst_types).")
    return priors


# ==========================================
# 5. MAIN
# ==========================================

def main(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not all([CLUSTER_ENDPOINT, TOKEN]):
        raise ValueError("Missing MILVUS_CLUSTER_ENDPOINT / MILVUS_TOKEN env vars.")

    edge_collection_name = args.edge_collection_name or os.getenv("MILVUS_EDGE_COLLECTION_NAME", "rst_edge_index")

    with open(args.split_file, "r") as f:
        splits = json.load(f)
    include_videos = set(splits.get("train", [])) | set(splits.get("val", []))
    print(f"[INFO] Train+Val videos to index: {len(include_videos)}")

    client = MilvusClient(uri=CLUSTER_ENDPOINT, token=TOKEN)
    try:
        setup_edge_collection(client, edge_collection_name, force=args.force)
        records, prior_counts = build_edge_records(data_root, include_videos)
        insert_edges(client, edge_collection_name, records)
        priors = compute_and_save_priors(prior_counts, output_dir)

        # Thống kê chẩn đoán nhanh — thay cho "diversity diagnostic" cũ
        rst_dist = {rst: c["total"] for rst, c in prior_counts.items()}
        with open(output_dir / "edge_stats.json", "w", encoding="utf-8") as f:
            json.dump({
                "edge_collection_name": edge_collection_name,
                "total_edges": len(records),
                "rst_type_distribution": rst_dist,
                "priors": priors,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n[SUCCESS] Edge index built in Milvus collection '{edge_collection_name}'.")
        print(f"[SUCCESS] Artifacts saved in: {output_dir}")
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RST edge similarity index in a dedicated Milvus collection.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--edge_collection_name", type=str, default=None,
                         help="Tên collection Milvus RIÊNG cho RST edges (khác collection scene-level). "
                              "Mặc định lấy từ env MILVUS_EDGE_COLLECTION_NAME hoặc 'rst_edge_index'.")
    parser.add_argument("--force", action="store_true", help="Drop & recreate edge collection nếu đã tồn tại.")
    args = parser.parse_args()
    main(args)
