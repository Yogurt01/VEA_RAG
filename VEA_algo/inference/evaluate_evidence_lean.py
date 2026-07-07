"""
evaluate_evidence_lean.py
---------------------------
Công cụ đo accuracy của TỪNG kênh evidence riêng lẻ (Content Similarity /
Dense Edge Retrieval) SO VỚI ground truth trực tiếp — KHÔNG cần chạy LLM,
KHÔNG cần GPU. Dùng để sweep hyperparameter rẻ trước khi quyết định chạy
lại toàn bộ pipeline (tốn GPU/thời gian hơn nhiều).

Sweep được:
    - Content Similarity: --temperature của milvus_compute_video_query.py
      (chạy lại compute_video_query với --temperature khác nhau ra các
      --video_reps_dir khác nhau, rồi trỏ script này vào từng dir để so sánh)
    - Dense Edge Retrieval: --sim_weight / --prior_weight / --top_k_evidence /
      --candidate_limit (sweep trực tiếp qua tham số dòng lệnh, không cần
      build lại index)

Usage — đo content-lean:
    python evaluate_evidence_lean.py content \
        --data_root /path/to/All_Videos \
        --split_file /path/to/dataset_splits.json \
        --video_reps_dir /path/to/video_representations_temp0.1 \
        --top_k 5

Usage — đo narrative-lean, sweep trọng số:
    python evaluate_evidence_lean.py edge \
        --data_root /path/to/All_Videos \
        --split_file /path/to/dataset_splits.json \
        --edge_index_dir /path/to/edge_index \
        --sim_weight 0.85 --prior_weight 0.15 \
        --top_k_evidence 5 --candidate_limit 50

Environment variables (Milvus):
    MILVUS_CLUSTER_ENDPOINT, MILVUS_TOKEN
    MILVUS_COLLECTION_NAME        (cho content)
    MILVUS_EDGE_COLLECTION_NAME   (cho edge, mặc định 'rst_edge_index')
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from pymilvus import MilvusClient
from dotenv import load_dotenv

load_dotenv()

CLUSTER_ENDPOINT = os.getenv("MILVUS_CLUSTER_ENDPOINT")
TOKEN            = os.getenv("MILVUS_TOKEN")


# ==========================================
# SHARED HELPERS
# ==========================================

def normalize_rst_type(rst_type: str) -> str:
    return str(rst_type).strip().upper().replace(" ", "_").replace("-", "_")


def load_captions_by_index(segments, scene_ids_list, max_len: int = 990):
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


def evidence_lean(n_pos: int, n_total: int):
    if n_total == 0:
        return None
    n_neg = n_total - n_pos
    if n_pos == n_neg:
        return None
    return 1 if n_pos > n_neg else 0


def load_split_ids(split_file: Path, key: str) -> set:
    """Trả về set video_id thuộc 1 split (dùng để loại trừ khỏi candidate khi tuning trên val)."""
    with open(split_file, 'r') as f:
        return set(json.load(f).get(key, []))


def load_test_videos(data_root: Path, split_file: Path, split_key: str = "test") -> list:
    with open(split_file, 'r') as f:
        folders = json.load(f).get(split_key, [])
    valid = []
    for folder in folders:
        emb_path = data_root / folder / "scene_embeddings.pt"
        seg_path = data_root / folder / "segments.json"
        if emb_path.exists() and seg_path.exists():
            valid.append(folder)
    return valid


# ==========================================
# MODE: content
# ==========================================

def evaluate_content(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    video_reps_dir = Path(args.video_reps_dir)

    collection_name = args.content_collection_name or os.getenv("MILVUS_COLLECTION_NAME")
    if not all([CLUSTER_ENDPOINT, TOKEN, collection_name]):
        raise ValueError("Missing MILVUS_CLUSTER_ENDPOINT/MILVUS_TOKEN/collection name.")

    reps = torch.load(video_reps_dir / "video_representations.pt", map_location="cpu")
    test_folders = load_test_videos(data_root, Path(args.split_file), args.split_key)

    exclude_ids = set()
    if args.exclude_split_key:
        exclude_ids = load_split_ids(Path(args.split_file), args.exclude_split_key)
        print(f"[INFO] Excluding {len(exclude_ids)} videos from split '{args.exclude_split_key}' "
              f"from the candidate pool (clean holdout for tuning).")

    client = MilvusClient(uri=CLUSTER_ENDPOINT, token=TOKEN)
    correct, total, no_evidence = 0, 0, 0

    for folder in test_folders:
        if folder not in reps:
            no_evidence += 1
            continue
        emb_path = data_root / folder / "scene_embeddings.pt"
        pt_data = torch.load(emb_path, map_location="cpu")
        gt = int(pt_data["y"].item() if isinstance(pt_data["y"], torch.Tensor) else pt_data["y"])

        res = client.search(
            collection_name=collection_name,
            data=[reps[folder].tolist()],
            limit=args.search_limit,
            output_fields=["video_id", "video_label"],
        )
        # [FIX] dedupe theo video_id trước khi vote — khớp với milvus_inference_pipeline.py v3,
        # tránh 1 video có nhiều scene giống nhau chiếm hết top-K (bug đã phát hiện).
        best_per_video = {}
        for hits in res:
            for hit in hits:
                vid = hit["entity"]["video_id"]
                if vid == folder or vid in exclude_ids:
                    continue
                score = hit["distance"]
                if vid not in best_per_video or score > best_per_video[vid][0]:
                    best_per_video[vid] = (score, hit["entity"]["video_label"])
        ranked = sorted(best_per_video.values(), key=lambda x: x[0], reverse=True)[:args.top_k]

        n_pos = sum(1 for _, lbl in ranked if lbl == 1)
        lean = evidence_lean(n_pos, len(ranked))
        if lean is None:
            continue
        total += 1
        correct += int(lean == gt)

    client.close()
    print("=" * 60)
    print(f" CONTENT-LEAN ACCURACY  (split={args.split_key}, video_reps_dir={video_reps_dir.name}, "
          f"top_k={args.top_k}, search_limit={args.search_limit}, exclude={args.exclude_split_key or 'none'})")
    print("=" * 60)
    print(f" Videos evaluated : {total} (no representation: {no_evidence})")
    print(f" Accuracy         : {correct}/{total} = {correct/total:.4f}" if total else " No valid videos.")
    print("=" * 60)


# ==========================================
# MODE: edge
# ==========================================

EDGE_OUTPUT_FIELDS = ["video_id", "video_label", "rst_type"]


def compute_scene_depths(rst_links, n_scenes: int) -> list:
    import collections as _collections
    adj = _collections.defaultdict(set)
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
    queue = _collections.deque([anchor])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if depth[v] == -1:
                depth[v] = depth[u] + 1
                queue.append(v)
    return depth


def build_query_edges(embeddings_norm, rst_links, depths):
    n_scenes = embeddings_norm.shape[0]
    queries = []
    for link in rst_links:
        try:
            raw_src, raw_tgt, raw_rst = link[0], link[1], link[2]
            s_idx, t_idx = int(raw_src) - 1, int(raw_tgt) - 1
        except Exception:
            continue
        if not (0 <= s_idx < n_scenes and 0 <= t_idx < n_scenes):
            continue
        vec = torch.cat([embeddings_norm[s_idx], embeddings_norm[t_idx]], dim=0).tolist()
        queries.append({
            "vector": vec,
            "rst_type": normalize_rst_type(raw_rst),
            "depth_src": depths[s_idx] if s_idx < len(depths) else -1,
        })
    return queries


def search_edge(client, edge_collection_name, query_vec, rst_type, limit):
    try:
        hits = client.search(
            collection_name=edge_collection_name, data=[query_vec],
            filter=f'rst_type == "{rst_type}"', limit=limit,
            output_fields=EDGE_OUTPUT_FIELDS,
        )[0]
    except Exception:
        hits = []
    if not hits:
        try:
            hits = client.search(
                collection_name=edge_collection_name, data=[query_vec],
                limit=limit, output_fields=EDGE_OUTPUT_FIELDS,
            )[0]
        except Exception:
            hits = []
    return hits


def evaluate_edge(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    edge_index_dir = Path(args.edge_index_dir)
    edge_collection_name = args.edge_collection_name or os.getenv("MILVUS_EDGE_COLLECTION_NAME", "rst_edge_index")

    if not all([CLUSTER_ENDPOINT, TOKEN]):
        raise ValueError("Missing MILVUS_CLUSTER_ENDPOINT/MILVUS_TOKEN.")

    prior_path = edge_index_dir / "prior_scores.json"
    prior_scores = {}
    if prior_path.exists():
        with open(prior_path, 'r', encoding='utf-8') as f:
            prior_scores = json.load(f)

    test_folders = load_test_videos(data_root, Path(args.split_file), args.split_key)

    exclude_ids = set()
    if args.exclude_split_key:
        exclude_ids = load_split_ids(Path(args.split_file), args.exclude_split_key)
        print(f"[INFO] Excluding {len(exclude_ids)} videos from split '{args.exclude_split_key}' "
              f"from the candidate pool (clean holdout for tuning).")

    client = MilvusClient(uri=CLUSTER_ENDPOINT, token=TOKEN)
    correct, total, no_evidence = 0, 0, 0

    for folder in test_folders:
        emb_path = data_root / folder / "scene_embeddings.pt"
        seg_path = data_root / folder / "segments.json"
        pt_data = torch.load(emb_path, map_location="cpu")
        gt = int(pt_data["y"].item() if isinstance(pt_data["y"], torch.Tensor) else pt_data["y"])
        embeddings_norm = F.normalize(pt_data["embeddings"].float(), p=2, dim=1)
        n_scenes = embeddings_norm.shape[0]

        rst_links = pt_data.get("rst_links", [])
        depths = compute_scene_depths(rst_links, n_scenes) if args.depth_penalty_weight > 0 else [-1] * n_scenes

        query_edges = build_query_edges(embeddings_norm, rst_links, depths)
        if not query_edges:
            no_evidence += 1
            continue

        best_per_video = {}
        for qe in query_edges:
            hits = search_edge(client, edge_collection_name, qe["vector"], qe["rst_type"], args.candidate_limit)
            for h in hits:
                vid = h["entity"]["video_id"]
                if vid == folder or vid in exclude_ids:
                    continue
                rst_type = h["entity"]["rst_type"]
                prior = prior_scores.get(rst_type, 0.5)
                score = args.sim_weight * float(h["distance"]) + args.prior_weight * prior
                if args.depth_penalty_weight > 0 and qe["depth_src"] >= 0:
                    hit_depth_src = h["entity"].get("depth_src", -1)
                    if hit_depth_src >= 0:
                        penalty = args.depth_penalty_weight * min(abs(qe["depth_src"] - hit_depth_src), 3) / 3
                        score -= penalty
                if vid not in best_per_video or score > best_per_video[vid][0]:
                    best_per_video[vid] = (score, h["entity"]["video_label"])

        if not best_per_video:
            no_evidence += 1
            continue

        ranked = sorted(best_per_video.items(), key=lambda kv: kv[1][0], reverse=True)[:args.top_k_evidence]
        n_pos = sum(1 for _, (_, lbl) in ranked if lbl == 1)
        lean = evidence_lean(n_pos, len(ranked))
        if lean is None:
            continue
        total += 1
        correct += int(lean == gt)

    client.close()
    print("=" * 60)
    print(f" NARRATIVE-LEAN ACCURACY  (sim_weight={args.sim_weight}, prior_weight={args.prior_weight}, "
          f"top_k={args.top_k_evidence}, candidate_limit={args.candidate_limit}, "
          f"depth_penalty_weight={args.depth_penalty_weight})")
    print("=" * 60)
    print(f" Videos evaluated : {total} (no evidence: {no_evidence})")
    print(f" Accuracy         : {correct}/{total} = {correct/total:.4f}" if total else " No valid videos.")
    print("=" * 60)


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate evidence-lean accuracy without running the LLM.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_content = subparsers.add_parser("content", help="Evaluate Content Similarity lean accuracy.")
    p_content.add_argument("--data_root", type=str, required=True)
    p_content.add_argument("--split_file", type=str, required=True)
    p_content.add_argument("--content_collection_name", type=str, default=None,
                           help="Milvus collection cho Content Similarity. Mặc định env MILVUS_COLLECTION_NAME.")
    p_content.add_argument("--video_reps_dir", type=str, required=True)
    p_content.add_argument("--top_k", type=int, default=5)
    p_content.add_argument("--search_limit", type=int, default=50,
                           help="Số scene thô lấy trước khi dedupe theo video (nên >> top_k, "
                                "và cần LỚN HƠN khi dùng --exclude_split_key vì 1 phần candidate sẽ bị loại bỏ).")
    p_content.add_argument("--exclude_split_key", type=str, default=None,
                           help="Loại bỏ mọi candidate thuộc split này khỏi kết quả (ví dụ 'val') — "
                                "dùng khi đang sweep bằng query=val để mô phỏng index chỉ gồm train, "
                                "tránh val 'giúp' val. Để trống khi chạy thật trên test.")
    p_content.add_argument("--split_key", type=str, default="test",
                           help="Key trong split_file để lấy danh sách video ('test' hoặc 'val'). "
                                "Dùng 'val' khi đang sweep hyperparameter để tránh tune trực tiếp trên tập test.")

    p_edge = subparsers.add_parser("edge", help="Evaluate Dense Edge Retrieval lean accuracy.")
    p_edge.add_argument("--data_root", type=str, required=True)
    p_edge.add_argument("--split_file", type=str, required=True)
    p_edge.add_argument("--edge_index_dir", type=str, required=True)
    p_edge.add_argument("--edge_collection_name", type=str, default=None)
    p_edge.add_argument("--sim_weight", type=float, default=0.85)
    p_edge.add_argument("--prior_weight", type=float, default=0.15)
    p_edge.add_argument("--top_k_evidence", type=int, default=5)
    p_edge.add_argument("--candidate_limit", type=int, default=50)
    p_edge.add_argument("--depth_penalty_weight", type=float, default=0.0,
                        help="Trọng số phạt lệch vị trí trong mạch truyện. 0.0 = tắt (mặc định).")
    p_edge.add_argument("--exclude_split_key", type=str, default=None,
                        help="Loại bỏ mọi candidate thuộc split này khỏi kết quả (ví dụ 'val') — "
                             "dùng khi đang sweep bằng query=val để mô phỏng index chỉ gồm train. "
                             "Để trống khi chạy thật trên test.")
    p_edge.add_argument("--split_key", type=str, default="test",
                        help="Key trong split_file để lấy danh sách video ('test' hoặc 'val'). "
                             "Dùng 'val' khi đang sweep hyperparameter để tránh tune trực tiếp trên tập test.")

    args = parser.parse_args()
    if args.mode == "content":
        evaluate_content(args)
    else:
        evaluate_edge(args)
