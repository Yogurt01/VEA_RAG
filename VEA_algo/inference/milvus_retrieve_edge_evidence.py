"""
milvus_retrieve_edge_evidence.py
----------------------------------
Standalone (Phase 2 mới, dùng để test/đo retrieval độc lập trước khi chạy full
inference) — Với mỗi video test, truy vấn collection Milvus `rst_edge_index`
(xây bởi milvus_build_edge_index.py) để lấy evidence cho LLM, THAY THẾ HOÀN
TOÀN Neo4j exact-match + concept_id của pipeline ConceptRAG cũ. Logic ở đây
được milvus_inference_pipeline.py gọi lại trực tiếp (--evidence_mode edge/full).

Khác biệt cốt lõi so với bản cũ:
    - Không còn concept_id / K-Means -> không còn concept collapse.
    - Match bằng ANN similarity trên Milvus (COSINE) thay vì exact ID match
      -> luôn có candidate (giải quyết vấn đề "23/248 video không retrieve
      được" — dense retrieval không bao giờ trả về rỗng, khác exact match).
    - Tier 1: filter đúng rst_type trước khi search (ưu tiên đúng loại quan hệ)
      Tier 2: nếu Tier 1 rỗng (rst_type hiếm/lạ) -> search KHÔNG filter,
              vẫn dựa trên similarity nội dung scene, không phải fallback rời rạc.
    - Evidence trả về không phải 1 triple đơn lẻ mà là "local subgraph":
      matched edge (src--rst_type-->tgt) + các scene liền kề của nó ngay
      trong video training gốc, giúp LLM đọc được mạch chuyện, không chỉ 2 câu
      caption cô lập.

Usage — single video:
    python milvus_retrieve_edge_evidence.py \
        --data_root  /path/to/All_Videos \
        --edge_index_dir /path/to/edge_index \
        --video_id   b0f9d31bde3573c94ba3580a36af6b70 \
        --top_k 5 --candidate_limit 200

Usage — batch:
    python milvus_retrieve_edge_evidence.py \
        --data_root  /path/to/All_Videos \
        --edge_index_dir /path/to/edge_index \
        --split_file /path/to/dataset_splits.json \
        --n_samples 248 --top_k 5 --candidate_limit 200

Environment variables (Milvus — CHUNG cluster, collection khác):
    MILVUS_CLUSTER_ENDPOINT, MILVUS_TOKEN
    MILVUS_EDGE_COLLECTION_NAME (hoặc dùng --edge_collection_name)
"""

import os
import json
import argparse
import random
import collections
from pathlib import Path

import torch
from pymilvus import MilvusClient
from dotenv import load_dotenv

load_dotenv()

CLUSTER_ENDPOINT = os.getenv("MILVUS_CLUSTER_ENDPOINT")
TOKEN            = os.getenv("MILVUS_TOKEN")

SIM_WEIGHT   = 0.85   # trọng số cho cosine similarity trong final score
PRIOR_WEIGHT = 0.15   # trọng số cho prior theo rst_type


# ==========================================
# 0. RST NORMALIZATION (giữ đồng bộ với build script)
# ==========================================

def normalize_rst_type(rst_type: str) -> str:
    return str(rst_type).strip().upper().replace(" ", "_").replace("-", "_")


# ==========================================
# 1. CAPTION / DEPTH HELPERS (giống build script, để encode query edge)
# ==========================================

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


def compute_scene_depths(rst_links, n_scenes: int) -> list:
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
# 2. LOAD TEST VIDEO
# ==========================================

def load_video_test(video_id: str, data_root: Path):
    video_dir = data_root / video_id
    emb_path = video_dir / "scene_embeddings.pt"
    seg_path = video_dir / "segments.json"
    if not emb_path.exists() or not seg_path.exists():
        return None

    pt_data = torch.load(emb_path, map_location="cpu")
    embeddings = torch.nn.functional.normalize(pt_data["embeddings"].float(), p=2, dim=1)
    scene_ids_list = pt_data["scene_ids"]
    rst_links = pt_data.get("rst_links", [])
    label = pt_data.get("y", None)
    if label is not None:
        label = int(label.item() if isinstance(label, torch.Tensor) else label)

    with open(seg_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
    captions = load_captions_by_index(segments, scene_ids_list)
    depths = compute_scene_depths(rst_links, embeddings.shape[0])

    return {
        "video_id": video_id,
        "embeddings": embeddings,
        "scene_ids": scene_ids_list,
        "rst_links": rst_links,
        "captions": captions,
        "depths": depths,
        "label": label,
    }


def build_query_edges(video: dict) -> list:
    """Encode mỗi RST edge của video test thành query vector 4096-dim."""
    emb = video["embeddings"]
    n_scenes = emb.shape[0]
    queries = []
    for link in video["rst_links"]:
        try:
            raw_src, raw_tgt, raw_rst = link[0], link[1], link[2]
            s_idx, t_idx = int(raw_src) - 1, int(raw_tgt) - 1
        except Exception:
            continue
        if not (0 <= s_idx < n_scenes and 0 <= t_idx < n_scenes):
            continue
        rst_norm = normalize_rst_type(raw_rst)
        vec = torch.cat([emb[s_idx], emb[t_idx]], dim=0).tolist()
        queries.append({
            "vector": vec,
            "rst_type": rst_norm,
            "s_idx": s_idx,
            "t_idx": t_idx,
            "src_caption": video["captions"][s_idx] if s_idx < len(video["captions"]) else "",
            "tgt_caption": video["captions"][t_idx] if t_idx < len(video["captions"]) else "",
        })
    return queries


# ==========================================
# 3. MILVUS SEARCH (Tier 1 filter -> Tier 2 fallback)
# ==========================================

OUTPUT_FIELDS = [
    "video_id", "video_label", "rst_type", "src_caption", "tgt_caption",
    "src_scene_id", "tgt_scene_id", "depth_src", "depth_tgt",
]


def search_edge(client, collection_name: str, query_vec: list, rst_type: str, limit: int):
    """Tier 1: filter đúng rst_type. Tier 2: nếu rỗng, search không filter."""
    try:
        hits = client.search(
            collection_name=collection_name,
            data=[query_vec],
            filter=f'rst_type == "{rst_type}"',
            limit=limit,
            output_fields=OUTPUT_FIELDS,
        )[0]
    except Exception:
        hits = []

    tier = 1
    if not hits:
        tier = 2
        try:
            hits = client.search(
                collection_name=collection_name,
                data=[query_vec],
                limit=limit,
                output_fields=OUTPUT_FIELDS,
            )[0]
        except Exception:
            hits = []
    return hits, tier


def score_hit(hit: dict, prior_scores: dict) -> float:
    cos_sim = float(hit["distance"])  # COSINE metric -> Milvus trả trực tiếp similarity in [-1, 1]
    rst_type = hit["entity"]["rst_type"]
    prior = prior_scores.get(rst_type, 0.5)
    return SIM_WEIGHT * cos_sim + PRIOR_WEIGHT * prior


# ==========================================
# 4. LOCAL SUBGRAPH CONTEXT (mở rộng 1-hop trong video training gốc)
# ==========================================

def get_local_subgraph_context(video_id: str, src_scene_id: int, tgt_scene_id: int,
                                 data_root: Path, max_neighbors: int = 2) -> str:
    """
    Đọc lại rst_links + segments.json của video training gốc để lấy thêm
    1-2 scene liền kề với matched edge -> evidence là một đoạn subgraph nhỏ
    thay vì 1 triple cô lập.
    """
    try:
        emb_path = data_root / video_id / "scene_embeddings.pt"
        seg_path = data_root / video_id / "segments.json"
        pt_data = torch.load(emb_path, map_location="cpu")
        scene_ids_list = [int(x) for x in pt_data["scene_ids"]]
        rst_links = pt_data.get("rst_links", [])
        with open(seg_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
        captions = load_captions_by_index(segments, scene_ids_list)

        try:
            s_idx = scene_ids_list.index(src_scene_id)
            t_idx = scene_ids_list.index(tgt_scene_id)
        except ValueError:
            return ""

        neighbor_idxs = set()
        for link in rst_links:
            try:
                a, b, _ = int(link[0]) - 1, int(link[1]) - 1, link[2]
            except Exception:
                continue
            if a in (s_idx, t_idx) and b not in (s_idx, t_idx):
                neighbor_idxs.add(b)
            if b in (s_idx, t_idx) and a not in (s_idx, t_idx):
                neighbor_idxs.add(a)

        neighbor_idxs = list(neighbor_idxs)[:max_neighbors]
        if not neighbor_idxs:
            return ""
        lines = [f"      + Ngữ cảnh liền kề: {captions[i][:150]}" for i in neighbor_idxs if i < len(captions)]
        return "\n".join(lines)
    except Exception:
        return ""


# ==========================================
# 5. AGGREGATE + FORMAT EVIDENCE
# ==========================================

def retrieve_and_format(video: dict, client, collection_name: str, prior_scores: dict,
                          data_root: Path, top_k: int, candidate_limit: int):
    query_edges = build_query_edges(video)
    if not query_edges:
        return (f"[WARN] Video {video['video_id']} has no RST links -> no query edges.\n"), None

    best_per_video = {}   # video_id -> best hit dict
    raw_candidate_count = 0
    tier2_used = 0

    for qe in query_edges:
        hits, tier = search_edge(client, collection_name, qe["vector"], qe["rst_type"], candidate_limit)
        if tier == 2:
            tier2_used += 1
        raw_candidate_count += len(hits)
        for h in hits:
            score = score_hit(h, prior_scores)
            vid = h["entity"]["video_id"]
            if vid == video["video_id"]:
                continue  # tránh tự-match nếu video test lỡ nằm trong index
            if vid not in best_per_video or score > best_per_video[vid]["score"]:
                best_per_video[vid] = {
                    "score": score,
                    "video_label": h["entity"]["video_label"],
                    "rst_type": h["entity"]["rst_type"],
                    "src_caption": h["entity"]["src_caption"],
                    "tgt_caption": h["entity"]["tgt_caption"],
                    "src_scene_id": h["entity"]["src_scene_id"],
                    "tgt_scene_id": h["entity"]["tgt_scene_id"],
                }

    if not best_per_video:
        return (f"[WARN] No candidates found for video {video['video_id']} "
                f"(hiếm khi xảy ra với dense retrieval — kiểm tra collection có rỗng không).\n"), None

    ranked = sorted(best_per_video.items(), key=lambda kv: kv[1]["score"], reverse=True)
    top = ranked[:top_k]

    engaging_count = sum(1 for _, c in top if c["video_label"] == 1)
    not_engaging_count = sum(1 for _, c in top if c["video_label"] == 0)
    if engaging_count > not_engaging_count:
        leans = "engaging"
    elif not_engaging_count > engaging_count:
        leans = "not_engaging"
    else:
        leans = "mixed"
    avg_score = sum(c["score"] for _, c in top) / len(top)

    lines = [
        "=" * 70,
        f"EVIDENCE cho video: {video['video_id']}",
        "=" * 70,
        f"[Evidence Summary] {len(top)} video tham khảo | "
        f"Engaging: {engaging_count} | Not Engaging: {not_engaging_count} | "
        f"Nghiêng về: {leans.upper()} | avg_score={avg_score:.3f}",
        f"[Debug] raw_candidates={raw_candidate_count} | "
        f"tier2_fallback_used={tier2_used}/{len(query_edges)} query edges",
        "",
    ]
    for rank, (vid, c) in enumerate(top, 1):
        outcome = "Engaging" if c["video_label"] == 1 else "Not Engaging"
        lines.append(
            f"  #{rank} | Outcome: {outcome} | Relevance Score: {c['score']:.3f} | RST: {c['rst_type']}"
        )
        lines.append(f"      Scene A: {c['src_caption'][:200]}")
        lines.append(f"      Scene B: {c['tgt_caption'][:200]}")
        ctx = get_local_subgraph_context(vid, c["src_scene_id"], c["tgt_scene_id"], data_root)
        if ctx:
            lines.append(ctx)
        lines.append("")

    metrics = {
        "video_id": video["video_id"],
        "ground_truth": video["label"],
        "engaging_count": engaging_count,
        "not_engaging_count": not_engaging_count,
        "leans": leans,
        "raw_candidates": raw_candidate_count,
        "unique_top_k": len(top),
        "avg_score": avg_score,
    }
    return "\n".join(lines), metrics


# ==========================================
# 6. MAIN
# ==========================================

def main(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    edge_index_dir = Path(args.edge_index_dir)
    edge_collection_name = args.edge_collection_name or os.getenv("MILVUS_EDGE_COLLECTION_NAME", "rst_edge_index")

    if not all([CLUSTER_ENDPOINT, TOKEN]):
        raise ValueError("Missing MILVUS_CLUSTER_ENDPOINT / MILVUS_TOKEN env vars.")

    with open(edge_index_dir / "prior_scores.json", "r", encoding="utf-8") as f:
        prior_scores = json.load(f)

    client = MilvusClient(uri=CLUSTER_ENDPOINT, token=TOKEN)
    if not client.has_collection(edge_collection_name):
        raise RuntimeError(f"Collection '{edge_collection_name}' not found. Run build_edge_index_milvus.py first.")

    if args.video_id:
        video_ids = [args.video_id]
    elif args.split_file:
        with open(args.split_file, "r") as f:
            splits = json.load(f)
        test_ids = splits.get("test", [])
        if args.n_samples and args.n_samples < len(test_ids):
            random.seed(42)
            video_ids = random.sample(test_ids, args.n_samples)
        else:
            video_ids = test_ids
        print(f"[INFO] Processing {len(video_ids)} test videos.")
    else:
        raise ValueError("Provide either --video_id or --split_file.")

    all_metrics = []
    out_file = open(args.output_file, "w", encoding="utf-8") if args.output_file else None

    for vid in video_ids:
        video = load_video_test(vid, data_root)
        if video is None:
            print(f"[SKIP] {vid}: missing scene_embeddings.pt or segments.json")
            continue
        output, metrics = retrieve_and_format(
            video, client, edge_collection_name, prior_scores,
            data_root, args.top_k, args.candidate_limit,
        )
        print(output)
        if metrics is not None:
            all_metrics.append(metrics)
        if out_file:
            out_file.write(output + "\n")

    if out_file:
        out_file.close()
        print(f"[INFO] Full results saved to: {args.output_file}")

    # Batch metrics — giữ nguyên format để so sánh trực tiếp với pipeline cũ
    if all_metrics and args.split_file:
        print("\n" + "=" * 70)
        print(" BATCH METRICS SUMMARY (Milvus dense retrieval)")
        print("=" * 70)
        valid = [m for m in all_metrics if m["ground_truth"] is not None]
        y_true, y_pred = [], []
        for m in valid:
            if m["leans"] == "mixed":
                continue
            y_true.append(m["ground_truth"])
            y_pred.append(1 if m["leans"] == "engaging" else 0)
        total = len(y_true)
        print(f"Total videos (exclude mixed): {total}")
        print(f"Videos with NO evidence at all: {len(video_ids) - len(all_metrics)} "
              f"(kỳ vọng ~0 vì dense retrieval luôn trả kết quả)")
        if total > 0:
            from sklearn.metrics import confusion_matrix, classification_report
            correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
            print(f"Accuracy: {correct}/{total} = {correct/total:.4f}")
            cm = confusion_matrix(y_true, y_pred)
            print("Confusion Matrix:\n          Predicted\n          NotEng  Eng")
            print(f"Actual 0   {cm[0,0]:5d}  {cm[0,1]:5d}")
            print(f"       1   {cm[1,0]:5d}  {cm[1,1]:5d}")
            print(classification_report(y_true, y_pred, target_names=["Not Engaging", "Engaging"]))
        print("=" * 70)

    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve RST-edge evidence from Milvus dense index (thay Neo4j).")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--edge_index_dir", type=str, required=True,
                         help="Thư mục chứa prior_scores.json (từ build_edge_index_milvus.py).")
    parser.add_argument("--edge_collection_name", type=str, default=None)
    parser.add_argument("--video_id", type=str, default=None)
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--candidate_limit", type=int, default=50,
                         help="Top-K candidates lấy từ Milvus MỖI query edge (không phải mỗi video). "
                              "50 thường đủ vì đây là ANN similarity, không cần limit lớn như exact-match cũ.")
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()
    main(args)
