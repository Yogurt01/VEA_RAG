
"""
stage1_retrieval.py
--------------------------------------------------------------------------------------------------
QUY TRÌNH 1 — RETRIEVAL ONLY.

Việc thực hiện:
    1. Load data (data_root + split_file).
    2. Với mỗi video test, chạy các truy vấn evidence (tuỳ --evidence_mode):
         - "milvus" / "full": Content Similarity (Cross-Modal Self-Querying) +
                               Discourse Pattern (Milvus dense search từng scene + NW re-rank)
         - "graph"  / "full": Neo4j structural matching (RST logic-chain) + Concept alignment
    3. Lưu lại TOÀN BỘ các thành phần dùng để build LLM prompt (text, lean, confidence, hits...)
       vào 1 file JSON duy nhất (--evidence_path).
    4. Kết thúc chương trình. KHÔNG load tokenizer/model, KHÔNG gọi LLM.

Chạy lại (resume) an toàn: nếu --evidence_path đã tồn tại, các folder đã có trong đó sẽ được bỏ qua.

Usage:
    python stage1_retrieval.py --evidence_mode full \
        --data_root .../All_Videos --split_file .../dataset_splits.json \
        --evidence_path .../evidence_full.json \
        --content_collection_name video_scenes_collection \
        --video_reps_dir .../video_representations_dir \
        --alpha 0.7 --content_top_k 5 --content_search_limit 50 \
        --discourse_top_k 5 --discourse_search_limit 30 --graph_top_k 5
"""

import os
import gc
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from pymilvus import MilvusClient
from neo4j import GraphDatabase
from dotenv import load_dotenv

import common_utils as cu

load_dotenv()

EVIDENCE_MODE_CHOICES = ["none", "milvus", "graph", "full"]


def dictify_neo4j_records(records: list) -> list:
    """Chuyển list các neo4j.Record thành list[dict] thuần Python để JSON-serializable."""
    return [dict(r) for r in records]


def build_graph_hits_record(graph_evidence: dict) -> dict:
    # Lấy video_details để có rst_summary
    video_details = graph_evidence.get("video_details", {})

    structural_matches = []
    for rec in graph_evidence.get("structural_matches", []):
        match_details = rec.get("match_details", [])
        rst_chain = match_details[0].get("relation_chain", []) if match_details else []
        vid = rec.get("video_id")
        structural_matches.append({
            "video_id": vid,
            "label": rec.get("label"),
            "max_nodes_matched": rec.get("max_nodes_matched"),
            "total_matched_sequences": rec.get("total_matched_sequences"),
            "rst_chain": rst_chain,
            "video_rst_summary": video_details.get(vid, {}).get("rst_summary", ""),
        })

    return {
        "structural_matches": structural_matches,
        "similarity_videos": [
            {
                "video_id": rec.get("video_id"),
                "label": rec.get("label"),
                "avg_score": rec.get("avg_score"),
            }
            for rec in graph_evidence.get("similarity_videos", [])
        ],
        "concept_ids": list(graph_evidence.get("concept_ids_set", set())),
        "concept_details": {
            cid: {
                "label_distribution": details.get("label_distribution"),
                "audio_style": details.get("audio_style")[:100] if details.get("audio_style") else "",
                "visual_style": details.get("visual_style")[:100] if details.get("visual_style") else "",
                "keywords": details.get("keywords", [])[:5],
            }
            for cid, details in graph_evidence.get("concept_details", {}).items()
        },
    }


def load_evidence_checkpoint(evidence_path: Path) -> tuple:
    evidence_results = []
    processed = set()
    if evidence_path.exists():
        try:
            with open(evidence_path, 'r', encoding='utf-8') as f:
                evidence_results = json.load(f)
            processed = {r['folder_name'] for r in evidence_results}
        except Exception:
            pass
    return evidence_results, processed


def main(args: argparse.Namespace) -> None:
    data_root      = Path(args.data_root)
    split_file     = Path(args.split_file)
    evidence_path  = Path(args.evidence_path)
    evidence_mode  = args.evidence_mode

    need_milvus = evidence_mode in ("milvus", "full")
    need_graph  = evidence_mode in ("graph", "full")

    milvus_client   = None
    collection_name = None
    reps_by_folder  = {}

    if need_milvus:
        milvus_endpoint = os.getenv("MILVUS_CLUSTER_ENDPOINT")
        milvus_token    = os.getenv("MILVUS_TOKEN")
        if not all([milvus_endpoint, milvus_token]):
            raise ValueError("Missing MILVUS_CLUSTER_ENDPOINT / MILVUS_TOKEN env vars.")
        milvus_client = MilvusClient(uri=milvus_endpoint, token=milvus_token)
        collection_name = args.content_collection_name or os.getenv("MILVUS_COLLECTION_NAME")

        if not args.video_reps_dir:
            raise ValueError("--video_reps_dir is required when evidence_mode='milvus'/'full'.")
        reps_by_folder = cu.load_video_representations(Path(args.video_reps_dir))

    neo4j_driver   = None
    neo4j_database = None
    if need_graph:
        neo4j_uri      = os.getenv("NEO4J_URI")
        neo4j_username = os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
        neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
        neo4j_driver.verify_connectivity()

    valid_folders, valid_data, _ = cu.load_valid_videos(data_root, split_file, reps_by_folder, require_reps=need_milvus)

    evidence_results, processed_folders = load_evidence_checkpoint(evidence_path)

    queue = [f for f in valid_folders if f not in processed_folders]
    print(f"[INFO] {len(queue)} / {len(valid_folders)} videos left to process (evidence_mode={evidence_mode}).")

    for current_idx, folder in enumerate(queue, 1):
        print(f"[{current_idx}/{len(queue)}] {folder}", end=" ", flush=True)
        sample_data = valid_data[folder]

        try:
            with open(data_root / folder / "segments.json", 'r', encoding='utf-8') as f:
                segments = json.load(f)
            captions = cu.load_captions_by_index(segments, sample_data['scene_ids'])
            video_context_text = cu.generate_input_video_context(folder, sample_data, data_root, evidence_mode)

            # ---------- (A) Milvus: Content Similarity + Discourse Pattern ----------
            content_text = discourse_text = None
            content_lean = content_conf = narrative_lean = narrative_conf = None
            content_hits_record = narrative_hits_record = []

            if need_milvus:
                rep_vec = reps_by_folder.get(folder)

                # 1) Content Similarity (Cross-Modal Self-Querying)
                res = milvus_client.search(
                    collection_name=collection_name,
                    data=[rep_vec.tolist()],
                    limit=args.content_search_limit,
                    output_fields=cu.DISCOURSE_OUTPUT_FIELDS,
                )
                top_hits = cu.dedupe_content_hits_by_video(res, args.content_top_k)
                content_text, content_lean, content_conf = cu.build_content_similarity_context(top_hits)
                content_hits_record = top_hits

                # 2) Discourse Pattern (Milvus dense search từng scene + topology re-rank)
                embeddings_norm = F.normalize(sample_data['embeddings'].float(), p=2, dim=1)
                query_dps_captions, ranked_top = cu.retrieve_discourse_evidence(
                    embeddings_norm, sample_data.get('rst_links', []), captions,
                    current_video_id=folder, milvus_client=milvus_client, collection_name=collection_name,
                    data_root=data_root, top_k=args.discourse_top_k,
                    search_limit=args.discourse_search_limit, alpha=args.alpha,
                )
                discourse_text, narrative_lean, narrative_conf = cu.build_discourse_context(query_dps_captions, ranked_top)
                narrative_hits_record = [{"video_id": vid, **cand} for vid, cand in ranked_top]

            # ---------- (B) Neo4j: Knowledge Graph structural matching ----------
            graph_text = None
            graph_lean = graph_conf = None
            graph_hits_record = {}

            if need_graph:
                try:
                    with neo4j_driver.session(database=neo4j_database) as session:
                        session.execute_write(cu.reconstruct_multimodal_subgraph_with_logs, folder, sample_data, data_root)
                        session.execute_write(cu.create_similarity_relationships_for_test, folder, top_k=3, threshold=0.85)

                    with neo4j_driver.session(database=neo4j_database) as session:
                        raw_graph_evidence = session.execute_read(cu.get_graph_evidence, folder, limit=args.graph_top_k)

                    # Chuyển neo4j.Record -> dict thuần trước khi build text / lưu JSON
                    graph_evidence = {
                        "structural_matches": dictify_neo4j_records(raw_graph_evidence.get("structural_matches", [])),
                        "similarity_videos":  dictify_neo4j_records(raw_graph_evidence.get("similarity_videos", [])),
                        "concept_priors":     dictify_neo4j_records(raw_graph_evidence.get("concept_priors", [])),
                        "video_details":      raw_graph_evidence.get("video_details", {}),
                        "concept_details":    raw_graph_evidence.get("concept_details", {}),
                        "concept_ids_set":    raw_graph_evidence.get("concept_ids_set", set()),
                    }

                    graph_text, graph_lean, graph_conf = cu.build_graph_context_v2(graph_evidence)
                    graph_hits_record = build_graph_hits_record(graph_evidence)
                finally:
                    with neo4j_driver.session(database=neo4j_database) as session:
                        session.execute_write(cu.cleanup_test_subgraph, folder)

            ground_truth = int(sample_data['y'].item() if isinstance(sample_data['y'], torch.Tensor) else sample_data['y'])

            evidence_results.append({
                "folder_name":            folder,
                "evidence_mode":          evidence_mode,
                "ground_truth":           ground_truth,
                "video_context_text":     video_context_text,
                "content_text":           content_text,
                "content_lean":           content_lean,
                "content_conf":           content_conf,
                "content_hits_record":    content_hits_record,
                "discourse_text":         discourse_text,
                "narrative_lean":         narrative_lean,
                "narrative_conf":         narrative_conf,
                "narrative_hits_record":  narrative_hits_record,
                "graph_text":             graph_text,
                "graph_lean":             graph_lean,
                "graph_conf":             graph_conf,
                "graph_hits_record":      graph_hits_record,
            })
            print("| OK")

        except Exception as e:
            print(f"| ERROR: {e}")
        finally:
            gc.collect()

        with open(evidence_path, 'w', encoding='utf-8') as f:
            json.dump(evidence_results, f, ensure_ascii=False, indent=2)

    if neo4j_driver is not None:
        neo4j_driver.close()

    print(f"[DONE] Saved evidence for {len(evidence_results)} videos -> {evidence_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 — Evidence retrieval only (no LLM)")
    parser.add_argument("--data_root",         type=str, required=True)
    parser.add_argument("--split_file",        type=str, required=True)
    parser.add_argument("--evidence_path",     type=str, required=True,
                         help="Where to save the evidence JSON (input for stage2_inference.py).")
    parser.add_argument("--evidence_mode",     type=str, default="full", choices=EVIDENCE_MODE_CHOICES)
    parser.add_argument("--content_collection_name", type=str, default=None)
    parser.add_argument("--video_reps_dir",    type=str, default=None)

    parser.add_argument("--content_top_k",        type=int, default=5)
    parser.add_argument("--content_search_limit", type=int, default=50)
    parser.add_argument("--discourse_top_k",      type=int, default=5)
    parser.add_argument("--discourse_search_limit", type=int, default=30)
    parser.add_argument("--alpha",                type=float, default=0.7)
    parser.add_argument("--graph_top_k",          type=int, default=5)

    args = parser.parse_args()
    main(args)
