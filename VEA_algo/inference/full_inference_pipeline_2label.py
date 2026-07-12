"""
full_inference_pipeline_2label.py
--------------------------------------------------------------------------------------------------
Hỗ trợ 4 evidence_mode: none | milvus | graph | full

Thiết kế 2 giai đoạn (giống cách dùng milvus trước đây), nay áp dụng cho CẢ milvus lẫn graph:

    1) PRECOMPUTE (CPU, rẻ tiền) — dùng --precompute_retrieval:
       Chạy toàn bộ truy vấn Milvus (content similarity + discourse pattern) và/hoặc
       Neo4j (structural / similarity / concept graph evidence), rồi lưu kết quả
       (text đã build sẵn + lean/confidence + hits_record) vào MỘT file JSON duy nhất:
           indexing/retrieval_cache.json
       Không load tokenizer/model, không cần GPU.

    2) INFERENCE (GPU) — chạy KHÔNG có --precompute_retrieval:
       Đọc lại indexing/retrieval_cache.json (không cần kết nối Milvus/Neo4j nếu đã có đủ cache),
       build lại LLM prompt, load model 1 lần, sinh dự đoán tuần tự, ensemble với
       lean/confidence đã cache -> final_prediction. Nếu 1 video chưa có trong cache (ví dụ mới
       thêm sau khi đã precompute), pipeline sẽ tự động fallback tính online (cần Milvus/Neo4j).

Usage (milvus, giữ nguyên như trước):
    # 1) precompute (CPU)
    python full_inference_pipeline_2label.py --evidence_mode milvus \
        --data_root .../All_Videos --split_file .../dataset_splits.json \
        --checkpoint_path .../ablation_milvus.json \
        --collection_name video_scenes_collection \
        --video_reps_dir .../video_representations_dir \
        --content_top_k 5 --content_search_limit 50 \
        --alpha 0.7 --discourse_top_k 5 --discourse_search_limit 30 \
        --retrieval_cache_path .../indexing/retrieval_cache.json --precompute_retrieval

    # 2) inference (GPU)
    python full_inference_pipeline_2label.py --evidence_mode milvus \
        ... (same args) ... \
        --model_name .../Qwen3-4B-Instruct-2507 \
        --retrieval_cache_path .../indexing/retrieval_cache.json

Usage (graph — mới):
    # 1) precompute (CPU, cần Neo4j)
    python full_inference_pipeline_2label.py --evidence_mode graph \
        --data_root .../Test_Graph --split_file .../dataset_splits.json \
        --checkpoint_path .../ablation_graph.json \
        --graph_top_k 5 \
        --retrieval_cache_path .../indexing/retrieval_cache.json --precompute_retrieval

    # 2) inference (GPU, không cần Neo4j nếu cache đã đủ)
    python full_inference_pipeline_2label.py --evidence_mode graph \
        --data_root .../Test_Graph --split_file .../dataset_splits.json \
        --checkpoint_path .../ablation_graph.json \
        --model_name .../Qwen3-4B-Instruct-2507 \
        --retrieval_cache_path .../indexing/retrieval_cache.json

Usage (full = milvus + graph, dùng chung 1 retrieval_cache.json):
    python full_inference_pipeline_2label.py --evidence_mode full \
        --data_root .../All_Videos --split_file .../dataset_splits.json \
        --checkpoint_path .../ablation_full.json \
        --collection_name video_scenes_collection --video_reps_dir .../video_representations_dir \
        --graph_top_k 5 \
        --retrieval_cache_path .../indexing/retrieval_cache.json --precompute_retrieval
    python full_inference_pipeline_2label.py --evidence_mode full \
        ... (same args) ... --model_name .../Qwen3-4B-Instruct-2507 \
        --retrieval_cache_path .../indexing/retrieval_cache.json
"""

import os
import re
import gc
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter, defaultdict, deque
from pymilvus import MilvusClient
from neo4j import GraphDatabase
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MAX_CONTEXT_TOKENS  = 32768
EVIDENCE_MODE_CHOICES = ["none", "milvus", "graph", "full"]


# ==========================================
# 1. RST NORMALIZATION & CONFIG
# ==========================================

def normalize_rst_type(rst_type: str) -> str:
    return str(rst_type).strip().upper().replace(" ", "_").replace("-", "_")


RST_DESCRIPTIONS = {
    'TEMPORAL':             'followed in time by',
    'JOINT':                'alongside',
    'ELABORATION':          'elaborates on',
    'SPAN':                 'continues with',
    'ROOT':                 'opens with',
    'CONTRAST':             'contrasts with',
    'CAUSE':                'causes',
    'BACKGROUND':           'sets the background for',
    'EXPLANATION':          'explains',
    'EVALUATION':           'evaluates',
    'TOPIC_COMMENT':        'comments on',
    'TOPIC_CHANGE':         'shifts topic from',
    'CONDITION':            'conditions',
    'ENABLEMENT':           'enables',
    'MANNER_MEANS':         'means of',
    'COMPARISON':           'compares with',
    'SAME_UNIT':            'continues from',
    'SUMMARY':              'summarizes',
    'TEXTUAL_ORGANIZATION': 'organizes',
    'ATTRIBUTION':          'attributed to',
}

CANONICAL_RST_TYPES = [
    "TEMPORAL",
    "JOINT",
    "ROOT",
    "ELABORATION",
    "SPAN",
    "CONTRAST",
    "TOPIC_COMMENT",
    "CAUSE",
    "BACKGROUND",
    "COMPARISON",
    "EVALUATION"
]


def rst_to_natural(rst_type: str) -> str:
    return RST_DESCRIPTIONS.get(rst_type.upper(), rst_type.lower().replace('_', ' '))


# ==========================================
# 2. DISCOURSE PATH SERIALIZATION (DPS)
# ==========================================

def serialize_discourse_captions(scene_ids: list, rst_links: list, captions_list: list) -> str:
    """
    Tạo cấu trúc phân cấp DPS cho toàn bộ video (dùng cho input video context).
    """
    if not rst_links:
        return "\n".join([f"  Scene {sid}: \"{cap[:150]}\"" for sid, cap in zip(scene_ids, captions_list)])

    adj = {}
    parent_map = {}
    relations = {}

    for src, tgt, rel in rst_links:
        src, tgt = int(src), int(tgt)
        rel_norm = normalize_rst_type(rel)
        adj.setdefault(tgt, []).append((src, rel_norm))
        parent_map[src] = tgt
        relations[(src, tgt)] = rel_norm

    roots = [sid for sid in scene_ids if sid not in parent_map]
    if not roots:
        roots = [scene_ids[0]] if scene_ids else []

    visited = set()
    lines = []

    def dfs(node, depth=0):
        if node in visited:
            return
        visited.add(node)
        try:
            idx = scene_ids.index(node)
            cap = captions_list[idx]
        except ValueError:
            cap = "No caption available."
        cap_short = cap[:180] + "..." if len(cap) > 180 else cap
        indent = "  " * depth

        if node in parent_map:
            p = parent_map[node]
            rel = relations.get((node, p), "SPAN")
            rel_nat = rst_to_natural(rel)
            lines.append(f"{indent}- Scene {node} ({rel_nat} Scene {p}): \"{cap_short}\"")
        else:
            lines.append(f"{indent}- Scene {node} [Narrative Core/Root]: \"{cap_short}\"")

        if node in adj:
            for child, _ in adj[node]:
                dfs(child, depth + 1)

    for r in roots:
        dfs(r)

    for sid in scene_ids:
        if sid not in visited:
            try:
                idx = scene_ids.index(sid)
                cap = captions_list[idx]
            except ValueError:
                cap = "No caption available."
            cap_short = cap[:180] + "..." if len(cap) > 180 else cap
            lines.append(f"- Scene {sid} [Isolated]: \"{cap_short}\"")

    return "\n".join(lines)


# ==========================================
# 3. LABEL DEFINITIONS
# ==========================================

LABEL_DEFINITIONS = {
    0: "Low Engagement — The video fails to attract or retain viewers; narrative flow is weak or incoherent.",
    1: "High Engagement — The video effectively utilises multimodal elements and discourse structure to attract and retain viewers.",
}

_label_def_lines  = "\n".join(
    f"  Label {lbl}: {desc}" for lbl, desc in sorted(LABEL_DEFINITIONS.items())
)
_valid_labels_str = " or ".join(str(k) for k in sorted(LABEL_DEFINITIONS.keys()))


# ==========================================
# 4. SYSTEM PROMPT
# ==========================================

REFERENCE_SOURCE_DOCS = {
    "milvus": """1. CONTENT SIMILARITY REFERENCE (PRIMARY SOURCE):
   - These are the most visually and thematically similar REFERENCE VIDEOS found in the
     library (deduplicated — each is a distinct video, not repeated scenes from the same video).
   - This serves as your primary external baseline for evaluation, as it reflects solid thematic alignment.
   - Each block reports its own internal agreement (strong / weak / mixed). Treat "weak" or "mixed" as low-confidence.
   2. DISCOURSE PATTERN REFERENCE (SECONDARY/SUPPLEMENTARY SOURCE):
   - These results come from matching individual scenes of the input video against a library
     of scenes using BOTH content similarity AND discourse structure similarity.
   - This block is purely supplementary and should be used as secondary reference context 
     rather than an equal decision-making signal to content similarity.""",
    "graph": """3. KNOWLEDGE GRAPH EVIDENCE (from Neo4j):
   Three sources from training corpus:

   a) **RST Structural Matches** — videos with same RST discourse chain. Shows label, max nodes, total paths, RST chain,
      and Video RST Summary (total relations, dominant type, distribution). Use to compare narrative structure.

   b) **Similarity-based Neighbors** — videos with semantically similar scenes (cosine similarity).
      Shows label and avg similarity score. Indicates thematic resemblance.

   c) **Concept Details** — semantic clusters from matched scenes. Shows label distribution (prior probability), Audio/Visual Style, and Keywords.
      Strong distribution (>70% one label) is a powerful signal.

   Use strong consensus across sources as high-confidence evidence. Video RST Summary connects structure to content.
""",
}

CONFLICT_RESOLUTION_NOTE = (
    "\nIn full mode, you must give significantly more weight and priority to the CONTENT SIMILARITY REFERENCE "
    "over the DISCOURSE PATTERN REFERENCE. Content similarity provides a well-balanced benchmark of overall thematic "
    "alignment, whereas the discourse reference should only act as a secondary, auxiliary source to assist in "
    "calibrating borderline cases or adding context."
)

GRAPH_USAGE_NOTE = (
    "\nWhen using the Knowledge Graph evidence, prioritize Concept Details first "
    "(they give you a prior distribution of engagement labels from semantically similar clusters), "
    "then RST structural matches (to confirm narrative logic), and finally similarity-based neighbors "
    "(as supplementary thematic cues). "
    "Use the Video RST Summary to understand the overall discourse structure of matched videos, "
    "and compare it with the input video's own structure (Section 1) to assess structural similarity."
)

CALIBRATION_NOTE = (
    "- Judge the video on its own specific merits. Both High Engagement and Low Engagement "
    "are equally valid default-free outcomes — each requires you to point to a concrete, "
    "identifiable reason in the video (or in the references) rather than an assumption in "
    "either direction."
)

CONTENT_PATTERN_NOTES = """
### CORE DISTINCTION CRITERIA

To accurately separate High-Engagement (Label 1) from Low-Engagement (Label 0), evaluate the structural progression of the captions across these 4 dimensions:

| Dimension | High Engagement (Label 1) | Low Engagement (Label 0) |
| :--- | :--- | :--- |
| **1. Hook & Topic Focus** | Introduces a clear, specific favorite item, central claim, or a unique process immediately in the opening scenes. | Has a flat, generic, or slow introduction with no clear central subject or purpose established early on. |
| **2. Narrative Progression** | The sequence of scenes shows a clear purpose, logical cause-effect, or a structured build-up (e.g., presenting a choice, step-by-step creation, or transformation). | The sequence is static, repetitive, or unorganized, simply piling up scenes without any clear logical development or climax. |
| **3. Content Specificity** | Elaborates with high-sensory details, unique flavor profiles, surprising contrasts, or strong reactions/outcomes later in the sequence. | Remains shallow or uniformly vague throughout; lacks detailed elaboration, unique selling points, or definitive outcomes. |
| **4. Structural Dynamics** | Creates a complete meaningful loop (e.g., establishing a plan -> execution -> reaction, or setting a boundary -> conflict -> resolution). | Feels incomplete, randomly cut off, or disjointed, failing to connect the opening premise to a satisfying conclusion. |

*Note: Since you are reading textual captions, do not penalize a video just because its scenes describe daily or routine tasks. Instead, look closely at whether those tasks are organized into a purposeful, high-progression sequence (Label 1) or remain a flat, directionless compilation (Label 0).*
"""

def build_system_prompt(evidence_mode: str) -> str:
    active_docs = []
    intro_parts = []

    # ---- none / milvus: giữ nguyên logic gốc ----
    if evidence_mode in ("milvus", "full"):
        active_docs.append(REFERENCE_SOURCE_DOCS["milvus"])
        intro_parts.append("similar contexts retrieved from a reference library")

    # ---- graph / full: thêm Knowledge Graph reference ----
    if evidence_mode in ("graph", "full"):
        active_docs.append(REFERENCE_SOURCE_DOCS["graph"])
        intro_parts.append("structural evidence from a Knowledge Graph")

    if active_docs:
        intro = " cross-reference it with " + " and ".join(intro_parts) + ","
        reference_block = "How to use the reference source(s) below:\n\n" + "\n\n".join(active_docs)
        if evidence_mode in ("milvus", "full"):
            reference_block += "\n" + CONFLICT_RESOLUTION_NOTE
        if evidence_mode == "graph":
            reference_block += GRAPH_USAGE_NOTE
        reasoning_intro = "- Read the video content structured via DPS first, then use the references as supporting evidence."
    else:
        intro = ""
        reference_block = "No external reference library is used in this setting — base your decision solely on the video content described below."
        reasoning_intro = "- Base your decision solely on the input video's own hierarchical structured content."

    return f"""You are an Advanced Video Analysis and Evaluation System.
Your task is to receive the structural, textual, and multimodal information of an input video,{intro}
and provide a final engagement prediction.

Label definitions:
{_label_def_lines}
IMPORTANT: Only two labels exist — 0 and 1.

{CONTENT_PATTERN_NOTES}

{reference_block}

Reasoning instructions:
{reasoning_intro}
{CALIBRATION_NOTE}
- Be specific: reference scene numbers and explain why the narrative works or doesn't.
- Write your explanation in plain language — no system names, no technical scores.
"""


# ==========================================
# 5. CORE MATHEMATICS & UTILS
# ==========================================

def evidence_lean_and_confidence(n_pos: int, n_total: int):
    if n_total == 0:
        return None, "none"
    n_neg = n_total - n_pos
    diff = abs(n_pos - n_neg)
    if n_pos == n_neg:
        return None, "mixed"
    lean = 1 if n_pos > n_neg else 0
    confidence = "weak" if diff <= 1 else "strong"
    return lean, confidence


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


def compute_topology_vector(rst_links: list) -> list:
    counts = Counter()
    for link in rst_links:
        try:
            counts[normalize_rst_type(link[2])] += 1
        except Exception:
            continue
    total = sum(counts.get(t, 0) for t in CANONICAL_RST_TYPES)
    if total == 0:
        return [0.0] * len(CANONICAL_RST_TYPES)
    return [counts.get(t, 0) / total for t in CANONICAL_RST_TYPES]


def cosine_sim(v1: list, v2: list) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def load_candidate_video_data(video_id: str, data_root: Path, cache: dict) -> dict:
    if video_id in cache:
        return cache[video_id]
    try:
        emb_path = data_root / video_id / "scene_embeddings.pt"
        seg_path = data_root / video_id / "segments.json"
        pt_data = torch.load(emb_path, map_location="cpu")
        scene_ids_list = [int(x) for x in pt_data["scene_ids"]]
        rst_links = pt_data.get("rst_links", [])
        with open(seg_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        raw_captions = load_captions_by_index(segments, scene_ids_list)
        data = {
            "scene_ids":       scene_ids_list,
            "raw_captions":    raw_captions,
            "topology_vector": compute_topology_vector(rst_links),
            "rst_links":       rst_links,
            "parent_map":      {},   # sẽ xây sau
            "children":        {},   # sẽ xây sau
        }
        # Xây parent_map và children từ rst_links
        parent_map = {}
        children = defaultdict(list)
        for src, tgt, rel in rst_links:
            src, tgt = int(src), int(tgt)
            parent_map[src] = (tgt, normalize_rst_type(rel))
            children[tgt].append((src, normalize_rst_type(rel)))
        data["parent_map"] = parent_map
        data["children"] = dict(children)
    except Exception:
        data = None
    cache[video_id] = data
    return data


def get_linked_scene_info(scene_id: int, cand_data: dict, captions: list) -> tuple:
    """
    Trả về (linked_scene_id, relation, caption) của scene liên kết trực tiếp
    (ưu tiên cha, nếu không có thì con đầu tiên). Nếu không có, trả về (None, None, None).
    """
    parent_map = cand_data.get("parent_map", {})
    children = cand_data.get("children", {})

    # Ưu tiên cha
    if scene_id in parent_map:
        parent_id, rel = parent_map[scene_id]
        if parent_id in cand_data["scene_ids"]:
            try:
                idx = cand_data["scene_ids"].index(parent_id)
                cap = captions[idx] if idx < len(captions) else ""
                return parent_id, rel, cap
            except ValueError:
                pass
    # Nếu không có cha, lấy con đầu tiên
    if scene_id in children and children[scene_id]:
        child_id, rel = children[scene_id][0]
        if child_id in cand_data["scene_ids"]:
            try:
                idx = cand_data["scene_ids"].index(child_id)
                cap = captions[idx] if idx < len(captions) else ""
                return child_id, rel, cap
            except ValueError:
                pass
    return None, None, None


# ==========================================
# 6. CONTENT SIMILARITY (chỉ show caption scene match)
# ==========================================

def dedupe_content_hits_by_video(search_results, top_k: int) -> list:
    best_per_video = {}
    for hits in search_results:
        for hit in hits:
            entity = hit['entity']
            vid = entity.get('video_id', 'N/A')
            score = hit['distance']
            if vid not in best_per_video or score > best_per_video[vid]['score']:
                best_per_video[vid] = {
                    'score':       score,
                    'scene_uid':   entity.get('scene_uid') or hit.get('id'),
                    'video_label': entity.get('video_label', 'N/A'),
                    'caption':     entity.get('caption', ''),
                }
    ranked = sorted(best_per_video.values(), key=lambda x: x['score'], reverse=True)
    return ranked[:top_k]


def build_content_similarity_context(top_hits: list) -> tuple:
    """Chỉ hiển thị caption của scene được match (không DPS, không video_id)."""
    if not top_hits:
        return "No similar content found in the reference library.", None, "none"

    label_counts = Counter(h['video_label'] for h in top_hits if h.get('video_label') is not None)
    n_pos = label_counts.get(1, 0)
    lean, confidence = evidence_lean_and_confidence(n_pos, sum(label_counts.values()))

    lines = [
        f"Label distribution across {len(top_hits)} distinct reference videos: "
        f"{', '.join(f'Label {lbl}: {cnt}' for lbl, cnt in sorted(label_counts.items()))}  [agreement: {confidence}]",
        ""
    ]

    for idx, hit in enumerate(top_hits, 1):
        s_uid = hit.get('scene_uid', '')
        scene_num = s_uid.split('_')[-1] if '_' in s_uid else "?"
        lines.append(
            f"Match {idx} (Similarity: {hit['score']:.4f}) — "
            f"Reference label: {hit['video_label']}\n"
            f"  Scene {scene_num}: {hit['caption']}"
        )
    return "\n".join(lines), lean, confidence


# ==========================================
# 7. DISCOURSE PATTERNS (chỉ hiển thị scene match + 1 scene liên kết)
# ==========================================

DISCOURSE_OUTPUT_FIELDS = ["scene_uid", "video_id", "video_label", "caption"]

def retrieve_discourse_evidence(
    embeddings_norm: torch.Tensor, rst_links: list,
    current_video_id: str, milvus_client, collection_name: str, data_root: Path,
    top_k: int, search_limit: int, alpha: float, video_cache: dict
) -> list:
    n_scenes = embeddings_norm.shape[0]
    if n_scenes == 0:
        return []

    query_topology = compute_topology_vector(rst_links)
    best_per_video = {}

    for scene_idx in range(n_scenes):
        query_vec = embeddings_norm[scene_idx].tolist()
        try:
            res = milvus_client.search(
                collection_name=collection_name, data=[query_vec], limit=search_limit, output_fields=DISCOURSE_OUTPUT_FIELDS
            )
        except Exception:
            res = []

        hits = res[0] if res else []
        for h in hits:
            vid = h["entity"]["video_id"]
            if vid == current_video_id:
                continue
            cand_data = load_candidate_video_data(vid, data_root, video_cache)
            if cand_data is None:
                continue

            dense_sim = float(h["distance"])
            topo_sim = cosine_sim(query_topology, cand_data["topology_vector"])
            final_score = alpha * dense_sim + (1 - alpha) * topo_sim

            if vid not in best_per_video or final_score > best_per_video[vid]["score"]:
                # Lấy scene match
                s_uid = h["entity"].get("scene_uid", "")
                matched_scene_id = None
                try:
                    matched_scene_id = int(s_uid.rsplit("_", 1)[-1]) if '_' in s_uid else None
                except:
                    pass
                if matched_scene_id is None or matched_scene_id not in cand_data["scene_ids"]:
                    # fallback: dùng caption để tìm
                    matched_caption = h["entity"].get("caption", "")
                    for sid, cap in zip(cand_data["scene_ids"], cand_data["raw_captions"]):
                        if matched_caption[:40] in cap or cap[:40] in matched_caption:
                            matched_scene_id = sid
                            break
                if matched_scene_id is None and cand_data["scene_ids"]:
                    matched_scene_id = cand_data["scene_ids"][0]

                # Lấy caption scene match
                try:
                    idx = cand_data["scene_ids"].index(matched_scene_id)
                    matched_caption = cand_data["raw_captions"][idx]
                except:
                    matched_caption = h["entity"].get("caption", "")

                # Lấy scene liên kết trực tiếp (cha hoặc con)
                linked_id, rel, linked_caption = get_linked_scene_info(matched_scene_id, cand_data, cand_data["raw_captions"])

                best_per_video[vid] = {
                    "score":           final_score,
                    "dense_sim":       dense_sim,
                    "topology_sim":    topo_sim,
                    "video_label":     h["entity"]["video_label"],
                    "matched_scene_id": matched_scene_id,
                    "matched_caption":  matched_caption,
                    "linked_scene_id":  linked_id,
                    "linked_caption":   linked_caption,
                    "relation":         rel,
                }

    ranked = sorted(best_per_video.items(), key=lambda kv: kv[1]["score"], reverse=True)
    return ranked[:top_k]


def build_discourse_context(ranked_top: list) -> tuple:
    """Hiển thị scene match + scene liên kết (không video_id)."""
    if not ranked_top:
        return "No matching discourse patterns found in the reference library.", None, "none"

    n_eng = sum(1 for _, c in ranked_top if c['video_label'] == 1)
    lean, confidence = evidence_lean_and_confidence(n_eng, len(ranked_top))

    lines = [f"Top {len(ranked_top)} matching discourse patterns from reference library:", ""]
    for rank, (vid, cand) in enumerate(ranked_top, 1):
        outcome = "HIGH ENGAGEMENT" if cand['video_label'] == 1 else "LOW ENGAGEMENT"
        lines.append(
            f"[{rank}] Outcome: {outcome}  |  Blended score: {cand['score']:.4f} "
            f"(dense={cand['dense_sim']:.3f}, topology={cand['topology_sim']:.3f})"
        )
        # Hiển thị scene match
        lines.append(f'     Match Scene {cand["matched_scene_id"]}: "{cand["matched_caption"][:150]}"')
        # Hiển thị scene liên kết nếu có
        if cand["linked_scene_id"] is not None and cand["linked_caption"]:
            rel_nat = rst_to_natural(cand["relation"]) if cand["relation"] else "related to"
            lines.append(f'     Linked Scene {cand["linked_scene_id"]} ({rel_nat}): "{cand["linked_caption"][:150]}"')
        lines.append("")

    n_neng = len(ranked_top) - n_eng
    lines.append(
        f"Label distribution across {len(ranked_top)} distinct reference videos: "
        f"Label 1: {n_eng}, Label 0: {n_neng}  [agreement: {confidence}]"
    )
    return "\n".join(lines), lean, confidence


# ==========================================
# 7b. KNOWLEDGE GRAPH (Neo4j) — structural / concept evidence [MODE MỚI]
# ==========================================

def explain_rst_chain(chain: list) -> str:
    """
    Chuyển danh sách các quan hệ RST thành chuỗi có kèm giải nghĩa.
    Ví dụ: ['ROOT', 'TEMPORAL'] -> 'ROOT (opens with) -> TEMPORAL (followed in time by)'
    """
    if not chain:
        return "N/A"
    explained = []
    for rel in chain:
        rel_upper = rel.upper()
        meaning = RST_DESCRIPTIONS.get(rel_upper, rel_upper.lower())
        explained.append(f"{rel_upper} ({meaning})")
    return " -> ".join(explained)


def create_similarity_relationships_for_test(tx, test_video_id: str, top_k: int = 5, threshold: float = 0.7):
    query = """
    MATCH (v_test:Video {id: $test_video_id, is_test: true})-[:HAS_SCENE]->(test_scene:Scene)
    CALL db.index.vector.queryNodes('scene_embeddings_idx', $top_k, test_scene.embedding)
    YIELD node AS db_scene, score
    WHERE db_scene.uid <> test_scene.uid
      AND score >= $threshold
    MATCH (db_scene)<-[:HAS_SCENE]-(db_video:Video)
    WHERE db_video.is_test IS NULL OR db_video.is_test = false
    MERGE (test_scene)-[r:SIMILAR_TO]->(db_scene)
    SET r.cosine_score = score
    """
    tx.run(query, test_video_id=test_video_id, top_k=top_k, threshold=threshold)


def reconstruct_multimodal_subgraph_with_logs(tx, video_id: str, data: dict, data_root: Path) -> None:
    seg_path = data_root / video_id / "segments.json"
    segments_list = []
    if seg_path.exists():
        try:
            with open(seg_path, 'r', encoding='utf-8') as f:
                segments_list = json.load(f)
        except Exception as e:
            print(f"  [WARNING] Failed to load segments.json for {video_id}: {e}")

    tx.run(
        "MERGE (v:Video {id: $video_id}) SET v.is_test = true, v.updated_at = timestamp()",
        video_id=video_id,
    )

    scene_query = """
    MATCH (v:Video {id: $video_id})
    MERGE (s:Scene {uid: $scene_uid})
    SET s.scene_id = $scene_id,
        s.caption = $caption,
        s.embedding = $embedding
    MERGE (v)-[:HAS_SCENE]->(s)
    """
    embeddings_list = data['embeddings'].tolist()
    scene_ids_list = data['scene_ids']

    for idx, scene_id in enumerate(scene_ids_list):
        scene_id_int = int(scene_id)
        caption_val = "No caption available"
        if idx < len(segments_list):
            caption_val = segments_list[idx].get('caption', caption_val)

        tx.run(
            scene_query,
            video_id=video_id,
            scene_uid=f"{video_id}_scene_{scene_id_int}",
            scene_id=scene_id_int,
            caption=caption_val,
            embedding=embeddings_list[idx],
        )

    for src, tgt, rel_type in data.get('rst_links', []):
        rel_type_upper = normalize_rst_type(rel_type)
        link_query = f"""
        MATCH (src:Scene {{uid: $src_uid}}), (tgt:Scene {{uid: $tgt_uid}})
        MERGE (src)-[:{rel_type_upper}]->(tgt)
        """
        tx.run(link_query, src_uid=f"{video_id}_scene_{int(src)}", tgt_uid=f"{video_id}_scene_{int(tgt)}")


def cleanup_test_subgraph(tx, video_id: str) -> None:
    tx.run(
        """
        MATCH (v:Video {id: $id})
        OPTIONAL MATCH (v)-[:HAS_SCENE]->(s:Scene)
        DETACH DELETE v, s
        """,
        id=video_id,
    )


def get_graph_evidence(tx, test_video_id: str, limit: int = 10) -> dict:
    """
    Trả về:
        {
            "structural_matches": [...],
            "similarity_videos": [...],
            "concept_priors": [...],
            "video_details": {...},
            "concept_details": {...},
            "concept_ids_set": {...},
        }
    """
    result = {}

    # ====== Structural matches ======
    structural_query = """
    MATCH (v_test:Video {id: $test_video_id})
    MATCH (v_test)-[:HAS_SCENE]->(tsStart:Scene)
    MATCH p = (tsStart)-[
        :SIMILAR_TO|TEMPORAL|ELABORATION|CONTRAST|SPAN|ROOT|JOINT|CAUSE|
        TOPIC_COMMENT|EXPLANATION|EVALUATION|BACKGROUND|TOPIC_CHANGE|
        ATTRIBUTION|TEXTUAL_ORGANIZATION|COMPARISON|SUMMARY|SAME_UNIT|
        CONDITION|ENABLEMENT|MANNER_MEANS*2..4
    ]->(tsEnd:Scene)
    WHERE (v_test)-[:HAS_SCENE]->(tsEnd)
      AND all(s IN nodes(p) WHERE (v_test)-[:HAS_SCENE]->(s))
    WITH p
    ORDER BY length(p) DESC LIMIT 50
    WITH collect(DISTINCT [r IN relationships(p) | type(r)]) AS target_chains
    MATCH (v_cand:Video)
    WHERE v_cand.id <> $test_video_id
      AND (v_cand.is_test IS NULL OR v_cand.is_test = false)
      AND (v_cand.video_label IS NOT NULL OR v_cand.predicted_label IS NOT NULL)
    MATCH (v_cand)-[:HAS_SCENE]->(csStart:Scene)
    MATCH q = (csStart)-[
        :SIMILAR_TO|TEMPORAL|ELABORATION|CONTRAST|SPAN|ROOT|JOINT|CAUSE|
        TOPIC_COMMENT|EXPLANATION|EVALUATION|BACKGROUND|TOPIC_CHANGE|
        ATTRIBUTION|TEXTUAL_ORGANIZATION|COMPARISON|SUMMARY|SAME_UNIT|
        CONDITION|ENABLEMENT|MANNER_MEANS*2..4
    ]->(csEnd:Scene)
    WHERE (v_cand)-[:HAS_SCENE]->(csEnd)
      AND all(s IN nodes(q) WHERE (v_cand)-[:HAS_SCENE]->(s))
    WITH v_cand, q, [r IN relationships(q) | type(r)] AS cand_chain, target_chains
    WHERE cand_chain IN target_chains
    RETURN
        v_cand.id AS video_id,
        coalesce(v_cand.video_label, v_cand.predicted_label) AS label,
        max(length(q) + 1) AS max_nodes_matched,
        count(distinct q) AS total_matched_sequences,
        collect({
            length: length(q) + 1,
            sequence_ids: [s IN nodes(q) | s.scene_id],
            relation_chain: cand_chain,
            scene_captions: [s IN nodes(q) | coalesce(s.caption, "No caption")],
            scene_concepts: [s IN nodes(q) | head([
                (s)-[:BELONGS_TO]->(c:Concept) | {
                    concept_id: c.id,
                    label_distribution: c.label_distribution
                }
            ])]
        }) AS match_details
    ORDER BY max_nodes_matched DESC, total_matched_sequences DESC
    LIMIT $limit
    """
    structural = list(tx.run(structural_query, test_video_id=test_video_id, limit=limit))
    result["structural_matches"] = structural

    # ====== Similarity neighbors (grouped by video) ======
    sim_query = """
    MATCH (v_test:Video {id: $test_video_id, is_test: true})-[:HAS_SCENE]->(test_scene:Scene)
    MATCH (test_scene)-[r:SIMILAR_TO]->(db_scene:Scene)<-[:HAS_SCENE]-(db_video:Video)
    WHERE db_video.is_test IS NULL OR db_video.is_test = false
      AND db_video.id <> $test_video_id
    WITH db_video, avg(r.cosine_score) AS avg_score, collect(db_scene)[0..3] AS sample_scenes
    WITH db_video, avg_score, sample_scenes, sample_scenes[0] AS first_scene
    OPTIONAL MATCH (first_scene)-[:BELONGS_TO]->(c:Concept)
    RETURN db_video.id AS video_id,
           db_video.video_label AS label,
           avg_score,
           sample_scenes[0].caption AS sample_caption,
           sample_scenes[0].uid AS sample_uid,
           c.id AS concept_id
    ORDER BY avg_score DESC
    LIMIT $limit
    """
    similarity_videos = list(tx.run(sim_query, test_video_id=test_video_id, limit=limit))
    result["similarity_videos"] = similarity_videos

    # ====== Concept priors ======
    concept_query = """
    MATCH (v:Video {id: $test_video_id})-[:HAS_SCENE]->(s:Scene)-[:BELONGS_TO]->(c:Concept)
    WITH c, count(s) AS scene_count
    ORDER BY scene_count DESC
    LIMIT 5
    OPTIONAL MATCH (c)<-[:BELONGS_TO]-(otherScene:Scene)<-[:HAS_SCENE]-(otherVideo:Video)
    WHERE otherVideo.id <> $test_video_id
      AND (otherVideo.is_test IS NULL OR otherVideo.is_test = false)
      AND (otherVideo.video_label IS NOT NULL OR otherVideo.predicted_label IS NOT NULL)
    WITH c, scene_count, otherVideo, count(distinct otherScene) AS overlap_scenes
    RETURN
        c.id AS concept_id,
        c.label_distribution AS label_distribution,
        scene_count,
        otherVideo.id AS video_id,
        coalesce(otherVideo.video_label, otherVideo.predicted_label) AS label,
        overlap_scenes
    ORDER BY scene_count DESC, overlap_scenes DESC
    LIMIT 20
    """
    concept_priors = list(tx.run(concept_query, test_video_id=test_video_id))
    result["concept_priors"] = concept_priors

    # ====== Video details ======
    video_ids = set()
    for rec in structural:
        video_ids.add(rec.get("video_id"))
    for rec in similarity_videos:
        video_ids.add(rec.get("video_id"))
    video_details = {}
    if video_ids:
        vid_query = """
        MATCH (v:Video)
        WHERE v.id IN $video_ids
        RETURN v.id AS id,
               v.rst_summary AS rst_summary,
               v.dominant_rst AS dominant_rst,
               v.num_scenes AS num_scenes,
               v.num_rst_relations AS num_rst_relations,
               v.relation_statistics AS relation_statistics,
               v.video_label AS label
        """
        result_vid = tx.run(vid_query, video_ids=list(video_ids))
        for rec in result_vid:
            video_details[rec["id"]] = dict(rec)
    result["video_details"] = video_details

    # ====== Concept details ======
    concept_ids_set = set()
    for rec in structural:
        for det in rec.get("match_details", []):
            for sc in det.get("scene_concepts", []):
                if sc and sc.get("concept_id"):
                    concept_ids_set.add(sc["concept_id"])
    for rec in similarity_videos:
        cid = rec.get("concept_id")
        if cid:
            concept_ids_set.add(cid)
    for rec in concept_priors:
        concept_ids_set.add(rec.get("concept_id"))

    result["concept_ids_set"] = concept_ids_set

    concept_details = {}
    if concept_ids_set:
        con_query = """
        MATCH (c:Concept)
        WHERE c.id IN $concept_ids
        RETURN c.id AS id,
               c.summary AS summary,
               c.visual_style AS visual_style,
               c.audio_style AS audio_style,
               c.storyline AS storyline,
               c.keywords AS keywords,
               c.label_distribution AS label_distribution
        """
        result_con = tx.run(con_query, concept_ids=list(concept_ids_set))
        for rec in result_con:
            concept_details[rec["id"]] = dict(rec)
    result["concept_details"] = concept_details

    return result


def dictify_neo4j_records(records: list) -> list:
    """Chuyển list các neo4j.Record thành list[dict] thuần Python để JSON-serializable."""
    return [dict(r) for r in records]


def build_graph_hits_record(graph_evidence: dict) -> dict:
    """Rút gọn graph_evidence (đã dictify) thành 1 record JSON-serializable, gọn nhẹ để lưu cache."""
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


def format_concept_distribution(label_distribution_json) -> str:
    if not label_distribution_json:
        return "No distribution data"
    try:
        dist = json.loads(label_distribution_json) if isinstance(label_distribution_json, str) else label_distribution_json
        return ", ".join(f"L{k}={v}" for k, v in sorted(dist.items(), key=lambda x: int(x[0])))
    except Exception:
        return str(label_distribution_json)


def get_dominant_label_and_count(label_distribution_json):
    """
    Trả về (dominant_label, dominant_count, total) từ chuỗi JSON label distribution.
    Ví dụ: '{"0": 1, "1": 3}' -> (1, 3, 4)
    """
    try:
        if isinstance(label_distribution_json, str):
            dist = json.loads(label_distribution_json)
        else:
            dist = label_distribution_json
        if not dist:
            return None, 0, 0
        dominant_label = max(dist, key=lambda k: dist[k])
        dominant_count = dist[dominant_label]
        total = sum(dist.values())
        return int(dominant_label), dominant_count, total
    except Exception:
        return None, 0, 0


def build_graph_text_from_hits(graph_hits_record: dict) -> tuple:
    """
    Build graph_text (+ lean/confidence) từ graph_hits_record đã cache — KHÔNG cần truy vấn Neo4j.
    Dùng ở giai đoạn inference (GPU) để không phải mở lại kết nối Neo4j.
    """
    lines = []
    label_counts = Counter()

    structural = graph_hits_record.get("structural_matches", []) if graph_hits_record else []
    similarity = graph_hits_record.get("similarity_videos", []) if graph_hits_record else []
    concept_details = graph_hits_record.get("concept_details", {}) if graph_hits_record else {}

    # ----- RST Structural Matches -----
    if structural:
        lines.append("** RST Structural Matches** (exact chain matches):")
        for idx, rec in enumerate(structural[:5], 1):
            label = rec.get("label")
            if label is not None:
                label_counts[label] += 1
            max_nodes = rec.get("max_nodes_matched", 0)
            total_paths = rec.get("total_matched_sequences", 0)
            rst_chain = rec.get("rst_chain", [])
            rst_chain_str = explain_rst_chain(rst_chain) if rst_chain else "N/A"
            rst_summary = rec.get("video_rst_summary", "")

            lines.append(f"  [{idx}] Match Video {idx} | Label: {label}  | Max nodes: {max_nodes}, Total paths: {total_paths}")
            lines.append(f"      RST Chain: [{rst_chain_str}]")
            if rst_summary:
                lines.append(f"      Video RST Summary: {rst_summary[:300]}...")
            lines.append("")
        lines.append("")
    else:
        lines.append("** RST Structural Matches**: None found.")
        lines.append("")

    # ----- Similarity-based Neighbors -----
    if similarity:
        lines.append("** Similarity-based Neighbors** (videos with similar scenes):")
        for idx, rec in enumerate(similarity[:5], 1):
            label = rec.get("label")
            avg_score = rec.get("avg_score")
            if label is not None:
                label_counts[label] += 1
            avg_score_str = f"{avg_score:.3f}" if isinstance(avg_score, (int, float)) else "N/A"
            lines.append(f"  [{idx}] Video Neighbor '{idx}' (Label: {label}) — Avg similarity {avg_score_str}")
            lines.append("")
        lines.append("")
    else:
        lines.append("** Similarity-based Neighbors**: None found.")
        lines.append("")

    # ----- Concept Details (sắp xếp theo số lượng nhãn khác nhau) -----
    if concept_details:
        concept_list = list(concept_details.keys())

        def sort_key(cid):
            cdetail = concept_details.get(cid, {})
            label_dist = cdetail.get("label_distribution", "{}")
            try:
                dist = json.loads(label_dist) if isinstance(label_dist, str) else label_dist
                return len(dist)
            except Exception:
                return 0
        concept_list.sort(key=sort_key, reverse=True)

        lines.append("** Concept Details** (from matched scenes):")
        for cid in concept_list[:5]:
            cdetail = concept_details.get(cid, {})
            if not cdetail:
                continue
            label_dist = cdetail.get("label_distribution", "N/A")
            audio_style = cdetail.get("audio_style", "")
            visual_style = cdetail.get("visual_style", "")
            keywords = cdetail.get("keywords", [])

            lines.append(f"  [Concept] {cid}")
            dom_label, dom_count, total = get_dominant_label_and_count(label_dist)
            if dom_label is not None:
                lines.append(f"      Dominant label: {dom_label}")
            else:
                lines.append(f"      Domain Label: N/A")
            if audio_style:
                lines.append(f"      Audio Style: {audio_style[:150]}...")
            if visual_style:
                lines.append(f"      Visual Style: {visual_style[:150]}...")
            if keywords:
                lines.append(f"      Keywords: {', '.join(keywords[:10])}")
            lines.append("")
        lines.append("")
    else:
        lines.append("** Concept Details**: No concept information available from matched scenes.")
        lines.append("")

    # ----- Tổng hợp label distribution -----
    n_total = sum(label_counts.values())
    if n_total == 0:
        return "\n".join(lines), None, "none"
    n_pos = label_counts.get(1, 0)
    lean, confidence = evidence_lean_and_confidence(n_pos, n_total)
    label_summary = ", ".join(f"Label {lbl}: {cnt}" for lbl, cnt in sorted(label_counts.items()))
    lines.append(f"** Overall label distribution across all graph evidence: {label_summary}  [agreement: {confidence}]")

    return "\n".join(lines), lean, confidence


def compute_graph_evidence_for_video(neo4j_driver, neo4j_database: str, folder: str, sample_data: dict, data_root: Path, graph_top_k: int) -> dict:
    """Chạy toàn bộ chu trình Neo4j (reconstruct -> similarity -> get evidence -> cleanup) cho 1 video và trả về graph_hits_record."""
    try:
        with neo4j_driver.session(database=neo4j_database) as session:
            session.execute_write(reconstruct_multimodal_subgraph_with_logs, folder, sample_data, data_root)
            session.execute_write(create_similarity_relationships_for_test, folder, top_k=3, threshold=0.85)

        with neo4j_driver.session(database=neo4j_database) as session:
            raw_graph_evidence = session.execute_read(get_graph_evidence, folder, limit=graph_top_k)

        graph_evidence = {
            "structural_matches": dictify_neo4j_records(raw_graph_evidence.get("structural_matches", [])),
            "similarity_videos":  dictify_neo4j_records(raw_graph_evidence.get("similarity_videos", [])),
            "concept_priors":     dictify_neo4j_records(raw_graph_evidence.get("concept_priors", [])),
            "video_details":      raw_graph_evidence.get("video_details", {}),
            "concept_details":    raw_graph_evidence.get("concept_details", {}),
            "concept_ids_set":    raw_graph_evidence.get("concept_ids_set", set()),
        }
        return build_graph_hits_record(graph_evidence)
    finally:
        with neo4j_driver.session(database=neo4j_database) as session:
            session.execute_write(cleanup_test_subgraph, folder)


def init_neo4j_driver(args: argparse.Namespace):
    neo4j_uri      = args.neo4j_uri      or os.getenv("NEO4J_URI")
    neo4j_username = args.neo4j_username or os.getenv("NEO4J_USERNAME")
    neo4j_password = args.neo4j_password or os.getenv("NEO4J_PASSWORD")
    neo4j_database = args.neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")
    if not all([neo4j_uri, neo4j_username, neo4j_password]):
        raise ValueError("Missing NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD (env or --neo4j_* args).")
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
    driver.verify_connectivity()
    return driver, neo4j_database


# ==========================================
# 8. PROMPT BUILDERS & ENSEMBLE
# ==========================================

def build_reasoning_example(evidence_mode: str) -> str:
    bullets = []
    if evidence_mode in ("milvus", "full"):
        bullets.append("- Content similarity: label distribution across 5 distinct reference videos is 4×Label 1, 1×Label 0. [agreement: strong]")
        bullets.append("- Discourse pattern: label distribution across 5 distinct reference videos is 3×Label 0, 2×Label 1. [agreement: weak]")
        bullets.append("- The two sources disagree; content shows 'strong' agreement while discourse pattern shows only 'weak' agreement.")
    if evidence_mode in ("graph", "full"):
        bullets.append(
            "- Knowledge Graph: RST structural matches show 5 videos, all Label 1, with chain ROOT -> TEMPORAL -> TEMPORAL. "
            "Concept Details show a dominant Concept with label distribution {'1': 3}, suggesting high engagement. "
            "Similarity neighbors are also mostly Label 1."
        )
    if not bullets:
        bullets.append("- No reference library is used in this setting — judged purely from the video's own content and structure.")

    return f"""---\n[Example]:\n{"\n".join(bullets)}\n\n[Expected output]:
{{
  "predicted_label": "1",
  "explanation": "Scene 2 delivers a clear, specific hook which acts as the narrative root, and references with similar pattern had strong agreement.",
  "improvement_suggestions": ["Add a brief reaction shot right after the reveal.", "Vary the pacing slightly earlier."]
}}\n---"""


def build_reasoning_guidelines(evidence_mode: str) -> str:
    primary_filter = (
        "1. **BALANCED CONTENT EVALUATION**: Carefully analyze the input video's structural progression "
        "and logical flow based on the text. Evaluate whether the scenes are organized with a clear focus, "
        "as defined in the Core Distinction Criteria."
    )
    if evidence_mode == "none":
        return f"{primary_filter}\n2. **FINAL DECISION**: Synthesize objective structural analysis."

    result = f"{primary_filter}\n2. **REFERENCE CROSS-EXAMINATION**: Examine provided references as anchors.\n3. **INTEGRATED JUDGMENT**: Give strong agreement reference systems significant weight."

    if evidence_mode in ("graph", "full"):
        graph_guidance = (
            "\n4. **GRAPH EVIDENCE INTERPRETATION**: When Knowledge Graph evidence is available:\n"
            "   - **Concept Details**: First, inspect the label distribution and semantic details (Audio/Visual Style, Keywords) "
            "of Concepts found in the matched RST paths. A strong label distribution (e.g., >70% Label 1) that aligns "
            "with the input video's content is a powerful signal.\n"
            "   - **RST Structural Matches**: Evaluate how many matched videos share the same RST chain and their labels. "
            "A majority of Label 1 among matched videos supports High Engagement.\n"
            "   - **Video RST Summary**: Use this to understand the overall discourse structure of the matched videos. "
            "Compare it with your own analysis of the input video's structure (Section 1) to see if they align.\n"
            "   - **Similarity-based Neighbors**: Consider videos with high average similarity. If they are mostly Label 1, "
            "it reinforces the prediction; if mixed, treat with caution.\n"
            "Use strong consensus across multiple sources as high-confidence evidence."
        )
        result += graph_guidance

    return result


def build_llm_prompt(video_context_text, content_similarity_text, narrative_pattern_text, evidence_mode, graph_text=None) -> str:
    sections = [f"=== 1. INPUT VIDEO CONTENT ===\n{video_context_text}"]
    n = 2
    if content_similarity_text is not None:
        sections.append(f"=== {n}. CONTENT SIMILARITY REFERENCE ===\n{content_similarity_text}")
        n += 1
    if narrative_pattern_text is not None:
        sections.append(f"=== {n}. DISCOURSE PATTERN REFERENCE ===\n{narrative_pattern_text}")
        n += 1
    if graph_text is not None:
        sections.append(f"=== {n}. REFERENCE — KNOWLEDGE GRAPH (STRUCTURAL / RST-CHAIN MATCH) ===\n{graph_text}")
        n += 1

    sections.append(f"=== {n}. REASONING EXAMPLE ===\n{build_reasoning_example(evidence_mode)}")
    sections.append(f"=== REASONING GUIDELINES ===\n{build_reasoning_guidelines(evidence_mode)}")
    sections.append(f"""Respond ONLY with a JSON object, no text outside the braces:
{{
  "predicted_label": "{_valid_labels_str}",
  "explanation": "A plain-language review. Mention specific scenes. No system names or scores.",
  "improvement_suggestions": ["Actionable suggestion 1.", "Actionable suggestion 2."]
}}""")
    return "\n\n".join(sections)


def compute_ensemble_label(llm_pred, content_lean, content_conf, narrative_lean, narrative_conf):
    """Ensemble gốc cho mode milvus (giữ nguyên, không đổi)."""
    votes = [v for v in [llm_pred, content_lean, narrative_lean] if v in (0, 1)]
    if not votes:
        return llm_pred if llm_pred in (0, 1) else -1

    count = Counter(votes)
    top = count.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        strong_candidates = []
        if content_lean in (0, 1) and content_conf == "strong": strong_candidates.append(content_lean)
        if narrative_lean in (0, 1) and narrative_conf == "strong": strong_candidates.append(narrative_lean)
        if strong_candidates: return strong_candidates[0]
        if llm_pred in (0, 1): return llm_pred
        for candidate in (content_lean, narrative_lean):
            if candidate in (0, 1): return candidate
    return top[0][0]


def compute_ensemble_label_two_signal(llm_pred, ref_lean, ref_conf):
    """Ensemble cho mode graph (2 tín hiệu: LLM + graph)."""
    votes = [v for v in [llm_pred, ref_lean] if v in (0, 1)]
    if not votes:
        return llm_pred if llm_pred in (0, 1) else -1
    count = Counter(votes)
    top = count.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        if ref_lean in (0, 1) and ref_conf == "strong":
            return ref_lean
        return llm_pred if llm_pred in (0, 1) else ref_lean
    return top[0][0]


def compute_ensemble_label_full(llm_pred, content_lean, content_conf, narrative_lean, narrative_conf, graph_lean, graph_conf):
    """Ensemble cho mode full (4 tín hiệu: LLM + content + narrative + graph)."""
    votes = [v for v in [llm_pred, content_lean, narrative_lean, graph_lean] if v in (0, 1)]
    if not votes:
        return llm_pred if llm_pred in (0, 1) else -1
    count = Counter(votes)
    top = count.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        strong_candidates = []
        if content_lean in (0, 1) and content_conf == "strong": strong_candidates.append(content_lean)
        if narrative_lean in (0, 1) and narrative_conf == "strong": strong_candidates.append(narrative_lean)
        if graph_lean in (0, 1) and graph_conf == "strong": strong_candidates.append(graph_lean)
        if strong_candidates: return strong_candidates[0]
        if llm_pred in (0, 1): return llm_pred
        for candidate in (content_lean, narrative_lean, graph_lean):
            if candidate in (0, 1): return candidate
    return top[0][0]


def compute_final_prediction(evidence_mode, pred_label, content_lean, content_conf, narrative_lean, narrative_conf, graph_lean, graph_conf):
    if evidence_mode == "full":
        return compute_ensemble_label_full(pred_label, content_lean, content_conf, narrative_lean, narrative_conf, graph_lean, graph_conf)
    elif evidence_mode == "milvus":
        return compute_ensemble_label(pred_label, content_lean, content_conf, narrative_lean, narrative_conf)
    elif evidence_mode == "graph":
        return compute_ensemble_label_two_signal(pred_label, graph_lean, graph_conf)
    else:
        return pred_label if pred_label in (0, 1) else -1


# ==========================================
# 9. INFRASTRUCTURE & LOADERS
# ==========================================

def generate_input_video_context(folder_name: str, data: dict, data_root: Path, mode="milvus") -> str:
    seg_path = data_root / folder_name / "segments.json"
    scene_ids_list = data['scene_ids']
    captions_dict = {}

    try:
        with open(seg_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        if isinstance(segments, list):
            for idx, seg in enumerate(segments):
                if idx >= len(scene_ids_list): break
                captions_dict[int(scene_ids_list[idx])] = seg.get('caption', '')
        elif isinstance(segments, dict):
            for k, v in segments.items():
                captions_dict[int(k)] = v.get('caption', str(v)) if isinstance(v, dict) else str(v)
    except Exception:
        pass

    raw_caption_list = [captions_dict.get(int(sid), "No caption available.") for sid in scene_ids_list]

    # none & graph -> hiển thị caption thô (giống stage1/2 gốc cho graph);
    # milvus & full -> dùng DPS cho input video (giữ nguyên hành vi milvus cũ).
    if mode in ("none", "graph"):
        return "\n".join([f"  Scene {sid}: \"{c[:130]}\"" for sid, c in zip(scene_ids_list, raw_caption_list)])
    else:
        return serialize_discourse_captions(scene_ids_list, data.get("rst_links", []), raw_caption_list)


def load_video_representations(video_reps_dir: Path) -> dict:
    pt_path = video_reps_dir / "video_representations.pt"
    if not pt_path.exists(): raise FileNotFoundError(f"{pt_path} not found.")
    return torch.load(pt_path, map_location="cpu")


def load_valid_videos(data_root: Path, split_file: Path, reps_by_folder: dict, require_reps: bool) -> tuple:
    with open(split_file, 'r') as f: test_folders = json.load(f).get("test", [])
    valid_folders, valid_data, invalid_folders = [], {}, []

    for f_name in test_folders:
        if not (data_root / f_name / "scene_embeddings.pt").exists() or not (data_root / f_name / "segments.json").exists():
            invalid_folders.append((f_name, "Missing files"))
            continue
        if require_reps and f_name not in reps_by_folder:
            invalid_folders.append((f_name, "Missing global video representations"))
            continue
        try:
            d = torch.load(data_root / f_name / "scene_embeddings.pt", map_location='cpu')
            valid_folders.append(f_name)
            valid_data[f_name] = d
        except Exception as e:
            invalid_folders.append((f_name, str(e)))

    return valid_folders, valid_data, invalid_folders


def load_retrieval_cache(path) -> dict:
    if path and Path(path).exists():
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)
    return {}


def save_retrieval_cache(cache: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f: json.dump(cache, f, ensure_ascii=False, indent=2)


# ==========================================
# 10. PRECOMPUTE RETRIEVAL (CPU only)
#     -> nạp KẾT QUẢ MILVUS + GRAPH vào CHUNG 1 file indexing/retrieval_cache.json
# ==========================================

def run_precompute_retrieval(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if not args.retrieval_cache_path: raise ValueError("--retrieval_cache_path is required.")

    data_root, split_file, cache_path = Path(args.data_root), Path(args.split_file), Path(args.retrieval_cache_path)
    need_milvus = args.evidence_mode in ("milvus", "full")
    need_graph  = args.evidence_mode in ("graph", "full")

    if not need_milvus and not need_graph:
        print("[INFO] evidence_mode='none' does not require any precompute retrieval. Nothing to do.")
        return

    milvus_client, c_name, reps_by_folder = None, None, {}
    if need_milvus:
        milvus_client = MilvusClient(uri=os.getenv("MILVUS_CLUSTER_ENDPOINT"), token=os.getenv("MILVUS_TOKEN"))
        c_name = args.collection_name or os.getenv("MILVUS_COLLECTION_NAME")
        reps_by_folder = load_video_representations(Path(args.video_reps_dir))

    neo4j_driver, neo4j_database = None, None
    if need_graph:
        neo4j_driver, neo4j_database = init_neo4j_driver(args)

    valid_folders, valid_data, _ = load_valid_videos(data_root, split_file, reps_by_folder, require_reps=need_milvus)

    cache = load_retrieval_cache(cache_path)
    # Một video được coi là "đã xong" nếu đã có đủ các thành phần mà evidence_mode hiện tại cần.
    def is_done(entry):
        if entry is None or entry.get("error"):
            return False
        if need_milvus and "content_similarity_text" not in entry:
            return False
        if need_graph and "graph_hits_record" not in entry:
            return False
        return True

    queue = [f for f in valid_folders if not is_done(cache.get(f))]
    print(f"[PRECOMPUTE] {len(queue)} / {len(valid_folders)} videos left (evidence_mode={args.evidence_mode}).")
    video_cache = {}

    for i, folder in enumerate(queue, 1):
        print(f"[PRECOMPUTE][{i}/{len(queue)}] {folder}", end=" ", flush=True)
        sample_data = valid_data[folder]
        entry = cache.get(folder, {}) or {}
        try:
            if need_milvus:
                res = milvus_client.search(
                    collection_name=c_name, data=[reps_by_folder[folder].tolist()], limit=args.content_search_limit, output_fields=DISCOURSE_OUTPUT_FIELDS
                )
                top_hits = dedupe_content_hits_by_video(res, args.content_top_k)
                c_text, c_lean, c_conf = build_content_similarity_context(top_hits)
                entry.update({
                    "content_similarity_text": c_text,
                    "content_lean": c_lean,
                    "content_confidence": c_conf,
                    "content_hits_record": top_hits
                })

                embs_norm = F.normalize(sample_data['embeddings'].float(), p=2, dim=1)
                ranked_top = retrieve_discourse_evidence(
                    embs_norm, sample_data.get('rst_links', []), folder, milvus_client, c_name, data_root,
                    args.discourse_top_k, args.discourse_search_limit, args.alpha, video_cache
                )
                n_text, n_lean, n_conf = build_discourse_context(ranked_top)
                hits_record = []
                for vid, cand in ranked_top:
                    hits_record.append({
                        "score": cand["score"],
                        "dense_sim": cand["dense_sim"],
                        "topology_sim": cand["topology_sim"],
                        "video_label": cand["video_label"],
                        "matched_scene_id": cand["matched_scene_id"],
                        "matched_caption": cand["matched_caption"],
                        "linked_scene_id": cand.get("linked_scene_id"),
                        "linked_caption": cand.get("linked_caption"),
                        "relation": cand.get("relation")
                    })
                entry.update({
                    "narrative_pattern_text": n_text,
                    "narrative_lean": n_lean,
                    "narrative_confidence": n_conf,
                    "narrative_hits_record": hits_record
                })

            if need_graph:
                graph_hits_record = compute_graph_evidence_for_video(
                    neo4j_driver, neo4j_database, folder, sample_data, data_root, args.graph_top_k
                )
                g_text, g_lean, g_conf = build_graph_text_from_hits(graph_hits_record)
                entry.update({
                    "graph_hits_record": graph_hits_record,
                    "graph_lean": g_lean,
                    "graph_confidence": g_conf,
                })

            entry.pop("error", None)
            cache[folder] = entry
            print("| OK")
        except Exception as e:
            print(f"| ERROR: {e}")
            entry["error"] = str(e)
            cache[folder] = entry

        if i % 10 == 0 or i == len(queue): save_retrieval_cache(cache, cache_path)

    save_retrieval_cache(cache, cache_path)

    if neo4j_driver is not None:
        neo4j_driver.close()

    print(f"\n[DONE] Precompute finished. {len(cache)} videos cached at {cache_path}")


# ==========================================
# 11. INFERENCE CORE
# ==========================================

def count_tokens_qwen(tokenizer, messages: list) -> int:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tokenizer.encode(text))

def run_qwen_inference(tokenizer, model, messages: list, model_type: str = "instruct") -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    gen_kwargs = {"max_new_tokens": 1024, "do_sample": True, "temperature": 0.6, "top_p": 0.95} if model_type == "thinking" else {"max_new_tokens": 1024, "do_sample": False, "temperature": 0.0}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    raw_output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    if model_type == "thinking":
        try:
            idx = len(outputs[0]) - outputs[0].tolist()[::-1].index(151668)
            return tokenizer.decode(outputs[0][idx:], skip_special_tokens=True).strip()
        except ValueError:
            return raw_output
    return raw_output


def extract_and_parse_json(raw_text: str) -> dict:
    if not raw_text: return {"predicted_label": -1, "explanation": "Empty output"}
    clean = re.sub(r'^```json\s*|\s*```$', '', raw_text.strip(), flags=re.IGNORECASE)
    try:
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        parsed = json.loads(match.group(0) if match else clean)
        parsed["predicted_label"] = int(str(parsed.get("predicted_label")).strip())
        return parsed
    except Exception:
        return {"predicted_label": -1, "explanation": "Failed to parse json"}


def main(args: argparse.Namespace) -> None:
    data_root, split_file, checkpoint_path = Path(args.data_root), Path(args.split_file), Path(args.checkpoint_path)
    need_milvus = args.evidence_mode in ("milvus", "full")
    need_graph  = args.evidence_mode in ("graph", "full")

    milvus_client, c_name, reps_by_folder = None, None, {}
    if need_milvus:
        reps_by_folder = load_video_representations(Path(args.video_reps_dir))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto").eval()

    valid_folders, valid_data, _ = load_valid_videos(data_root, split_file, reps_by_folder, require_reps=need_milvus)
    evaluation_results = []
    processed_folders = set()

    if checkpoint_path.exists():
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            evaluation_results = json.load(f)
        processed_folders = {r['folder_name'] for r in evaluation_results if r.get('prediction') != -1}

    queue = [f for f in valid_folders if f not in processed_folders]
    retrieval_cache = load_retrieval_cache(args.retrieval_cache_path)
    video_cache = {}

    # Kết nối Milvus/Neo4j "lười" — chỉ khởi tạo khi thực sự cần fallback (video thiếu trong cache).
    def get_milvus_client():
        nonlocal milvus_client, c_name
        if milvus_client is None:
            milvus_client = MilvusClient(uri=os.getenv("MILVUS_CLUSTER_ENDPOINT"), token=os.getenv("MILVUS_TOKEN"))
            c_name = args.collection_name or os.getenv("MILVUS_COLLECTION_NAME")
        return milvus_client, c_name

    neo4j_state = {"driver": None, "database": None}
    def get_neo4j_driver():
        if neo4j_state["driver"] is None:
            neo4j_state["driver"], neo4j_state["database"] = init_neo4j_driver(args)
        return neo4j_state["driver"], neo4j_state["database"]

    for current_idx, folder in enumerate(queue, 1):
        print(f"[{current_idx}/{len(queue)}] {folder}", end=" ", flush=True)
        sample_data = valid_data[folder]

        try:
            with open(data_root / folder / "segments.json", 'r', encoding='utf-8') as f:
                segments = json.load(f)
            captions = load_captions_by_index(segments, sample_data['scene_ids'])

            video_context_text = generate_input_video_context(folder, sample_data, data_root, args.evidence_mode)
            cached = retrieval_cache.get(folder)

            # ---------- (A) MILVUS: content similarity + discourse pattern ----------
            content_similarity_text, content_hits_record, content_lean, content_conf = None, [], None, None
            narrative_pattern_text, narrative_hits_record, narrative_lean, narrative_conf = None, [], None, None
            if need_milvus:
                if cached and "content_similarity_text" in cached:
                    content_similarity_text = cached["content_similarity_text"]
                    content_lean = cached["content_lean"]
                    content_conf = cached["content_confidence"]
                    content_hits_record = cached["content_hits_record"]
                else:
                    mc, cn = get_milvus_client()
                    res = mc.search(collection_name=cn, data=[reps_by_folder[folder].tolist()], limit=args.content_search_limit, output_fields=DISCOURSE_OUTPUT_FIELDS)
                    top_hits = dedupe_content_hits_by_video(res, args.content_top_k)
                    content_similarity_text, content_lean, content_conf = build_content_similarity_context(top_hits)
                    content_hits_record = top_hits

                if cached and "narrative_pattern_text" in cached:
                    narrative_pattern_text = cached["narrative_pattern_text"]
                    narrative_lean = cached["narrative_lean"]
                    narrative_conf = cached["narrative_confidence"]
                    narrative_hits_record = cached["narrative_hits_record"]
                else:
                    mc, cn = get_milvus_client()
                    embs_norm = F.normalize(sample_data['embeddings'].float(), p=2, dim=1)
                    ranked_top = retrieve_discourse_evidence(
                        embs_norm, sample_data.get('rst_links', []), folder, mc, cn, data_root,
                        args.discourse_top_k, args.discourse_search_limit, args.alpha, video_cache
                    )
                    narrative_pattern_text, narrative_lean, narrative_conf = build_discourse_context(ranked_top)
                    narrative_hits_record = []
                    for vid, cand in ranked_top:
                        narrative_hits_record.append({
                            "score": cand["score"],
                            "dense_sim": cand["dense_sim"],
                            "topology_sim": cand["topology_sim"],
                            "video_label": cand["video_label"],
                            "matched_scene_id": cand["matched_scene_id"],
                            "matched_caption": cand["matched_caption"],
                            "linked_scene_id": cand.get("linked_scene_id"),
                            "linked_caption": cand.get("linked_caption"),
                            "relation": cand.get("relation")
                        })

            # ---------- (B) NEO4J: knowledge graph structural evidence ----------
            graph_text, graph_hits_record, graph_lean, graph_conf = None, {}, None, None
            if need_graph:
                if cached and "graph_hits_record" in cached:
                    graph_hits_record = cached["graph_hits_record"]
                    graph_lean = cached.get("graph_lean")
                    graph_conf = cached.get("graph_confidence")
                    graph_text, _, _ = build_graph_text_from_hits(graph_hits_record)
                else:
                    driver, db = get_neo4j_driver()
                    graph_hits_record = compute_graph_evidence_for_video(driver, db, folder, sample_data, data_root, args.graph_top_k)
                    graph_text, graph_lean, graph_conf = build_graph_text_from_hits(graph_hits_record)

            llm_prompt = build_llm_prompt(video_context_text, content_similarity_text, narrative_pattern_text, args.evidence_mode, graph_text)
            messages = [{"role": "system", "content": build_system_prompt(args.evidence_mode)}, {"role": "user", "content": llm_prompt}]
            n_tokens = count_tokens_qwen(tokenizer, messages)

            llm_response = run_qwen_inference(tokenizer, llm_model, messages, model_type=args.model_type)
            verdict = extract_and_parse_json(llm_response)
            pred_label = verdict.get("predicted_label", -1)

            final_prediction = compute_final_prediction(
                args.evidence_mode, pred_label, content_lean, content_conf, narrative_lean, narrative_conf, graph_lean, graph_conf
            )
            ground_truth = int(sample_data['y'].item() if isinstance(sample_data['y'], torch.Tensor) else sample_data['y'])

            status = "✓" if ground_truth == pred_label else "✗"
            print(f"[{current_idx}/{len(queue)}] {folder} | GT={ground_truth} llm={pred_label} {status} | tokens={n_tokens:,} | final={final_prediction}")

            evaluation_results.append({
                "folder_name": folder,
                "evidence_mode": args.evidence_mode,
                "ground_truth": ground_truth,
                "prediction": pred_label,
                "content_lean": content_lean,
                "content_confidence": content_conf,
                "narrative_lean": narrative_lean,
                "narrative_confidence": narrative_conf,
                "graph_lean": graph_lean,
                "graph_confidence": graph_conf,
                "final_prediction": final_prediction,
                "explanation": verdict.get("explanation", ""),
                "improvement_suggestions": verdict.get("improvement_suggestions", []),
                "raw_llm_output": llm_response,
                "content_similarity_hits": content_hits_record,
                "narrative_evidence_hits": narrative_hits_record,
                "graph_evidence_hits": graph_hits_record,
            })
        except Exception as e:
            print(f"| ERROR: {e}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        with open(checkpoint_path, 'w', encoding='utf-8') as f: json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

    if neo4j_state["driver"] is not None:
        neo4j_state["driver"].close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--evidence_mode", type=str, default="milvus", choices=EVIDENCE_MODE_CHOICES)

    # --- milvus ---
    parser.add_argument("--collection_name", type=str, default=None)
    parser.add_argument("--video_reps_dir", type=str, default=None)
    parser.add_argument("--content_top_k", type=int, default=5)
    parser.add_argument("--content_search_limit", type=int, default=50)
    parser.add_argument("--discourse_top_k", type=int, default=5)
    parser.add_argument("--discourse_search_limit", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.7)

    # --- graph (neo4j) ---
    parser.add_argument("--graph_top_k", type=int, default=5)
    parser.add_argument("--neo4j_uri", type=str, default=None, help="Defaults to env NEO4J_URI.")
    parser.add_argument("--neo4j_username", type=str, default=None, help="Defaults to env NEO4J_USERNAME.")
    parser.add_argument("--neo4j_password", type=str, default=None, help="Defaults to env NEO4J_PASSWORD.")
    parser.add_argument("--neo4j_database", type=str, default=None, help="Defaults to env NEO4J_DATABASE (or 'neo4j').")

    # --- model / inference ---
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model_type", type=str, default="instruct", choices=["instruct", "thinking"])

    # --- precompute (CPU) / cache ---
    parser.add_argument("--precompute_retrieval", action="store_true",
                         help="Chỉ chạy truy vấn Milvus/Neo4j (CPU) và lưu vào --retrieval_cache_path, không load LLM.")
    parser.add_argument("--retrieval_cache_path", type=str, default=None,
                         help="Đường dẫn indexing/retrieval_cache.json — dùng chung cho cả milvus và graph.")

    args = parser.parse_args()
    if args.precompute_retrieval: run_precompute_retrieval(args)
    else: main(args)