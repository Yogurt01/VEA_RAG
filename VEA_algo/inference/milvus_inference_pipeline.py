"""
milvus_inference_pipeline.py
---------------------------------
VideoRAG Inference Pipeline — Content Library + Dense Edge Retrieval (Milvus)
+ Qwen3-4B-Instruct, với hỗ trợ ABLATION STUDY qua flag --evidence_mode.

Bốn chế độ (dùng chung 1 pipeline, cùng 1 đường code, chỉ khác việc có gọi
2 khối retrieval hay không — đảm bảo so sánh ablation "táo với táo"):

    --evidence_mode none     : chỉ dùng nội dung video test (không retrieval).
    --evidence_mode content  : + Content Similarity (Cross-Modal Self-Querying, Milvus).
    --evidence_mode edge     : + Dense Edge Retrieval + Case-based Subgraph Evidence (Milvus).
    --evidence_mode full     : cả 2 nguồn (mặc định, tương đương pipeline đầy đủ).

Usage — ví dụ chạy đủ 4 ablation:
    python milvus_inference_pipeline.py --evidence_mode none  --checkpoint_path .../ablation_none.json  ...
    python milvus_inference_pipeline.py --evidence_mode content --video_reps_dir .../video_representations --checkpoint_path .../ablation_content.json ...
    python milvus_inference_pipeline.py --evidence_mode edge    --edge_index_dir .../edge_index --checkpoint_path .../ablation_edge.json ...
    python milvus_inference_pipeline.py --evidence_mode full    --video_reps_dir .../video_representations --edge_index_dir .../edge_index --checkpoint_path .../ablation_full.json ...

    Đủ tham số (mode = full):
        python milvus_inference_pipeline.py \
            --data_root         /path/to/All_Videos \
            --split_file        /path/to/dataset_splits.json \
            --checkpoint_path   /path/to/results_full.json \
            --video_reps_dir    /path/to/video_representations \
            --edge_index_dir    /path/to/edge_index \
            --evidence_mode     full \
            [--model_name       Qwen/Qwen3-4B-Instruct-2507] \
            [--top_k_evidence   5] \
            [--candidate_limit  50]

Environment variables (KHÔNG qua args):
    MILVUS_CLUSTER_ENDPOINT, MILVUS_TOKEN
    MILVUS_COLLECTION_NAME        (chỉ cần khi evidence_mode ∈ {content, full})
    MILVUS_EDGE_COLLECTION_NAME   (chỉ cần khi evidence_mode ∈ {edge, full}; mặc định "rst_edge_index")

Ghi chú:
    - Đã bỏ chế độ --count_tokens_only (dry-run đếm token) theo yêu cầu.
    - Mỗi record trong checkpoint JSON giờ lưu thêm: evidence_mode,
      input_scene_captions, content_similarity_hits, narrative_evidence_hits
      — đủ để audit lại evidence đã dùng mà không cần chạy lại.
    - Với evidence_mode='none', KHÔNG cần kết nối Milvus / --video_reps_dir /
      --edge_index_dir — pipeline chạy độc lập, nhẹ hơn.
"""

import os
import re
import gc
import time
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from pymilvus import MilvusClient
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MAX_CONTEXT_TOKENS  = 32768
SIM_WEIGHT          = 0.85   # trọng số cosine similarity trong final score edge
PRIOR_WEIGHT        = 0.15   # trọng số prior theo rst_type
EVIDENCE_MODE_CHOICES = ["none", "content", "edge", "full"]


# ==========================================
# 1. RST NORMALIZATION
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


def rst_to_natural(rst_type: str) -> str:
    return RST_DESCRIPTIONS.get(rst_type.upper(), rst_type.lower().replace('_', ' '))


# ==========================================
# 2. LABEL DEFINITIONS
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
# 3. SYSTEM PROMPT — xây dựng ĐỘNG theo evidence_mode (cho ablation)
# ==========================================

REFERENCE_SOURCE_DOCS = {
    "content": """1. CONTENT SIMILARITY REFERENCE:
   - These are the most visually and thematically similar video segments found in the library.
   - The similarity score reflects overall content alignment — the higher the score, the more
     similar the video is in terms of topic, visual style, and scene content.
   - Pay attention to the label distribution (how many are Label 0 vs Label 1) as a prior
     about whether this type of content tends to engage viewers.""",
    "edge": """2. NARRATIVE TRANSITION REFERENCE:
   - These results come from matching individual scene-to-scene transitions of the input video
     against a large library of transitions from other videos, using content similarity.
   - Each match shows a reference video that had a similar transition (e.g., one type of scene
     followed by another type of scene, connected the same way) and its known engagement outcome.
   - The "Match strength" score (0–1) indicates how similar the matched transition is.
   - When the top matches consistently point to one label, that's a strong signal.
   - "Narrative relation diversity" measures how many distinct discourse relations
     (e.g., contrast, elaboration, temporal) a video's scene transitions use.
     Low diversity (e.g., almost all transitions use the same relation type) often indicates
     a repetitive narrative structure, which is more common among Low Engagement videos.""",
}


def build_system_prompt(evidence_mode: str) -> str:
    active_docs = []
    if evidence_mode in ("content", "full"):
        active_docs.append(REFERENCE_SOURCE_DOCS["content"])
    if evidence_mode in ("edge", "full"):
        active_docs.append(REFERENCE_SOURCE_DOCS["edge"])

    if active_docs:
        intro = " cross-reference it with similar contexts retrieved from a reference library,"
        reference_block = "How to use the reference source(s) below:\n\n" + "\n\n".join(active_docs)
        reasoning_intro = "- Read the video content first, then use the references as supporting evidence."
    else:
        intro = ""
        reference_block = "No external reference library is used in this setting — base your decision solely on the video content described below."
        reasoning_intro = "- Base your decision solely on the input video's own content and structure."

    return f"""You are an Advanced Video Analysis and Evaluation System.
Your task is to receive the structural, textual, and multimodal information of an input video,{intro}
and provide a final engagement prediction.

Label definitions:
{_label_def_lines}
IMPORTANT: Only two labels exist — 0 and 1.

{reference_block}

Reasoning instructions:
{reasoning_intro}
- Do not default to High Engagement when uncertain — Low Engagement is equally valid.
- Be specific: reference scene numbers and explain why the narrative works or doesn't.
- Write your explanation in plain language — no system names, no technical scores.
"""


# ==========================================
# 4. CONTEXT BUILDER — CONTENT SIMILARITY
# ==========================================

def build_content_similarity_context(top_hits: list) -> str:
    """Build context block for content similarity — neutral naming."""
    if not top_hits:
        return "No similar content found in the reference library."

    label_counts = Counter()
    for hit in top_hits:
        lbl = hit.get('video_label')
        if lbl is not None:
            label_counts[lbl] += 1

    label_summary = ", ".join(
        f"Label {lbl}: {cnt}" for lbl, cnt in sorted(label_counts.items())
    )

    lines = [
        f"Label distribution across top matches: {label_summary}",
        "",
    ]
    for idx, hit in enumerate(top_hits, 1):
        s_uid = hit.get('scene_uid', '')
        scene_num = s_uid.split('_')[-1] if '_' in s_uid else "?"
        lines.append(
            f"Match {idx} (Similarity: {hit['score']:.4f}) — "
            f"Reference label: {hit['video_label']}\n"
            f"  Scene {scene_num}: {hit['caption']}"
        )

    return "\n".join(lines)


# ==========================================
# 5. DENSE EDGE RETRIEVAL — Milvus (thay ConceptRAG/Neo4j)
# ==========================================

EDGE_OUTPUT_FIELDS = [
    "video_id", "video_label", "rst_type", "src_caption", "tgt_caption",
    "src_scene_id", "tgt_scene_id", "depth_src", "depth_tgt",
]


def load_prior_scores(edge_index_dir: Path) -> dict:
    prior_path = edge_index_dir / "prior_scores.json"
    if not prior_path.exists():
        print(f"[WARN] {prior_path} not found, using flat 0.5 prior for all rst_types.")
        return {}
    with open(prior_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_video_representations(video_reps_dir: Path) -> dict:
    """Đọc video_representations.pt (từ milvus_compute_video_query.py) -> {folder_name: tensor(2048,)}."""
    pt_path = video_reps_dir / "video_representations.pt"
    if not pt_path.exists():
        raise FileNotFoundError(
            f"{pt_path} not found. Run milvus_compute_video_query.py first."
        )
    return torch.load(pt_path, map_location="cpu")


def load_captions_by_index(segments, scene_ids_list, max_len: int = 990):
    """Trả về list caption theo đúng thứ tự index (0-based), khớp embeddings[i]."""
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


def build_query_edges(embeddings_norm: torch.Tensor, rst_links: list, captions: list) -> list:
    """Encode mỗi RST edge của video (test) thành query vector 4096-dim."""
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
        rst_norm = normalize_rst_type(raw_rst)
        vec = torch.cat([embeddings_norm[s_idx], embeddings_norm[t_idx]], dim=0).tolist()
        queries.append({
            "vector":      vec,
            "rst_type":    rst_norm,
            "src_caption": captions[s_idx] if s_idx < len(captions) else "",
            "tgt_caption": captions[t_idx] if t_idx < len(captions) else "",
        })
    return queries


def search_edge(milvus_client, edge_collection_name: str, query_vec: list, rst_type: str, limit: int):
    """Tier 1: filter đúng rst_type. Tier 2: nếu rỗng, search không filter (luôn có kết quả)."""
    try:
        hits = milvus_client.search(
            collection_name=edge_collection_name,
            data=[query_vec],
            filter=f'rst_type == "{rst_type}"',
            limit=limit,
            output_fields=EDGE_OUTPUT_FIELDS,
        )[0]
    except Exception:
        hits = []

    if not hits:
        try:
            hits = milvus_client.search(
                collection_name=edge_collection_name,
                data=[query_vec],
                limit=limit,
                output_fields=EDGE_OUTPUT_FIELDS,
            )[0]
        except Exception:
            hits = []
    return hits


def score_edge_hit(hit: dict, prior_scores: dict) -> float:
    cos_sim  = float(hit["distance"])  # metric COSINE -> similarity trực tiếp
    rst_type = hit["entity"]["rst_type"]
    prior    = prior_scores.get(rst_type, 0.5)
    return SIM_WEIGHT * cos_sim + PRIOR_WEIGHT * prior


def get_local_subgraph_context(video_id: str, src_scene_id: int, tgt_scene_id: int,
                                 data_root: Path, max_neighbors: int = 2) -> str:
    """Mở rộng 1-hop trong video training gốc để evidence là 1 đoạn subgraph nhỏ."""
    try:
        emb_path = data_root / video_id / "scene_embeddings.pt"
        seg_path = data_root / video_id / "segments.json"
        pt_data  = torch.load(emb_path, map_location="cpu")
        scene_ids_list = [int(x) for x in pt_data["scene_ids"]]
        rst_links = pt_data.get("rst_links", [])
        with open(seg_path, 'r', encoding='utf-8') as f:
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
                a, b = int(link[0]) - 1, int(link[1]) - 1
            except Exception:
                continue
            if a in (s_idx, t_idx) and b not in (s_idx, t_idx):
                neighbor_idxs.add(b)
            if b in (s_idx, t_idx) and a not in (s_idx, t_idx):
                neighbor_idxs.add(a)

        neighbor_idxs = list(neighbor_idxs)[:max_neighbors]
        if not neighbor_idxs:
            return ""
        lines = [f"     Nearby context: {captions[i][:150]}" for i in neighbor_idxs if i < len(captions)]
        return "\n".join(lines)
    except Exception:
        return ""


def retrieve_narrative_evidence(
    embeddings_norm: torch.Tensor, rst_links: list, captions: list,
    current_video_id: str, milvus_client, edge_collection_name: str,
    prior_scores: dict, data_root: Path, top_k: int, candidate_limit: int,
) -> tuple:
    """
    Trả về (query_edges, ranked_top) trong đó ranked_top là list
    [(video_id, candidate_dict), ...] đã dedupe theo video và sort giảm dần theo score.
    """
    query_edges = build_query_edges(embeddings_norm, rst_links, captions)
    if not query_edges:
        return query_edges, []

    best_per_video = {}
    for qe in query_edges:
        hits = search_edge(milvus_client, edge_collection_name, qe["vector"], qe["rst_type"], candidate_limit)
        for h in hits:
            vid = h["entity"]["video_id"]
            if vid == current_video_id:
                continue
            score = score_edge_hit(h, prior_scores)
            if vid not in best_per_video or score > best_per_video[vid]["score"]:
                best_per_video[vid] = {
                    "score":        score,
                    "video_label":  h["entity"]["video_label"],
                    "rst_type":     h["entity"]["rst_type"],
                    "src_caption":  h["entity"]["src_caption"],
                    "tgt_caption":  h["entity"]["tgt_caption"],
                    "src_scene_id": h["entity"]["src_scene_id"],
                    "tgt_scene_id": h["entity"]["tgt_scene_id"],
                }

    ranked = sorted(best_per_video.items(), key=lambda kv: kv[1]["score"], reverse=True)
    return query_edges, ranked[:top_k]


# ==========================================
# 6. CONTEXT BUILDER — NARRATIVE PATTERNS (dense retrieval, không còn concept)
# ==========================================

def build_narrative_pattern_context(
    query_edges: list, ranked_top: list, data_root: Path,
) -> str:
    """Build context block cho evidence dạng case-based subgraph — neutral naming."""
    lines = []

    if query_edges:
        distinct = len(set(qe['rst_type'] for qe in query_edges))
        total    = len(query_edges)
        diversity = distinct / total if total else 0.0
        div_note  = " (LOW — may indicate a repetitive narrative structure)" if diversity < 0.3 else ""
        lines.append(
            f"Narrative relation diversity: {distinct}/{total} distinct relation types used "
            f"({diversity:.0%}){div_note}."
        )
    else:
        lines.append("Narrative relation diversity: no RST relations extracted.")
    lines.append("")

    if query_edges:
        parts = []
        for qe in query_edges[:10]:
            src_short = qe['src_caption'][:40] + '...' if len(qe['src_caption']) > 40 else qe['src_caption']
            tgt_short = qe['tgt_caption'][:40] + '...' if len(qe['tgt_caption']) > 40 else qe['tgt_caption']
            parts.append(f'"{src_short}" → ({rst_to_natural(qe["rst_type"])}) → "{tgt_short}"')
        lines.append("Input video transition sequence:")
        lines.append("  " + "  |  ".join(parts))
    else:
        lines.append("Input video transition sequence: none extracted.")
    lines.append("")

    if not ranked_top:
        lines.append("No matching transition patterns found in the reference library.")
        lines.append("These narrative patterns appear to be novel relative to the training data.")
        return "\n".join(lines)

    lines.append(f"Top {len(ranked_top)} matching transition patterns from reference library:")
    lines.append("")

    for rank, (vid, cand) in enumerate(ranked_top, 1):
        outcome = "HIGH ENGAGEMENT" if cand['video_label'] == 1 else "LOW ENGAGEMENT"
        lines.append(f"[{rank}] Reference pattern ({rst_to_natural(cand['rst_type'])}):")
        lines.append(f"     Outcome: {outcome}  |  Match strength: {cand['score']:.4f}")
        lines.append(f'     "{cand["src_caption"][:150]}"')
        lines.append(f'     "{cand["tgt_caption"][:150]}"')
        ctx = get_local_subgraph_context(vid, cand['src_scene_id'], cand['tgt_scene_id'], data_root)
        if ctx:
            lines.append(ctx)
        lines.append("")

    n_eng  = sum(1 for _, c in ranked_top if c['video_label'] == 1)
    n_neng = len(ranked_top) - n_eng
    if n_eng > n_neng and n_eng >= 2:
        lines.append(f"Note: {n_eng}/{len(ranked_top)} reference patterns are from High Engagement videos.")
    elif n_neng > n_eng and n_neng >= 2:
        lines.append(f"Note: {n_neng}/{len(ranked_top)} reference patterns are from Low Engagement videos.")
    else:
        lines.append("Note: reference patterns are mixed — no clear lean.")

    return "\n".join(lines)


# ==========================================
# 7. USER PROMPT — dựng ĐỘNG theo evidence_mode
# ==========================================

def build_reasoning_example(evidence_mode: str) -> str:
    bullets = []
    if evidence_mode in ("content", "full"):
        bullets.append("- Content similarity: 4/5 reference videos are Label 1 (High Engagement).")
    if evidence_mode in ("edge", "full"):
        bullets.append(
            '- Narrative transitions: 3/5 reference patterns are from Low Engagement videos.\n'
            '  Top pattern: "Product reveal close-up" → (contrasts with) → "Casual low-energy reaction" '
            '| LOW ENGAGEMENT | Match strength: 0.81'
        )
        bullets.append("- Narrative relation diversity: 2/9 transitions (22%) — LOW.")
    if not bullets:
        bullets.append("- No reference library is used in this setting — judged purely from the video's own content and structure.")

    example_bullets = "\n".join(bullets)
    return f"""---
[Example]:
{example_bullets}

[Expected output]:
{{
  "predicted_label": "0",
  "explanation": "The video opens with a product reveal but then shifts to a casual, low-energy reaction that doesn't build engagement. By Scene 4, the viewer has already lost interest because there is no clear payoff or escalation.",
  "improvement_suggestions": [
    "After the product reveal in Scene 2, introduce a more energetic or surprising reaction to maintain momentum.",
    "Add more varied transitions across scenes to build a sense of progression and keep viewer interest."
  ]
}}
---"""


def build_reasoning_guidelines(evidence_mode: str) -> str:
    if evidence_mode == "none":
        return """1. Read the video content carefully. Does it have a clear hook, build-up, and satisfying arc?
2. Base your decision solely on the input video's own content and structure — no external reference library is available in this setting.
3. Be specific: mention scene numbers and explain what works or doesn't.
4. Write your explanation in plain language. No system names, no technical jargon."""
    return """1. Read the video content first. Does it have a clear hook, build-up, and satisfying arc?
2. Use the references as supporting evidence — not as the primary basis.
3. When references are mixed, rely more on the video content itself.
4. Be specific: mention scene numbers and explain what works or doesn't.
5. Write your explanation in plain language. No system names, no technical jargon."""


def build_llm_prompt(
    video_context_text: str,
    content_similarity_text: str,
    narrative_pattern_text: str,
    evidence_mode: str,
) -> str:
    sections = [f"=== 1. INPUT VIDEO CONTENT ===\n{video_context_text}"]
    n = 2
    if content_similarity_text is not None:
        sections.append(f"=== {n}. CONTENT SIMILARITY REFERENCE ===\n{content_similarity_text}")
        n += 1
    if narrative_pattern_text is not None:
        sections.append(f"=== {n}. NARRATIVE TRANSITION REFERENCE ===\n{narrative_pattern_text}")
        n += 1

    sections.append(f"=== {n}. REASONING EXAMPLE ===\n{build_reasoning_example(evidence_mode)}")
    sections.append(f"=== REASONING GUIDELINES ===\n{build_reasoning_guidelines(evidence_mode)}")
    sections.append(f"""Respond ONLY with a JSON object, no text outside the braces:
{{
  "predicted_label": "{_valid_labels_str}",
  "explanation": "A plain-language review. Mention specific scenes. No system names or scores.",
  "improvement_suggestions": [
    "Actionable suggestion 1.",
    "Actionable suggestion 2."
  ]
}}""")

    return "\n\n".join(sections)


# ==========================================
# 8. CONSTANTS & VALIDATION
# ==========================================

REQUIRED_DATA_KEYS = [
    'embeddings', 'scene_ids', 'metadata',
    'edge_index', 'edge_attr', 'rst_links', 'y',
]
MILVUS_OUTPUT_FIELDS = ["scene_uid", "video_id", "video_label", "caption"]


# ==========================================
# 9. HELPER FUNCTIONS
# ==========================================

def generate_input_video_context(folder_name: str, data: dict, data_root: Path) -> str:
    seg_path       = data_root / folder_name / "segments.json"
    scene_ids_list = data['scene_ids']
    captions_dict  = {}

    try:
        with open(seg_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        if isinstance(segments, list):
            for idx, seg in enumerate(segments):
                if idx >= len(scene_ids_list):
                    break
                captions_dict[int(scene_ids_list[idx])] = seg.get('caption', '')
        elif isinstance(segments, dict):
            for k, v in segments.items():
                captions_dict[int(k)] = (
                    v.get('caption', str(v)) if isinstance(v, dict) else str(v)
                )
    except Exception:
        pass

    lines = [f"Total scenes: {len(scene_ids_list)}", ""]
    for scene_id in scene_ids_list[:20]:
        s_id_int = int(scene_id)
        cap = captions_dict.get(s_id_int, "No caption available.")
        cap = cap[:130] + '...' if len(cap) > 130 else cap
        lines.append(f'  Scene {s_id_int}: "{cap}"')

    return "\n".join(lines)


def add_hits_to_pool(search_results, aggregated_hits: dict) -> None:
    for hits in search_results:
        for hit in hits:
            entity = hit['entity']
            uid    = entity.get('scene_uid') or hit.get('id')
            score  = hit['distance']
            if uid not in aggregated_hits or score > aggregated_hits[uid]['score']:
                aggregated_hits[uid] = {
                    'score':       score,
                    'scene_uid':   uid,
                    'video_id':    entity.get('video_id', 'N/A'),
                    'video_label': entity.get('video_label', 'N/A'),
                    'caption':     entity.get('caption', ''),
                }


def load_valid_videos(
    data_root: Path, split_file: Path, reps_by_folder: dict, require_reps: bool,
) -> tuple:
    with open(split_file, 'r') as f:
        test_folders = json.load(f).get("test", [])

    valid_folders, valid_data, invalid_folders = [], {}, []

    for folder_name in test_folders:
        emb_path = data_root / folder_name / "scene_embeddings.pt"
        seg_path = data_root / folder_name / "segments.json"

        if not emb_path.exists():
            invalid_folders.append((folder_name, "Missing scene_embeddings.pt"))
            continue
        if not seg_path.exists():
            invalid_folders.append((folder_name, "Missing segments.json"))
            continue
        if require_reps and folder_name not in reps_by_folder:
            invalid_folders.append((folder_name, "Missing video representation"))
            continue

        try:
            data = torch.load(emb_path, map_location='cpu')
            missing = [k for k in REQUIRED_DATA_KEYS if k not in data]
            if missing:
                invalid_folders.append((folder_name, f"Missing keys: {missing}"))
                continue
            valid_folders.append(folder_name)
            valid_data[folder_name] = data
        except Exception as e:
            invalid_folders.append((folder_name, f"Load error: {e}"))

    return valid_folders, valid_data, invalid_folders


# ==========================================
# 10. QWEN INFERENCE HELPERS
# ==========================================

def count_tokens_qwen(tokenizer, messages: list) -> int:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return len(tokenizer.encode(text))


def run_qwen_inference(tokenizer, model, messages: list, max_new_tokens: int = 1024) -> str:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )

    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    return response


def extract_and_parse_json(raw_text: str) -> dict:
    valid_labels = set(LABEL_DEFINITIONS.keys())

    if not raw_text:
        return {"predicted_label": -1, "explanation": "Empty LLM response"}

    clean = re.sub(r'^```json\s*', '', raw_text.strip(), flags=re.IGNORECASE)
    clean = re.sub(r'^```\s*', '', clean)
    clean = re.sub(r'\s*```$', '', clean).strip()

    try:
        match  = re.search(r'\{.*\}', clean, re.DOTALL)
        parsed = json.loads(match.group(0) if match else clean)
        pred   = parsed.get("predicted_label")
        if pred is not None:
            pred_int = int(str(pred).strip())
            if pred_int not in valid_labels:
                pred_int = min(valid_labels, key=lambda x: abs(x - pred_int))
            parsed["predicted_label"] = pred_int
            return parsed
    except Exception:
        pass

    broad = re.search(
        r'(predicted[-_ ]label|label|prediction)\s*["\']?\s*[:=]\s*["\']?\s*(\d+)',
        clean, re.IGNORECASE,
    )
    if broad:
        pred_int = int(broad.group(2))
        if pred_int not in valid_labels:
            pred_int = min(valid_labels, key=lambda x: abs(x - pred_int))
        return {"predicted_label": pred_int, "explanation": "Broad Regex"}

    return {"predicted_label": -1, "explanation": "Parsing failed"}


# ==========================================
# 11. MAIN
# ==========================================

def main(args: argparse.Namespace) -> None:

    data_root       = Path(args.data_root)
    split_file      = Path(args.split_file)
    checkpoint_path = Path(args.checkpoint_path)
    evidence_mode   = args.evidence_mode

    need_content = evidence_mode in ("content", "full")
    need_edge    = evidence_mode in ("edge", "full")

    milvus_client         = None
    collection_name       = None
    edge_collection_name  = None
    reps_by_folder        = {}
    prior_scores          = {}

    # ── Kết nối Milvus + load artifacts CHỈ khi mode cần ─────────────
    if need_content or need_edge:
        milvus_endpoint = os.getenv("MILVUS_CLUSTER_ENDPOINT")
        milvus_token    = os.getenv("MILVUS_TOKEN")
        if not all([milvus_endpoint, milvus_token]):
            raise ValueError("Missing MILVUS_CLUSTER_ENDPOINT / MILVUS_TOKEN env vars.")
        milvus_client = MilvusClient(uri=milvus_endpoint, token=milvus_token)

    if need_content:
        collection_name = os.getenv("MILVUS_COLLECTION_NAME")
        if not collection_name:
            raise ValueError("Missing MILVUS_COLLECTION_NAME env var (required for evidence_mode='content'/'full').")
        if not args.video_reps_dir:
            raise ValueError("--video_reps_dir is required when evidence_mode is 'content' or 'full'.")
        reps_by_folder = load_video_representations(Path(args.video_reps_dir))
        print(f"[INFO] Loaded {len(reps_by_folder)} video representations from {args.video_reps_dir}.")

    if need_edge:
        edge_collection_name = args.edge_collection_name or os.getenv("MILVUS_EDGE_COLLECTION_NAME", "rst_edge_index")
        if not milvus_client.has_collection(edge_collection_name):
            raise RuntimeError(
                f"Collection '{edge_collection_name}' not found. Run milvus_build_edge_index.py first."
            )
        if not args.edge_index_dir:
            raise ValueError("--edge_index_dir is required when evidence_mode is 'edge' or 'full'.")
        prior_scores = load_prior_scores(Path(args.edge_index_dir))
        print(f"[INFO] Loaded {len(prior_scores)} rst_type priors from {args.edge_index_dir}.")

    print(f"[INFO] evidence_mode = '{evidence_mode}' "
          f"(content={'ON' if need_content else 'off'}, edge={'ON' if need_edge else 'off'})\n")

    print(f"[INFO] Loading Qwen3-4B-Instruct from: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    llm_model.eval()
    print("[INFO] Model loaded.\n")

    valid_folders, valid_data, invalid_folders = load_valid_videos(
        data_root, split_file, reps_by_folder, require_reps=need_content,
    )
    print(f"[INFO] Test split: {len(valid_folders)} valid, {len(invalid_folders)} invalid.")

    # ── MAIN INFERENCE LOOP ──────────────────────────────────────────
    evaluation_results = []
    processed_folders  = set()

    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                evaluation_results = json.load(f)
            processed_folders = {r['folder_name'] for r in evaluation_results}
            print(f"[INFO] Resuming: {len(processed_folders)} already processed.")
        except Exception as e:
            print(f"[WARNING] Corrupted checkpoint, starting fresh: {e}")

    queue = [f for f in valid_folders if f not in processed_folders]
    print(f"[START] {len(queue)} videos to process.\n")

    for current_idx, folder in enumerate(queue, 1):
        print(f"[{current_idx}/{len(queue)}] {folder}", end=" ", flush=True)

        sample_data = valid_data[folder]

        try:
            with open(data_root / folder / "segments.json", 'r', encoding='utf-8') as f:
                segments = json.load(f)
            captions = load_captions_by_index(segments, sample_data['scene_ids'])

            # STEP A: Video context
            video_context_text = generate_input_video_context(folder, sample_data, data_root)

            # STEP B: Content similarity retrieval (Milvus, collection scene-level)
            content_similarity_text = None
            content_hits_record     = []
            if need_content:
                rep_vec = reps_by_folder.get(folder)
                if rep_vec is None:
                    raise ValueError("Missing video representation")
                res = milvus_client.search(
                    collection_name=collection_name,
                    data=[rep_vec.tolist()],
                    limit=5,
                    output_fields=MILVUS_OUTPUT_FIELDS,
                )
                aggregated_hits = {}
                add_hits_to_pool(res, aggregated_hits)
                top_5_hits = sorted(
                    aggregated_hits.values(), key=lambda x: x['score'], reverse=True
                )[:5]
                content_similarity_text = build_content_similarity_context(top_5_hits)
                content_hits_record = top_5_hits

            # STEP C: Narrative pattern retrieval — Dense Edge Retrieval (Milvus)
            narrative_pattern_text = None
            narrative_hits_record  = []
            if need_edge:
                embeddings_norm = F.normalize(sample_data['embeddings'].float(), p=2, dim=1)
                query_edges, ranked_top = retrieve_narrative_evidence(
                    embeddings_norm, sample_data.get('rst_links', []), captions,
                    current_video_id=folder,
                    milvus_client=milvus_client, edge_collection_name=edge_collection_name,
                    prior_scores=prior_scores, data_root=data_root,
                    top_k=args.top_k_evidence, candidate_limit=args.candidate_limit,
                )
                narrative_pattern_text = build_narrative_pattern_context(
                    query_edges, ranked_top, data_root,
                )
                narrative_hits_record = [{"video_id": vid, **cand} for vid, cand in ranked_top]

            # STEP D: Build prompt + token check
            llm_prompt = build_llm_prompt(
                video_context_text, content_similarity_text, narrative_pattern_text, evidence_mode,
            )
            system_prompt = build_system_prompt(evidence_mode)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": llm_prompt},
            ]
            n_tokens = count_tokens_qwen(tokenizer, messages)

            if n_tokens > MAX_CONTEXT_TOKENS:
                if narrative_pattern_text is not None:
                    messages[1]["content"] = build_llm_prompt(
                        video_context_text, content_similarity_text,
                        "Narrative pattern context truncated — prompt exceeded context window.",
                        evidence_mode,
                    )
                elif content_similarity_text is not None:
                    messages[1]["content"] = build_llm_prompt(
                        video_context_text,
                        "Content similarity context truncated — prompt exceeded context window.",
                        narrative_pattern_text, evidence_mode,
                    )
                n_tokens = count_tokens_qwen(tokenizer, messages)

            # STEP E: Qwen inference
            llm_response = run_qwen_inference(tokenizer, llm_model, messages)
            verdict      = extract_and_parse_json(llm_response)
            pred_label   = verdict.get("predicted_label", -1)

            ground_truth = int(
                sample_data['y'].item()
                if isinstance(sample_data['y'], torch.Tensor)
                else sample_data['y']
            )

            status = "✓" if ground_truth == pred_label else "✗"
            preview_parts = []
            if content_hits_record:
                t1 = content_hits_record[0]
                preview_parts.append(f"content_top1(L{t1['video_label']},{t1['score']:.2f})")
            if narrative_hits_record:
                t1 = narrative_hits_record[0]
                preview_parts.append(f"edge_top1(L{t1['video_label']},{t1['score']:.2f},{t1['rst_type']})")
            preview = (" | " + " | ".join(preview_parts)) if preview_parts else ""
            print(f"| GT={ground_truth} Pred={pred_label} {status} | tokens={n_tokens:,}{preview}")

            evaluation_results.append({
                "folder_name":              folder,
                "evidence_mode":            evidence_mode,
                "ground_truth":             ground_truth,
                "prediction":               pred_label,
                "token_count":              n_tokens,
                "explanation":              verdict.get("explanation", ""),
                "improvement_suggestions":  verdict.get("improvement_suggestions", []),
                "raw_llm_output":           llm_response,
                "input_scene_captions":     captions[:20],
                "content_similarity_hits":  content_hits_record,
                "narrative_evidence_hits":  narrative_hits_record,
            })

        except Exception as e:
            print(f"| ERROR: {e}")
            evaluation_results.append({
                "folder_name":              folder,
                "evidence_mode":            evidence_mode,
                "ground_truth":             None,
                "prediction":               -1,
                "token_count":              0,
                "explanation":              f"Pipeline error: {e}",
                "improvement_suggestions":  [],
                "raw_llm_output":           "",
                "input_scene_captions":     [],
                "content_similarity_hits":  [],
                "narrative_evidence_hits":  [],
            })

        finally:
            if 'inputs' in locals(): del inputs
            if 'outputs' in locals(): del outputs
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(0.4)

        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] {len(evaluation_results)} videos. Results: {checkpoint_path}")


# ==========================================
# 12. ENTRY POINT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VideoRAG Inference Pipeline — content + dense edge retrieval (Milvus) + Qwen, with ablation flag.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_root",         type=str, required=True)
    parser.add_argument("--split_file",        type=str, required=True)
    parser.add_argument("--checkpoint_path",   type=str, required=True)
    parser.add_argument("--evidence_mode",     type=str, default="full", choices=EVIDENCE_MODE_CHOICES,
                        help="Ablation mode: none | content | edge | full.")
    parser.add_argument("--video_reps_dir",    type=str, default=None,
                        help="Dir with video_representations.pt (from milvus_compute_video_query.py). "
                             "Required when evidence_mode ∈ {content, full}.")
    parser.add_argument("--edge_index_dir",    type=str, default=None,
                        help="Dir with prior_scores.json (from milvus_build_edge_index.py). "
                             "Required when evidence_mode ∈ {edge, full}.")
    parser.add_argument("--edge_collection_name", type=str, default=None,
                        help="Milvus collection cho RST edges. Mặc định env "
                             "MILVUS_EDGE_COLLECTION_NAME hoặc 'rst_edge_index'.")
    parser.add_argument("--model_name",        type=str, default=DEFAULT_MODEL_NAME,
                        help=f"Qwen model ID. Default: {DEFAULT_MODEL_NAME}")
    parser.add_argument("--top_k_evidence",    type=int, default=5,
                        help="Top-K unique reference videos in narrative pattern context.")
    parser.add_argument("--candidate_limit",   type=int, default=50,
                        help="Max candidates lấy từ Milvus MỖI query edge (ANN search).")
    args = parser.parse_args()
    main(args)
