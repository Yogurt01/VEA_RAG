"""
milvus_inference_pipeline.py
--------------------------------------------------------------------------------------------------
Usage: 
    python milvus_inference_pipeline.py --evidence_mode edge \
        --data_root .../All_Videos --split_file .../dataset_splits.json \
        --checkpoint_path .../ablation_edge.json \
        --content_collection_name video_scenes_collection \
        --alpha 0.7 --discourse_top_k 5 --discourse_search_limit 30 \
        --model_name .../Qwen3-4B-Instruct-2507
"""

import os
import re
import gc
import time
import json
import argparse
import collections
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

# Thứ tự CỐ ĐỊNH — dùng làm chiều cho topology vector (Nuclearity Scoring),
# đảm bảo mọi video (test lẫn training) đều so sánh trên đúng cùng 1 hệ trục.
# CANONICAL_RST_TYPES = tuple(sorted(RST_DESCRIPTIONS.keys()))
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
# 3. SYSTEM PROMPT
# ==========================================

REFERENCE_SOURCE_DOCS = {
    "content": """1. CONTENT SIMILARITY REFERENCE (PRIMARY SOURCE):
   - These are the most visually and thematically similar REFERENCE VIDEOS found in the
     library (deduplicated — each is a distinct video, not repeated scenes from the same video).
   - This serves as your primary external baseline for evaluation, as it reflects solid thematic alignment.
   - Each block reports its own internal agreement (strong / weak / mixed). Treat "weak" or "mixed" as low-confidence.""",
    "edge": """2. DISCOURSE PATTERN REFERENCE (SECONDARY/SUPPLEMENTARY SOURCE):
   - These results come from matching individual scenes of the input video against a library
     of scenes using BOTH content similarity AND discourse structure similarity.
   - This block is purely supplementary and should be used as secondary reference context 
     rather than an equal decision-making signal to content similarity.""",
}

CONFLICT_RESOLUTION_NOTE = (
    "\nIn full mode, you must give significantly more weight and priority to the CONTENT SIMILARITY REFERENCE "
    "over the DISCOURSE PATTERN REFERENCE. Content similarity provides a well-balanced benchmark of overall thematic "
    "alignment, whereas the discourse reference should only act as a secondary, auxiliary source to assist in "
    "calibrating borderline cases or adding context."
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
    if evidence_mode in ("content", "full"):
        active_docs.append(REFERENCE_SOURCE_DOCS["content"])
    if evidence_mode in ("edge", "full"):
        active_docs.append(REFERENCE_SOURCE_DOCS["edge"])

    if active_docs:
        intro = " cross-reference it with similar contexts retrieved from a reference library,"
        reference_block = "How to use the reference source(s) below:\n\n" + "\n\n".join(active_docs)
        if evidence_mode == "full":
            reference_block += "\n" + CONFLICT_RESOLUTION_NOTE
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

{CONTENT_PATTERN_NOTES}

{reference_block}

Reasoning instructions:
{reasoning_intro}
{CALIBRATION_NOTE}
- Be specific: reference scene numbers and explain why the narrative works or doesn't.
- Write your explanation in plain language — no system names, no technical scores.
"""

# ==========================================
# 4. EVIDENCE LEAN + CONFIDENCE (dùng chung cho cả 2 kênh)
# ==========================================

def evidence_lean_and_confidence(n_pos: int, n_total: int):
    """
    Trả về (lean, confidence_label).
    lean: 1 nếu đa số Engaging, 0 nếu đa số Not Engaging, None nếu hòa.
    confidence_label: "strong" nếu chênh lệch > 1, "weak" nếu chênh lệch == 1,
                       "mixed" nếu hòa tuyệt đối, "none" nếu không có evidence.
    """
    if n_total == 0:
        return None, "none"
    n_neg = n_total - n_pos
    diff = abs(n_pos - n_neg)
    if n_pos == n_neg:
        return None, "mixed"
    lean = 1 if n_pos > n_neg else 0
    confidence = "weak" if diff <= 1 else "strong"
    return lean, confidence


# ==========================================
# 5. CONTEXT BUILDER — CONTENT SIMILARITY (dedupe theo video)
# ==========================================

def dedupe_content_hits_by_video(search_results, top_k: int) -> list:
    """
    Search trả về hit theo SCENE — dedupe theo video_id (giữ hit điểm cao nhất
    mỗi video) trước khi lấy top_k, để mỗi video chỉ đóng góp ĐÚNG 1 phiếu.
    """
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
                    'video_id':    vid,
                    'video_label': entity.get('video_label', 'N/A'),
                    'caption':     entity.get('caption', ''),
                }
    ranked = sorted(best_per_video.values(), key=lambda x: x['score'], reverse=True)
    return ranked[:top_k]


def build_content_similarity_context(top_hits: list) -> tuple:
    """Trả về (context_text, lean, confidence)."""
    if not top_hits:
        return "No similar content found in the reference library.", None, "none"

    label_counts = Counter(h['video_label'] for h in top_hits if h.get('video_label') is not None)
    n_pos = label_counts.get(1, 0)
    n_total = sum(label_counts.values())
    lean, confidence = evidence_lean_and_confidence(n_pos, n_total)

    label_summary = ", ".join(f"Label {lbl}: {cnt}" for lbl, cnt in sorted(label_counts.items()))

    lines = [
        f"Label distribution across {len(top_hits)} distinct reference videos: "
        f"{label_summary}  [agreement: {confidence}]",
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

    return "\n".join(lines), lean, confidence


# ==========================================
# 6. DISCOURSE PATH SERIALIZATION (DPS)
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


def build_discourse_adjacency(rst_links: list, n_scenes: int) -> tuple:
    """
    Xây đồ thị VÔ HƯỚNG {node: [(neighbor, relation), ...]} từ rst_links, và
    xác định root_node (scene gắn với quan hệ ROOT, mặc định scene 0 nếu không có).
    """
    adj = collections.defaultdict(list)
    root_candidates = set()
    for link in rst_links:
        try:
            src, tgt, rst_type = link[0], link[1], link[2]
            s, t = int(src) - 1, int(tgt) - 1
        except Exception:
            continue
        if not (0 <= s < n_scenes and 0 <= t < n_scenes):
            continue
        rst_norm = normalize_rst_type(rst_type)
        adj[s].append((t, rst_norm))
        adj[t].append((s, rst_norm))
        if rst_norm == "ROOT":
            root_candidates.add(s)
            root_candidates.add(t)
    root_node = next(iter(root_candidates)) if root_candidates else 0
    return adj, root_node


def find_path_to_root(adj: dict, start: int, root: int) -> list:
    """
    BFS parent-pointer thuần Python (không dùng networkx, nhất quán với phần
    còn lại của file) — trả về list (from_scene, relation, to_scene) theo
    đúng thứ tự từ `start` đến `root`. Rỗng nếu start==root hoặc không tới được
    (đồ thị rời rạc).
    """
    if start == root:
        return []
    visited = {start}
    parent = {}
    queue = collections.deque([start])
    while queue:
        u = queue.popleft()
        if u == root:
            break
        for v, rel in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                parent[v] = (u, rel)
                queue.append(v)
    if root not in visited and root != start:
        return []
    chain = []
    node = root
    while node != start:
        if node not in parent:
            return []
        prev, rel = parent[node]
        chain.append((prev, rel, node))
        node = prev
    chain.reverse()
    return chain


def serialize_discourse_captions(captions: list, rst_links: list, n_scenes: int) -> list:
    """
    DPS: với mỗi scene, chú thích caption gốc bằng quan hệ diễn ngôn dẫn về
    scene gốc (ROOT) của video — chỉ lấy BƯỚC ĐẦU TIÊN của đường đi để tránh
    caption quá dài với cây sâu, vẫn giữ tinh thần "chú thích ngữ cảnh phân cấp".
    """
    adj, root_node = build_discourse_adjacency(rst_links, n_scenes)
    serialized = []
    for i in range(n_scenes):
        base = captions[i] if i < len(captions) else ""
        chain = find_path_to_root(adj, i, root_node)
        if not chain:
            serialized.append(base)
            continue
        _, rel, to_scene = chain[0]
        rel_natural = rst_to_natural(rel)
        if len(chain) == 1:
            annotation = f" [Discourse: {rel_natural} the video's core scene (Scene {to_scene})]"
        else:
            annotation = f" [Discourse: {rel_natural} Scene {to_scene}, eventually leading to the video's core scene]"
        serialized.append(base + annotation)
    return serialized


def compute_topology_vector(rst_links: list) -> list:
    """Vector tần suất (đã chuẩn hóa L1) theo CANONICAL_RST_TYPES — Nuclearity Scoring."""
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
    """
    Load (và cache trong 1 lần gọi retrieve) rst_links/caption DPS/topology vector
    của 1 video training — dùng cho Chặng 2 (Topology Re-ranking).
    """
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
        dps_captions = serialize_discourse_captions(raw_captions, rst_links, len(scene_ids_list))
        data = {
            "scene_ids":       scene_ids_list,
            "dps_captions":    dps_captions,
            "topology_vector": compute_topology_vector(rst_links),
        }
    except Exception:
        data = None
    cache[video_id] = data
    return data


DISCOURSE_OUTPUT_FIELDS = ["scene_uid", "video_id", "video_label", "caption"]


def retrieve_discourse_evidence(
    embeddings_norm: torch.Tensor, rst_links: list, captions: list,
    current_video_id: str, milvus_client, collection_name: str, data_root: Path,
    top_k: int, search_limit: int, alpha: float,
) -> tuple:
    """
    Chặng 1 (Dense Vector Search): với MỖI SCENE của video test, search trên
    collection scene-level đã có sẵn (DÙNG CHUNG với kênh Content Similarity).
    Chặng 2 (Topology/Nuclearity Re-ranking): blend dense similarity với
    cosine similarity giữa topology vector của video test và video ứng viên.

    Trả về (query_dps_captions, ranked_top) — ranked_top: list[(video_id, cand_dict)]
    đã dedupe theo video, sort giảm dần theo final blended score.
    """
    n_scenes = embeddings_norm.shape[0]
    query_dps_captions = serialize_discourse_captions(captions, rst_links, n_scenes)
    query_topology = compute_topology_vector(rst_links)

    if n_scenes == 0:
        return query_dps_captions, []

    video_cache = {}
    best_per_video = {}

    for scene_idx in range(n_scenes):
        query_vec = embeddings_norm[scene_idx].tolist()
        try:
            res = milvus_client.search(
                collection_name=collection_name,
                data=[query_vec],
                limit=search_limit,
                output_fields=DISCOURSE_OUTPUT_FIELDS,
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
                s_uid = h["entity"].get("scene_uid", "")
                matched_caption = h["entity"].get("caption", "")
                try:
                    matched_scene_id = int(s_uid.rsplit("_", 1)[-1])
                    matched_idx = cand_data["scene_ids"].index(matched_scene_id)
                    matched_caption = cand_data["dps_captions"][matched_idx]
                except Exception:
                    pass

                best_per_video[vid] = {
                    "score":           final_score,
                    "dense_sim":       dense_sim,
                    "topology_sim":    topo_sim,
                    "video_label":     h["entity"]["video_label"],
                    "matched_caption": matched_caption,
                }

    ranked = sorted(best_per_video.items(), key=lambda kv: kv[1]["score"], reverse=True)
    return query_dps_captions, ranked[:top_k]


# ==========================================
# 7. CONTEXT BUILDER — DISCOURSE PATTERNS (không mớm nhãn)
# ==========================================

def build_discourse_context(query_dps_captions: list, ranked_top: list) -> tuple:
    """Trả về (context_text, lean, confidence)."""
    lines = []

    if not ranked_top:
        lines.append("No matching discourse patterns found in the reference library.")
        return "\n".join(lines), None, "none"

    n_eng = sum(1 for _, c in ranked_top if c['video_label'] == 1)
    lean, confidence = evidence_lean_and_confidence(n_eng, len(ranked_top))

    lines.append(f"Top {len(ranked_top)} matching discourse patterns from reference library:")
    lines.append("")
    for rank, (vid, cand) in enumerate(ranked_top, 1):
        outcome = "HIGH ENGAGEMENT" if cand['video_label'] == 1 else "LOW ENGAGEMENT"
        lines.append(
            f"[{rank}] Outcome: {outcome}  |  Blended score: {cand['score']:.4f} "
            f"(dense={cand['dense_sim']:.3f}, topology={cand['topology_sim']:.3f})"
        )
        lines.append(f'     "{cand["matched_caption"][:220]}"')
        lines.append("")

    n_neng = len(ranked_top) - n_eng
    lines.append(
        f"Label distribution across {len(ranked_top)} distinct reference videos: "
        f"Label 1: {n_eng}, Label 0: {n_neng}  [agreement: {confidence}]"
    )

    return "\n".join(lines), lean, confidence


# ==========================================
# 8. USER PROMPT
# ==========================================

def build_reasoning_example(evidence_mode: str) -> str:
    bullets = []
    if evidence_mode in ("content", "full"):
        bullets.append("- Content similarity: label distribution across 5 distinct reference videos is 4×Label 1, 1×Label 0. [agreement: strong]")
    if evidence_mode in ("edge", "full"):
        bullets.append(
            '- Discourse pattern: label distribution across 5 distinct reference videos is 3×Label 0, 2×Label 1. [agreement: weak]\n'
            '  Top match: Blended score 0.81 (dense=0.85, topology=0.70) | LOW ENGAGEMENT\n'
            '  "A close-up product reveal [Discourse: elaborates on Scene 2, eventually leading to the video\'s core scene]"'
        )
    if evidence_mode == "full":
        bullets.append("- The two sources disagree; content shows 'strong' agreement while discourse pattern shows only "
                        "'weak' agreement, so content gets more weight here per the confidence-based rule — but "
                        "the video's own content is still the primary basis for the final call.")
    if not bullets:
        bullets.append("- No reference library is used in this setting — judged purely from the video's own content and structure.")

    example_bullets = "\n".join(bullets)
    return f"""---
[Example]:
{example_bullets}

[Expected output]:
{{
  "predicted_label": "1",
  "explanation": "Scene 2 delivers a clear, specific hook (the reveal itself), and the reference videos with a similar pattern and strong agreement were mostly High Engagement. The transition sequence builds toward that reveal rather than repeating itself.",
  "improvement_suggestions": [
    "Add a brief reaction shot right after the reveal in Scene 2 to extend the payoff.",
    "Vary the pacing slightly earlier in the video to build more anticipation before the reveal."
  ]
}}
---"""


def build_reasoning_guidelines(evidence_mode: str) -> str:
    primary_filter = (
        "1. **BALANCED CONTENT EVALUATION**: Carefully analyze the input video's structural progression "
        "and logical flow based on the captions. Evaluate whether the scenes are organized with a clear focus, "
        "a purposeful sequence, or rich sensory details (as defined in the Core Distinction Criteria). "
        "Do not automatically default to Label 0 just because the text describes everyday or routine actions; "
        "instead, judge whether those actions build toward a meaningful progression or outcome."
    )

    if evidence_mode == "none":
        return (
            f"{primary_filter}\n"
            "2. **FINAL DECISION**: Synthesize your observations across all four core dimensions with equal "
            "probability. Base your final label and plain-language explanation strictly on this objective structural analysis."
        )
    
    return (
        f"{primary_filter}\n"
        "2. **REFERENCE CROSS-EXAMINATION**: Examine the provided reference examples as contextual anchors. "
        "Look for similarities in structural dynamics, theme progression, or reaction patterns to help calibrate "
        "your judgment, especially for borderline cases.\n"
        "3. **INTEGRATED JUDGMENT**: Combine your independent content analysis with the evidence from the references. "
        "If the reference library shows a strong consensus (agreement: strong), give that structural signal "
        "significant weight in your final prediction."
    )


def build_llm_prompt(video_context_text, content_similarity_text, narrative_pattern_text, evidence_mode) -> str:
    sections = [f"=== 1. INPUT VIDEO CONTENT ===\n{video_context_text}"]
    n = 2
    if content_similarity_text is not None:
        sections.append(f"=== {n}. CONTENT SIMILARITY REFERENCE ===\n{content_similarity_text}")
        n += 1
    if narrative_pattern_text is not None:
        sections.append(f"=== {n}. DISCOURSE PATTERN REFERENCE ===\n{narrative_pattern_text}")
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
# 9. ENSEMBLE — tie-break theo CONFIDENCE, không theo kênh cố định
# ==========================================

def compute_ensemble_label(llm_pred, content_lean, content_conf, narrative_lean, narrative_conf):
    """
    Vote đa số giữa 3 tín hiệu. Tie-break dựa trên CONFIDENCE nội tại của
    từng lean (ưu tiên lean có confidence 'strong'), KHÔNG ưu tiên cố định
    1 kênh cụ thể.
    """
    votes = [v for v in [llm_pred, content_lean, narrative_lean] if v in (0, 1)]
    if not votes:
        return llm_pred if llm_pred in (0, 1) else -1

    count = Counter(votes)
    top = count.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        strong_candidates = []
        if content_lean in (0, 1) and content_conf == "strong":
            strong_candidates.append(content_lean)
        if narrative_lean in (0, 1) and narrative_conf == "strong":
            strong_candidates.append(narrative_lean)
        if strong_candidates:
            return strong_candidates[0]
        if llm_pred in (0, 1):
            return llm_pred
        for candidate in (content_lean, narrative_lean):
            if candidate in (0, 1):
                return candidate
    return top[0][0]


# ==========================================
# 10. CONSTANTS & VALIDATION
# ==========================================

REQUIRED_DATA_KEYS = [
    'embeddings', 'scene_ids', 'metadata',
    'edge_index', 'edge_attr', 'rst_links', 'y',
]
MILVUS_OUTPUT_FIELDS = ["scene_uid", "video_id", "video_label", "caption"]


# ==========================================
# 11. HELPER FUNCTIONS
# ==========================================

def generate_input_video_context(folder_name: str, data: dict, data_root: Path, mode="full") -> str:
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

    n_scenes = len(scene_ids_list)
    raw_caption_list = [captions_dict.get(int(sid), "No caption available.") for sid in scene_ids_list]

    if mode in ["content", "none"]:
        display_captions = raw_caption_list
    else:
        display_captions = serialize_discourse_captions(raw_caption_list, data.get('rst_links', []), n_scenes)

    lines = [f"Total scenes: {n_scenes}", ""]
    for idx, scene_id in enumerate(scene_ids_list[:20]):
        cap = display_captions[idx] if idx < len(display_captions) else "No caption available."
        cap = cap[:200] + '...' if len(cap) > 200 else cap
        lines.append(f'  Scene {int(scene_id)}: "{cap}"')

    return "\n".join(lines)


def load_video_representations(video_reps_dir: Path) -> dict:
    pt_path = video_reps_dir / "video_representations.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"{pt_path} not found. Run milvus_compute_video_query.py first.")
    return torch.load(pt_path, map_location="cpu")


def load_valid_videos(data_root: Path, split_file: Path, reps_by_folder: dict, require_reps: bool) -> tuple:
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
# 12. QWEN INFERENCE HELPERS
# ==========================================

def count_tokens_qwen(tokenizer, messages: list) -> int:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tokenizer.encode(text))


def run_qwen_inference(tokenizer, model, messages: list, model_type: str = "instruct",
                       max_new_tokens: int = 1024) -> str:
    """
    Chạy inference với Qwen3, hỗ trợ cả instruct và thinking mode.
    
    Với thinking mode: tự động tách thinking content và trả về final content.
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    
    # Cấu hình generation theo khuyến nghị của từng loại model
    if model_type == "thinking":
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        }
    else:  # instruct
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": 0.0,
        }
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    
    output_ids = outputs[0][input_len:].tolist()
    raw_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    
    # Nếu là thinking mode, tách thinking content và final content
    if model_type == "thinking":
        thinking_token_id = 151668  # Token đặc biệt đánh dấu kết thúc thinking[reference:6]
        try:
            # Tìm vị trí token 151668 từ cuối lên
            idx = len(output_ids) - output_ids[::-1].index(thinking_token_id)
            thinking_content = tokenizer.decode(output_ids[:idx], skip_special_tokens=True).strip("\n")
            final_content = tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip("\n")
            # Chỉ trả về final content (phần JSON)
            return final_content
        except ValueError:
            # Nếu không tìm thấy token thinking, trả về toàn bộ (fallback)
            return raw_output
    
    return raw_output


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
# 13. MAIN
# ==========================================

def main(args: argparse.Namespace) -> None:

    data_root       = Path(args.data_root)
    split_file      = Path(args.split_file)
    checkpoint_path = Path(args.checkpoint_path)
    evidence_mode   = args.evidence_mode

    need_content = evidence_mode in ("content", "full")
    need_edge    = evidence_mode in ("edge", "full")

    milvus_client    = None
    collection_name  = None
    reps_by_folder   = {}

    if need_content or need_edge:
        milvus_endpoint = os.getenv("MILVUS_CLUSTER_ENDPOINT")
        milvus_token    = os.getenv("MILVUS_TOKEN")
        if not all([milvus_endpoint, milvus_token]):
            raise ValueError("Missing MILVUS_CLUSTER_ENDPOINT / MILVUS_TOKEN env vars.")
        milvus_client = MilvusClient(uri=milvus_endpoint, token=milvus_token)

        # Cả 2 kênh giờ dùng CHUNG 1 collection scene-level — không cần rst_edge_index nữa.
        collection_name = args.content_collection_name or os.getenv("MILVUS_COLLECTION_NAME")
        if not collection_name:
            raise ValueError(
                "Missing collection name: pass --content_collection_name or set "
                "MILVUS_COLLECTION_NAME env var (required for evidence_mode='content'/'edge'/'full')."
            )

    if need_content:
        if not args.video_reps_dir:
            raise ValueError("--video_reps_dir is required when evidence_mode is 'content' or 'full'.")
        reps_by_folder = load_video_representations(Path(args.video_reps_dir))
        print(f"[INFO] Loaded {len(reps_by_folder)} video representations from {args.video_reps_dir}.")

    print(f"[INFO] evidence_mode = '{evidence_mode}' "
          f"(content={'ON' if need_content else 'off'}, edge/discourse={'ON' if need_edge else 'off'})")
    if need_content:
        print(f"[INFO] content hyperparams: top_k={args.content_top_k}, search_limit={args.content_search_limit}")
    if need_edge:
        print(f"[INFO] discourse hyperparams: alpha={args.alpha}, top_k={args.discourse_top_k}, "
              f"search_limit={args.discourse_search_limit}")
    print()

    print(f"[INFO] Loading Qwen3-4B-Instruct from: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    llm_model.eval()
    print("[INFO] Model loaded.\n")

    valid_folders, valid_data, invalid_folders = load_valid_videos(
        data_root, split_file, reps_by_folder, require_reps=need_content,
    )
    print(f"[INFO] Test split: {len(valid_folders)} valid, {len(invalid_folders)} invalid.")

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

            video_context_text = generate_input_video_context(folder, sample_data, data_root, args.evidence_mode)

            content_similarity_text = None
            content_hits_record     = []
            content_lean = content_conf = None
            if need_content:
                rep_vec = reps_by_folder.get(folder)
                if rep_vec is None:
                    raise ValueError("Missing video representation")
                res = milvus_client.search(
                    collection_name=collection_name,
                    data=[rep_vec.tolist()],
                    limit=args.content_search_limit,
                    output_fields=MILVUS_OUTPUT_FIELDS,
                )
                top_hits = dedupe_content_hits_by_video(res, args.content_top_k)
                content_similarity_text, content_lean, content_conf = build_content_similarity_context(top_hits)
                content_hits_record = top_hits

            narrative_pattern_text = None
            narrative_hits_record  = []
            narrative_lean = narrative_conf = None
            if need_edge:
                embeddings_norm = F.normalize(sample_data['embeddings'].float(), p=2, dim=1)
                query_dps_captions, ranked_top = retrieve_discourse_evidence(
                    embeddings_norm, sample_data.get('rst_links', []), captions,
                    current_video_id=folder,
                    milvus_client=milvus_client, collection_name=collection_name, data_root=data_root,
                    top_k=args.discourse_top_k, search_limit=args.discourse_search_limit,
                    alpha=args.alpha,
                )
                narrative_pattern_text, narrative_lean, narrative_conf = build_discourse_context(
                    query_dps_captions, ranked_top,
                )
                narrative_hits_record = [{"video_id": vid, **cand} for vid, cand in ranked_top]

            llm_prompt = build_llm_prompt(video_context_text, content_similarity_text, narrative_pattern_text, evidence_mode)
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
                        "Discourse pattern context truncated — prompt exceeded context window.",
                        evidence_mode,
                    )
                elif content_similarity_text is not None:
                    messages[1]["content"] = build_llm_prompt(
                        video_context_text,
                        "Content similarity context truncated — prompt exceeded context window.",
                        narrative_pattern_text, evidence_mode,
                    )
                n_tokens = count_tokens_qwen(tokenizer, messages)

            llm_response = run_qwen_inference(tokenizer, llm_model, messages)
            verdict      = extract_and_parse_json(llm_response)
            pred_label   = verdict.get("predicted_label", -1)

            final_prediction = compute_ensemble_label(pred_label, content_lean, content_conf, narrative_lean, narrative_conf)

            ground_truth = int(
                sample_data['y'].item() if isinstance(sample_data['y'], torch.Tensor) else sample_data['y']
            )

            status = "✓" if ground_truth == pred_label else "✗"
            preview_parts = []
            if content_hits_record:
                preview_parts.append(f"content={content_lean}({content_conf})")
            if narrative_hits_record:
                preview_parts.append(f"discourse={narrative_lean}({narrative_conf})")
            preview_parts.append(f"llm={pred_label}")
            preview_parts.append(f"final={final_prediction}")
            print(f"| GT={ground_truth} llm={pred_label} {status} | tokens={n_tokens:,} | " + " | ".join(preview_parts))

            evaluation_results.append({
                "folder_name":              folder,
                "evidence_mode":            evidence_mode,
                "ground_truth":             ground_truth,
                "prediction":               pred_label,
                "content_lean":             content_lean,
                "content_confidence":       content_conf,
                "narrative_lean":           narrative_lean,
                "narrative_confidence":     narrative_conf,
                "final_prediction":         final_prediction,
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
                "content_lean":             None,
                "content_confidence":       None,
                "narrative_lean":           None,
                "narrative_confidence":     None,
                "final_prediction":         -1,
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

    valid = [r for r in evaluation_results if r.get("ground_truth") is not None]
    if valid:
        llm_acc = sum(1 for r in valid if r["prediction"] == r["ground_truth"]) / len(valid)
        ens_acc = sum(1 for r in valid if r["final_prediction"] == r["ground_truth"]) / len(valid)
        print(f"[SUMMARY] LLM-only accuracy: {llm_acc:.4f} | Ensemble (final_prediction) accuracy: {ens_acc:.4f}")


# ==========================================
# 14. ENTRY POINT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VideoRAG Inference Pipeline — Content Similarity + Discourse Path Serialization "
                    "(DPS) / Nuclearity-weighted re-ranking (thay Dense Edge Retrieval).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_root",         type=str, required=True)
    parser.add_argument("--split_file",        type=str, required=True)
    parser.add_argument("--checkpoint_path",   type=str, required=True)
    parser.add_argument("--evidence_mode",     type=str, default="full", choices=EVIDENCE_MODE_CHOICES)
    parser.add_argument("--content_collection_name", type=str, default=None,
                        help="Milvus collection scene-level DÙNG CHUNG cho cả 2 kênh (content và edge/discourse). "
                             "Mặc định env MILVUS_COLLECTION_NAME.")
    parser.add_argument("--video_reps_dir",    type=str, default=None,
                        help="Dir chứa video_representations.pt. CHỈ cần khi evidence_mode ∈ {content, full}.")
    parser.add_argument("--model_name",        type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model_type", type=str, default="instruct",
                        choices=["instruct", "thinking"],

                    help="Loại mô hình: 'instruct' (Qwen3-4B-Instruct) hoặc 'thinking' (Qwen3-4B-Thinking-2507)")
    # --- Content Similarity hyperparameters ---
    parser.add_argument("--content_top_k", type=int, default=5,
                        help="Số VIDEO khác nhau (đã dedupe) lấy làm evidence cho kênh Content Similarity.")
    parser.add_argument("--content_search_limit", type=int, default=50,
                        help="Số scene thô lấy từ Milvus trước khi dedupe theo video (nên >> content_top_k).")

    # --- Discourse Path Serialization / Nuclearity-weighted hyperparameters ---
    parser.add_argument("--discourse_top_k", type=int, default=5,
                        help="Số VIDEO khác nhau (đã dedupe) lấy làm evidence cho kênh Discourse Pattern.")
    parser.add_argument("--discourse_search_limit", type=int, default=30,
                        help="Số candidate lấy từ Milvus MỖI scene của video test (ANN search).")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Trọng số blend: final = alpha*dense_sim + (1-alpha)*topology_sim. Mặc định 0.7.")

    args = parser.parse_args()
    main(args)