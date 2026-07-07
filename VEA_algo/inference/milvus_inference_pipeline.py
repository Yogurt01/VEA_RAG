"""
milvus_inference_pipeline.py
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
    "content": """1. CONTENT SIMILARITY REFERENCE:
   - These are the most visually and thematically similar REFERENCE VIDEOS found in the
     library (deduplicated — each is a distinct video, not repeated scenes from the same video).
   - The similarity score reflects overall content alignment.
   - Each block reports its own internal agreement (strong / weak / mixed — how consistently
     the matched videos point to one label). Treat "weak" or "mixed" as low-confidence.""",
    "edge": """2. NARRATIVE TRANSITION REFERENCE:
   - These results come from matching individual scene-to-scene transitions of the input video
     against a library of transitions from other (distinct) videos, using content similarity.
   - Each match shows a reference video with a similar transition and its known engagement outcome.
   - The "Match strength" score (0–1) indicates how similar the matched transition is.
   - "Narrative relation diversity" reports how many distinct discourse relations
     (e.g., contrast, elaboration, temporal) the video's own scene transitions use — read this
     alongside the video content itself; it is descriptive, not a rule for either label.
   - This block also reports its own internal agreement (strong / weak / mixed). Treat "weak"
     or "mixed" as low-confidence.""",
}

CONFLICT_RESOLUTION_NOTE = (
    "\nIf the two reference sources disagree, give more weight to whichever source reports "
    "'strong' internal agreement over one reporting 'weak' or 'mixed'. If both are equally "
    "strong (or equally weak), rely primarily on your own reading of the video content rather "
    "than either source."
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
    [FIX] Search trả về hit theo SCENE — dedupe theo video_id (giữ hit điểm
    cao nhất mỗi video) trước khi lấy top_k, để mỗi video chỉ đóng góp
    ĐÚNG 1 phiếu, tránh 1 video có nhiều scene giống nhau chiếm hết top-K.
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
# 6. DENSE EDGE RETRIEVAL — Milvus
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
    pt_path = video_reps_dir / "video_representations.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"{pt_path} not found. Run milvus_compute_video_query.py first.")
    return torch.load(pt_path, map_location="cpu")


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


def compute_scene_depths(rst_links: list, n_scenes: int) -> list:
    """
    BFS-lite depth (khoảng cách tới node ROOT/scene mở đầu) — dùng CHỈ khi
    --depth_penalty_weight > 0. Mặc định TẮT (0.0), không ảnh hưởng hành vi
    gốc trừ khi người dùng chủ động bật để thử nghiệm.
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


def build_query_edges(embeddings_norm: torch.Tensor, rst_links: list, captions: list, depths: list) -> list:
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
            "depth_src":   depths[s_idx] if s_idx < len(depths) else -1,
            "depth_tgt":   depths[t_idx] if t_idx < len(depths) else -1,
        })
    return queries


def search_edge(milvus_client, edge_collection_name: str, query_vec: list, rst_type: str, limit: int):
    try:
        hits = milvus_client.search(
            collection_name=edge_collection_name, data=[query_vec],
            filter=f'rst_type == "{rst_type}"', limit=limit,
            output_fields=EDGE_OUTPUT_FIELDS,
        )[0]
    except Exception:
        hits = []
    if not hits:
        try:
            hits = milvus_client.search(
                collection_name=edge_collection_name, data=[query_vec],
                limit=limit, output_fields=EDGE_OUTPUT_FIELDS,
            )[0]
        except Exception:
            hits = []
    return hits


def score_edge_hit(hit: dict, prior_scores: dict, sim_weight: float, prior_weight: float,
                    query_depth_src: int, depth_penalty_weight: float) -> float:
    cos_sim  = float(hit["distance"])
    rst_type = hit["entity"]["rst_type"]
    prior    = prior_scores.get(rst_type, 0.5)
    score = sim_weight * cos_sim + prior_weight * prior

    if depth_penalty_weight > 0 and query_depth_src >= 0:
        hit_depth_src = hit["entity"].get("depth_src", -1)
        if hit_depth_src >= 0:
            penalty = depth_penalty_weight * min(abs(query_depth_src - hit_depth_src), 3) / 3
            score -= penalty
    return score


def retrieve_narrative_evidence(
    embeddings_norm: torch.Tensor, rst_links: list, captions: list, depths: list,
    current_video_id: str, milvus_client, edge_collection_name: str,
    prior_scores: dict, data_root: Path, top_k: int, candidate_limit: int,
    sim_weight: float, prior_weight: float, depth_penalty_weight: float,
) -> tuple:
    query_edges = build_query_edges(embeddings_norm, rst_links, captions, depths)
    if not query_edges:
        return query_edges, []

    best_per_video = {}
    for qe in query_edges:
        hits = search_edge(milvus_client, edge_collection_name, qe["vector"], qe["rst_type"], candidate_limit)
        for h in hits:
            vid = h["entity"]["video_id"]
            if vid == current_video_id:
                continue
            score = score_edge_hit(h, prior_scores, sim_weight, prior_weight,
                                    qe["depth_src"], depth_penalty_weight)
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


def get_local_subgraph_context(video_id: str, src_scene_id: int, tgt_scene_id: int,
                                 data_root: Path, max_neighbors: int = 2) -> str:
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


# ==========================================
# 7. CONTEXT BUILDER — NARRATIVE PATTERNS (không mớm nhãn)
# ==========================================

def build_narrative_pattern_context(query_edges: list, ranked_top: list, data_root: Path) -> tuple:
    """Trả về (context_text, lean, confidence)."""
    lines = []

    if query_edges:
        distinct = len(set(qe['rst_type'] for qe in query_edges))
        total    = len(query_edges)
        diversity = distinct / total if total else 0.0
        lines.append(f"Narrative relation diversity: {distinct}/{total} distinct relation types used ({diversity:.0%}).")
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
        return "\n".join(lines), None, "none"

    lines.append(f"Top {len(ranked_top)} matching transition patterns from reference library:")
    lines.append("")

    n_eng = sum(1 for _, c in ranked_top if c['video_label'] == 1)
    lean, confidence = evidence_lean_and_confidence(n_eng, len(ranked_top))

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
            '- Narrative transitions: label distribution across 5 distinct reference videos is 3×Label 0, 2×Label 1. [agreement: weak]\n'
            '  Top pattern: "Product reveal close-up" → (contrasts with) → "Casual low-energy reaction" '
            '| LOW ENGAGEMENT | Match strength: 0.81'
        )
        bullets.append("- Narrative relation diversity: 2/9 transitions (22%).")
    if evidence_mode == "full":
        bullets.append("- The two sources disagree; content shows 'strong' agreement while narrative shows only "
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
# 9. ENSEMBLE — tie-break theo CONFIDENCE, không theo kênh cố định
# ==========================================

def compute_ensemble_label(llm_pred, content_lean, content_conf, narrative_lean, narrative_conf):
    """
    Vote đa số giữa 3 tín hiệu. Tie-break dựa trên CONFIDENCE nội tại của
    từng lean (ưu tiên lean có confidence 'strong'), KHÔNG ưu tiên cố định
    1 kênh cụ thể — để ổn định hơn qua các seed/split khác nhau, vì kênh
    nào mạnh hơn có thể đổi theo dữ liệu chứ không phải hằng số.
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


def run_qwen_inference(tokenizer, model, messages: list, max_new_tokens: int = 1024) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


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

    milvus_client         = None
    collection_name       = None
    edge_collection_name  = None
    reps_by_folder        = {}
    prior_scores          = {}

    if need_content or need_edge:
        milvus_endpoint = os.getenv("MILVUS_CLUSTER_ENDPOINT")
        milvus_token    = os.getenv("MILVUS_TOKEN")
        if not all([milvus_endpoint, milvus_token]):
            raise ValueError("Missing MILVUS_CLUSTER_ENDPOINT / MILVUS_TOKEN env vars.")
        milvus_client = MilvusClient(uri=milvus_endpoint, token=milvus_token)

    if need_content:
        collection_name = args.content_collection_name or os.getenv("MILVUS_COLLECTION_NAME")
        if not collection_name:
            raise ValueError("Missing collection name: pass --content_collection_name or set MILVUS_COLLECTION_NAME env var (required for evidence_mode='content'/'full').")
        if not args.video_reps_dir:
            raise ValueError("--video_reps_dir is required when evidence_mode is 'content' or 'full'.")
        reps_by_folder = load_video_representations(Path(args.video_reps_dir))
        print(f"[INFO] Loaded {len(reps_by_folder)} video representations from {args.video_reps_dir}.")

    if need_edge:
        edge_collection_name = args.edge_collection_name or os.getenv("MILVUS_EDGE_COLLECTION_NAME", "rst_edge_index")
        if not milvus_client.has_collection(edge_collection_name):
            raise RuntimeError(f"Collection '{edge_collection_name}' not found. Run milvus_build_edge_index.py first.")
        if not args.edge_index_dir:
            raise ValueError("--edge_index_dir is required when evidence_mode is 'edge' or 'full'.")
        prior_scores = load_prior_scores(Path(args.edge_index_dir))
        print(f"[INFO] Loaded {len(prior_scores)} rst_type priors from {args.edge_index_dir}.")

    print(f"[INFO] evidence_mode = '{evidence_mode}' "
          f"(content={'ON' if need_content else 'off'}, edge={'ON' if need_edge else 'off'})")
    if need_content:
        print(f"[INFO] content hyperparams: top_k={args.content_top_k}, search_limit={args.content_search_limit}")
    if need_edge:
        print(f"[INFO] edge hyperparams: sim_weight={args.sim_weight}, prior_weight={args.prior_weight}, "
              f"top_k_evidence={args.top_k_evidence}, candidate_limit={args.candidate_limit}, "
              f"depth_penalty_weight={args.depth_penalty_weight}")
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

            video_context_text = generate_input_video_context(folder, sample_data, data_root)

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
                depths = compute_scene_depths(sample_data.get('rst_links', []), embeddings_norm.shape[0]) \
                    if args.depth_penalty_weight > 0 else [-1] * embeddings_norm.shape[0]
                query_edges, ranked_top = retrieve_narrative_evidence(
                    embeddings_norm, sample_data.get('rst_links', []), captions, depths,
                    current_video_id=folder,
                    milvus_client=milvus_client, edge_collection_name=edge_collection_name,
                    prior_scores=prior_scores, data_root=data_root,
                    top_k=args.top_k_evidence, candidate_limit=args.candidate_limit,
                    sim_weight=args.sim_weight, prior_weight=args.prior_weight,
                    depth_penalty_weight=args.depth_penalty_weight,
                )
                narrative_pattern_text, narrative_lean, narrative_conf = build_narrative_pattern_context(
                    query_edges, ranked_top, data_root,
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
                preview_parts.append(f"narrative={narrative_lean}({narrative_conf})")
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
        description="VideoRAG Inference Pipeline v3 — dedupe fix for content channel, neutral prompt, "
                    "exposed hyperparameters for sweeping.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_root",         type=str, required=True)
    parser.add_argument("--split_file",        type=str, required=True)
    parser.add_argument("--checkpoint_path",   type=str, required=True)
    parser.add_argument("--evidence_mode",     type=str, default="full", choices=EVIDENCE_MODE_CHOICES)
    parser.add_argument("--content_collection_name", type=str, default=None,
                        help="Milvus collection cho Content Similarity. Mặc định env MILVUS_CONTENT_COLLECTION_NAME.")
    parser.add_argument("--video_reps_dir",    type=str, default=None)
    parser.add_argument("--edge_index_dir",    type=str, default=None)
    parser.add_argument("--edge_collection_name", type=str, default=None)
    parser.add_argument("--model_name",        type=str, default=DEFAULT_MODEL_NAME)

    # --- Content Similarity hyperparameters ---
    parser.add_argument("--content_top_k", type=int, default=5,
                        help="Số VIDEO khác nhau (đã dedupe) lấy làm evidence cho kênh Content Similarity.")
    parser.add_argument("--content_search_limit", type=int, default=50,
                        help="Số scene thô lấy từ Milvus trước khi dedupe theo video (nên >> content_top_k).")

    # --- Dense Edge Retrieval hyperparameters ---
    parser.add_argument("--top_k_evidence", type=int, default=5,
                        help="Số VIDEO khác nhau (đã dedupe) lấy làm evidence cho kênh Narrative Transition.")
    parser.add_argument("--candidate_limit", type=int, default=50,
                        help="Số candidate lấy từ Milvus MỖI query edge (ANN search).")
    parser.add_argument("--sim_weight", type=float, default=0.85,
                        help="Trọng số cosine similarity trong công thức chấm điểm edge.")
    parser.add_argument("--prior_weight", type=float, default=0.15,
                        help="Trọng số prior theo rst_type trong công thức chấm điểm edge.")
    parser.add_argument("--depth_penalty_weight", type=float, default=0.0,
                        help="Trọng số phạt lệch vị trí trong mạch truyện (DDE-lite). 0.0 = tắt (mặc định).")

    args = parser.parse_args()
    main(args)