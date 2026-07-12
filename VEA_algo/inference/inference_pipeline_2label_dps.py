"""
inference_pipeline_2label_dps.py
--------------------------------------------------------------------------------------------------
Usage:
    python inference_pipeline_2label_dps.py --evidence_mode milvus \
        --data_root .../All_Videos --split_file .../dataset_splits.json \
        --checkpoint_path .../ablation_dps.json \
        --collection_name video_scenes_collection \
        --video_reps_dir .../video_representations_dir \
        --alpha 0.7 --top_k 5 --search_limit 30 \
        --model_name .../Qwen3-4B-Instruct-2507

    # Precompute retrieval cache (CPU only)
    python inference_pipeline_2label_dps.py --evidence_mode milvus \
        --data_root .../All_Videos --split_file .../dataset_splits.json \
        --collection_name video_scenes_collection \
        --video_reps_dir .../video_representations_dir \
        --alpha 0.7 --top_k 5 --search_limit 30 \
        --retrieval_cache_path .../indexing/retrieval_cache_dps.json --precompute_retrieval
"""

import os
import re
import gc
import time
import json
import argparse
import torch
from pathlib import Path
from collections import Counter
from pymilvus import MilvusClient
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MAX_CONTEXT_TOKENS  = 32768
EVIDENCE_MODE_CHOICES = ["none", "milvus"]


# ==========================================
# 1. RST NORMALIZATION & TOPOLOGY CONFIG
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

def serialize_discourse_captions(scene_ids: list, rst_links: list, captions_list: list, hit_scene_id: int = None) -> str:
    """
    Sắp xếp và cấu trúc lại chuỗi caption dựa trên mối quan hệ Discourse (RST Links).
    Nếu có hit_scene_id (dành cho video tham chiếu), hàm sẽ tự động cắt tỉa nhánh cây câu chuyện 
    chỉ giữ lại cảnh đó và các cảnh cha/con liên quan trực tiếp để tối ưu hóa lượng token.
    """
    if not rst_links:
        if hit_scene_id is not None:
            try:
                idx = scene_ids.index(hit_scene_id)
                return f"  - Scene {hit_scene_id}: \"{captions_list[idx][:150]}\""
            except ValueError:
                pass
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
        
    # --- THUẬT TOÁN CẮT TỈA: CHỈ GIỮ LẠI CẢNH CHA VÀ CON LIÊN QUAN TRỰC TIẾP ---
    relevant_nodes = None
    if hit_scene_id is not None:
        relevant_nodes = {hit_scene_id}
        if hit_scene_id in parent_map:
            relevant_nodes.add(parent_map[hit_scene_id])
        if hit_scene_id in adj:
            for child, _ in adj[hit_scene_id]:
                relevant_nodes.add(child)
        
    visited = set()
    lines = []
    
    def dfs(node, depth=0):
        if node in visited:
            return
        visited.add(node)
        
        # Chỉ ghi nhận phân cảnh nếu nó nằm trong bộ lọc các node liên quan cục bộ
        if relevant_nodes is None or node in relevant_nodes:
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
        
    # Xử lý các node cô lập (Chỉ hiển thị nếu chính nó là node cô lập trúng tuyển từ Milvus)
    for sid in scene_ids:
        if sid not in visited:
            if relevant_nodes is None or sid in relevant_nodes:
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

REFERENCE_SOURCE_DOC = """REFERENCE VIDEOS (PRIMARY SOURCE):
   - These are the most similar REFERENCE VIDEOS found in the library, retrieved by combining content
     similarity with narrative/discourse-structure similarity.
   - For accuracy, each reference video has been structured using Discourse Path Serialization (DPS) 
     to preserve its logical story hierarchy.
   - Treat "weak" or "mixed" agreement blocks as low-confidence signals."""

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

*Note: When analyzing textual captions organized by Discourse Path Serialization (DPS), pay close attention to how sub-scenes elaborate on or contrast with the Narrative Core/Root scenes.*
"""


def build_system_prompt(evidence_mode: str) -> str:
    if evidence_mode == "milvus":
        intro = " cross-reference it with similar serialized discourse contexts retrieved from a reference library,"
        reference_block = "How to use the reference source below:\n\n" + REFERENCE_SOURCE_DOC
        reasoning_intro = "- Read the video content (structured via DPS) first, then use the reference as supporting evidence."
    else:
        intro = ""
        reference_block = "No external reference library is used in this setting — base your decision solely on the video content described below."
        reasoning_intro = "- Base your decision solely on the input video's own chronological content."

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
# 5. EVIDENCE LEAN + CONFIDENCE
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


# ==========================================
# 6. NUCLEARITY SCORING & HYBRID RETRIEVAL
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
            "rst_links":       rst_links
        }
    except Exception:
        data = None
    cache[video_id] = data
    return data


DISCOURSE_OUTPUT_FIELDS = ["scene_uid", "video_id", "video_label", "caption"]


def retrieve_discourse_evidence(
    query_vec: list, rst_links: list,
    current_video_id: str, milvus_client, collection_name: str, data_root: Path,
    top_k: int, search_limit: int, alpha: float,
) -> list:
    if query_vec is None:
        return []

    query_topology = compute_topology_vector(rst_links)
    video_cache = {}
    best_per_video = {}

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
            
            # --- BẮT ĐẦU TRÍCH XUẤT ĐÚNG ID CỦA SCENE ĐƯỢC RETRIEVE ---
            hit_scene_id = None
            scene_uid = h["entity"].get("scene_uid", "")
            
            # Khớp chuỗi UID dạng "video123_scene_2" hoặc "video123_2" để lấy ID số phân cảnh
            match = re.search(r'_scene_(\d+)$|_(\d+)$', str(scene_uid))
            if match:
                hit_scene_id = int(match.group(1) or match.group(2))
            
            # Dự phòng: Tự động so khớp văn bản nếu chuỗi UID không chứa số phân cảnh rõ ràng
            if hit_scene_id is None or hit_scene_id not in cand_data["scene_ids"]:
                hit_caption = h["entity"].get("caption", "").strip()
                for sid, cap in zip(cand_data["scene_ids"], cand_data["raw_captions"]):
                    if hit_caption[:40] in cap or cap[:40] in hit_caption:
                        hit_scene_id = sid
                        break
            
            # Nếu tất cả thất bại, đặt mặc định về phân cảnh đầu tiên của video để tránh lỗi crash
            if hit_scene_id is None and cand_data["scene_ids"]:
                hit_scene_id = cand_data["scene_ids"][0]
            # --------------------------------------------------------

            # Áp dụng DPS thu gọn cho Video Tham Chiếu bằng cách truyền tham số hit_scene_id
            serialized_dps = serialize_discourse_captions(
                cand_data["scene_ids"], cand_data["rst_links"], cand_data["raw_captions"],
                hit_scene_id=hit_scene_id
            )

            best_per_video[vid] = {
                "score":           final_score,
                "dense_sim":       dense_sim,
                "topology_sim":    topo_sim,
                "video_label":     h["entity"]["video_label"],
                "serialized_dps":  serialized_dps,
            }

    ranked = sorted(best_per_video.items(), key=lambda kv: kv[1]["score"], reverse=True)
    return ranked[:top_k]


# ==========================================
# 7. CONTEXT BUILDER — REFERENCE
# ==========================================

def build_discourse_context(ranked_top: list) -> tuple:
    lines = []

    if not ranked_top:
        lines.append("No similar reference videos found in the library.")
        return "\n".join(lines), None, "none"

    n_eng = sum(1 for _, c in ranked_top if c['video_label'] == 1)
    lean, confidence = evidence_lean_and_confidence(n_eng, len(ranked_top))

    lines.append(f"Top {len(ranked_top)} matching reference videos from the library (Serialized via DPS):")
    lines.append("")
    for rank, (vid, cand) in enumerate(ranked_top, 1):
        outcome = "HIGH ENGAGEMENT" if cand['video_label'] == 1 else "LOW ENGAGEMENT"
        lines.append(
            f"[{rank}] Reference Video ID: {vid} | Outcome: {outcome} | Blended score: {cand['score']:.4f}\n"
            f"     --- Serialized Discourse Path ---"
        )
        indented_dps = "\n".join(f"     {line}" for line in cand["serialized_dps"].split("\n"))
        lines.append(indented_dps)
        lines.append("     ---------------------------------")
        lines.append("")

    n_neng = len(ranked_top) - n_eng
    lines.append(
        f"Label distribution across {len(ranked_top)} distinct reference videos: "
        f"Label 1: {n_eng}, Label 0: {n_neng}  [agreement: {confidence}]"
    )

    return "\n".join(lines), lean, confidence


# ==========================================
# 8. USER PROMPT BUILDER
# ==========================================

def build_reasoning_example(evidence_mode: str) -> str:
    if evidence_mode == "milvus":
        bullets = ["- Reference videos: label distribution across 5 distinct reference videos is 4×Label 1, 1×Label 0. [agreement: strong]"]
    else:
        bullets = ["- No reference library is used in this setting — judged purely from the video's own content and structure."]

    example_bullets = "\n".join(bullets)
    return f"""---
[Example]:
{example_bullets}

[Expected output]:
{{
  "predicted_label": "1",
  "explanation": "Scene 2 delivers a clear, specific hook (the reveal itself) which acts as the narrative root, and the reference videos with a similar pattern and strong agreement were mostly High Engagement. The transition sequence builds toward that reveal rather than repeating itself.",
  "improvement_suggestions": [
    "Add a brief reaction shot right after the reveal in Scene 2 to extend the payoff.",
    "Vary the pacing slightly earlier in the video to build more anticipation before the reveal."
  ]
}}
---"""


def build_reasoning_guidelines(evidence_mode: str) -> str:
    primary_filter = (
        "1. **BALANCED CONTENT EVALUATION**: Carefully analyze the input video's structural progression "
        "and logical flow based on the text. Evaluate whether the scenes are organized with a clear focus, "
        "a purposeful sequence, or rich sensory details (as defined in the Core Distinction Criteria). "
        "Do not automatically default to Label 0 just because the text describes everyday or routine actions; "
        "instead, judge whether those actions build toward a meaningful progression or outcome."
    )

    if evidence_mode == "none":
        return (
            f"{primary_filter}\n"
            "2. **FINAL DECISION**: Synthesize your observations across all four core dimensions with equal "
            "probability. Base your final label and plain-language explanation strictly on this objective chronological analysis."
        )

    return (
        f"{primary_filter}\n"
        "2. **REFERENCE CROSS-EXAMINATION**: Examine the provided reference examples as contextual anchors. "
        "Look for similarities in structural dynamics (DPS hierarchies), theme progression, or reaction patterns to help calibrate "
        "your judgment, especially for borderline cases.\n"
        "3. **INTEGRATED JUDGMENT**: Combine your independent content analysis with the evidence from the references. "
        "If the reference library shows a strong consensus (agreement: strong), give that structural signal "
        "significant weight in your final prediction."
    )


def build_llm_prompt(video_context_text, reference_text, evidence_mode) -> str:
    sections = [f"=== 1. INPUT VIDEO CONTENT ===\n{video_context_text}"]
    n = 2
    if reference_text is not None:
        sections.append(f"=== {n}. REFERENCE ===\n{reference_text}")
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
# 9. ENSEMBLE SYSTEM
# ==========================================

def compute_ensemble_label(llm_pred, ref_lean, ref_conf):
    votes = [v for v in [llm_pred, ref_lean] if v in (0, 1)]
    if not votes:
        return llm_pred if llm_pred in (0, 1) else -1

    count = Counter(votes)
    top = count.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        if ref_lean in (0, 1) and ref_conf == "strong":
            return ref_lean
        if llm_pred in (0, 1):
            return llm_pred
        return ref_lean
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

def generate_input_video_context(folder_name: str, data: dict, data_root: Path, mode="milvus") -> str:
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

    raw_caption_list = [captions_dict.get(int(sid), "No caption available.") for sid in scene_ids_list]

    if mode == "milvus":
        # Áp dụng Discourse Path Serialization (DPS) đầy đủ không cắt tỉa cho video đầu vào
        rst_links = data.get("rst_links", [])
        return serialize_discourse_captions(scene_ids_list, rst_links, raw_caption_list, hit_scene_id=None)
    else:
        # Giữ nguyên cấu trúc thời gian tuyến tính bình thường ở mode "none"
        n_scenes = len(scene_ids_list)
        lines = [f"Total scenes: {n_scenes} (Chronological Sequence)", ""]
        for idx, scene_id in enumerate(scene_ids_list[:20]):
            cap = raw_caption_list[idx] if idx < len(raw_caption_list) else "No caption available."
            cap = cap[:130] + '...' if len(cap) > 130 else cap
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
# 12. RETRIEVAL CACHE HELPERS
# ==========================================

def load_retrieval_cache(path: Path) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        print(f"[INFO] Loaded retrieval cache: {len(cache)} videos from {path}")
        return cache
    except Exception as e:
        print(f"[WARNING] Failed to load retrieval cache ({e}), starting empty.")
        return {}


def save_retrieval_cache(cache: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ==========================================
# 13. PRECOMPUTE RETRIEVAL (CPU only)
# ==========================================

def run_precompute_retrieval(args: argparse.Namespace) -> None:
    """
    Chạy riêng bước retrieval (Milvus + topology re-rank + DPS) cho toàn bộ video test,
    lưu kết quả vào --retrieval_cache_path, KHÔNG load model LLM — chạy trên CPU.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if not args.retrieval_cache_path:
        raise ValueError("--retrieval_cache_path is required when --precompute_retrieval is set.")

    data_root     = Path(args.data_root)
    split_file    = Path(args.split_file)
    cache_path    = Path(args.retrieval_cache_path)
    evidence_mode = args.evidence_mode

    if evidence_mode != "milvus":
        print(f"[INFO] evidence_mode='{evidence_mode}' không cần retrieval — không có gì để precompute.")
        return

    milvus_endpoint = os.getenv("MILVUS_CLUSTER_ENDPOINT")
    milvus_token    = os.getenv("MILVUS_TOKEN")
    if not all([milvus_endpoint, milvus_token]):
        raise ValueError("Missing MILVUS_CLUSTER_ENDPOINT / MILVUS_TOKEN env vars.")
    milvus_client = MilvusClient(uri=milvus_endpoint, token=milvus_token)

    collection_name = args.collection_name or os.getenv("MILVUS_COLLECTION_NAME")
    if not collection_name:
        raise ValueError("Missing collection name: pass --collection_name or set MILVUS_COLLECTION_NAME.")

    if not args.video_reps_dir:
        raise ValueError("--video_reps_dir is required when evidence_mode='milvus'.")
    reps_by_folder = load_video_representations(Path(args.video_reps_dir))
    print(f"[INFO] Loaded {len(reps_by_folder)} video representations from {args.video_reps_dir}.")

    valid_folders, valid_data, invalid_folders = load_valid_videos(
        data_root, split_file, reps_by_folder, require_reps=True,
    )
    print(f"[INFO] Test split: {len(valid_folders)} valid, {len(invalid_folders)} invalid.")

    cache = load_retrieval_cache(cache_path)
    queue = [f for f in valid_folders if f not in cache]
    print(f"[PRECOMPUTE] evidence_mode='milvus' (DPS Local Subtree Pruning enabled) | {len(queue)} videos left to retrieve "
          f"(CPU only, no LLM loaded).\n")

    for i, folder in enumerate(queue, 1):
        print(f"[{i}/{len(queue)}] {folder}", end=" ", flush=True)
        sample_data = valid_data[folder]
        try:
            with open(data_root / folder / "segments.json", 'r', encoding='utf-8') as f:
                segments = json.load(f)
            captions = load_captions_by_index(segments, sample_data['scene_ids'])

            rep_vec = reps_by_folder.get(folder)
            if rep_vec is None:
                raise ValueError("Missing video representation")

            # Cross-Modal Self-Querying + Nuclearity-Weighted RAG với DPS (Cắt tỉa phân cảnh cục bộ)
            ranked_top = retrieve_discourse_evidence(
                rep_vec.squeeze().tolist(), sample_data.get('rst_links', []),
                current_video_id=folder,
                milvus_client=milvus_client, collection_name=collection_name, data_root=data_root,
                top_k=args.top_k, search_limit=args.search_limit,
                alpha=args.alpha,
            )
            reference_text, reference_lean, reference_conf = build_discourse_context(ranked_top)
            reference_hits_record = [{"video_id": vid, **cand} for vid, cand in ranked_top]

            cache[folder] = {
                "reference_text":      reference_text,
                "reference_lean":      reference_lean,
                "reference_confidence": reference_conf,
                "reference_hits_record": reference_hits_record,
            }
            print("| OK")

        except Exception as e:
            print(f"| ERROR: {e}")
            cache[folder] = {"error": str(e)}

        if i % 10 == 0 or i == len(queue):
            save_retrieval_cache(cache, cache_path)

    save_retrieval_cache(cache, cache_path)
    print(f"\n[DONE] Precompute finished. {len(cache)} videos cached at {cache_path}")


# ==========================================
# 14. QWEN INFERENCE HELPERS
# ==========================================

def count_tokens_qwen(tokenizer, messages: list) -> int:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tokenizer.encode(text))


def run_qwen_inference(tokenizer, model, messages: list, model_type: str = "instruct",
                       max_new_tokens: int = 1024) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    
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
    
    if model_type == "thinking":
        thinking_token_id = 151668
        try:
            idx = len(output_ids) - output_ids[::-1].index(thinking_token_id)
            final_content = tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip("\n")
            return final_content
        except ValueError:
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
# 15. MAIN EXECUTOR
# ==========================================

def main(args: argparse.Namespace) -> None:
    data_root       = Path(args.data_root)
    split_file      = Path(args.split_file)
    checkpoint_path = Path(args.checkpoint_path)
    evidence_mode   = args.evidence_mode

    need_reference = (evidence_mode == "milvus")

    milvus_client    = None
    collection_name  = None
    reps_by_folder   = {}

    if need_reference:
        milvus_endpoint = os.getenv("MILVUS_CLUSTER_ENDPOINT")
        milvus_token    = os.getenv("MILVUS_TOKEN")
        if not all([milvus_endpoint, milvus_token]):
            raise ValueError("Missing MILVUS_CLUSTER_ENDPOINT / MILVUS_TOKEN env vars.")
        milvus_client = MilvusClient(uri=milvus_endpoint, token=milvus_token)

        collection_name = args.collection_name or os.getenv("MILVUS_COLLECTION_NAME")
        if not collection_name:
            raise ValueError(
                "Missing collection name: pass --collection_name or set "
                "MILVUS_COLLECTION_NAME env var (required for evidence_mode='milvus')."
            )

        if not args.video_reps_dir:
            raise ValueError(
                "--video_reps_dir is required when evidence_mode='milvus' "
                "(V_video từ Cross-Modal Self-Querying là query vector cho retrieval)."
            )
        reps_by_folder = load_video_representations(Path(args.video_reps_dir))
        print(f"[INFO] Loaded {len(reps_by_folder)} video representations from {args.video_reps_dir}.")

    print(f"[INFO] evidence_mode = '{evidence_mode}' (reference retrieval={'ON' if need_reference else 'off'})")
    if need_reference:
        print(f"[INFO] retrieval hyperparams: alpha={args.alpha}, top_k={args.top_k}, "
              f"search_limit={args.search_limit}")
    print()

    print(f"[INFO] Loading Qwen model from: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    llm_model.eval()
    print("[INFO] Model loaded.\n")

    valid_folders, valid_data, invalid_folders = load_valid_videos(
        data_root, split_file, reps_by_folder, require_reps=need_reference,
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

    # Load retrieval cache nếu có
    retrieval_cache = {}
    if args.retrieval_cache_path:
        retrieval_cache = load_retrieval_cache(Path(args.retrieval_cache_path))

    for current_idx, folder in enumerate(queue, 1):
        print(f"[{current_idx}/{len(queue)}] {folder}", end=" ", flush=True)

        sample_data = valid_data[folder]

        try:
            with open(data_root / folder / "segments.json", 'r', encoding='utf-8') as f:
                segments = json.load(f)
            captions = load_captions_by_index(segments, sample_data['scene_ids'])

            # Tạo text ngữ cảnh video đầu vào (Tự động kích hoạt DPS toàn vẹn nếu mode là 'milvus')
            video_context_text = generate_input_video_context(folder, sample_data, data_root, args.evidence_mode)

            reference_text = None
            reference_hits_record = []
            reference_lean = reference_conf = None

            if need_reference:
                # Kiểm tra cache
                cached = retrieval_cache.get(folder)
                if cached and "reference_text" in cached:
                    reference_text = cached["reference_text"]
                    reference_lean = cached["reference_lean"]
                    reference_conf = cached["reference_confidence"]
                    reference_hits_record = cached["reference_hits_record"]
                else:
                    rep_vec = reps_by_folder.get(folder)
                    if rep_vec is None:
                        raise ValueError("Missing video representation (V_video) — required for reference retrieval.")
                    
                    # Cross-Modal Self-Querying truy vấn sinh ra các bản ghi Reference được tích hợp DPS thu gọn nhánh cây cục bộ
                    ranked_top = retrieve_discourse_evidence(
                        rep_vec.squeeze().tolist(), sample_data.get('rst_links', []),
                        current_video_id=folder,
                        milvus_client=milvus_client, collection_name=collection_name, data_root=data_root,
                        top_k=args.top_k, search_limit=args.search_limit,
                        alpha=args.alpha,
                    )
                    reference_text, reference_lean, reference_conf = build_discourse_context(ranked_top)
                    reference_hits_record = [{"video_id": vid, **cand} for vid, cand in ranked_top]

            llm_prompt = build_llm_prompt(video_context_text, reference_text, evidence_mode)
            system_prompt = build_system_prompt(evidence_mode)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": llm_prompt},
            ]

            if current_idx == 1:
                debug_dir = Path("debug_dps_prompts_output")
                debug_dir.mkdir(exist_ok=True)
                
                sys_file_path = debug_dir / f"exact_system_prompt_{evidence_mode}.txt"
                user_file_path = debug_dir / f"exact_user_prompt_{evidence_mode}.txt"
                
                with open(sys_file_path, "w", encoding="utf-8") as f:
                    f.write(messages[0]["content"])
                    
                with open(user_file_path, "w", encoding="utf-8") as f:
                    f.write(messages[1]["content"])
                    
                print(f"\n[DEBUG ALERT] Đã lưu chính xác System Prompt vào: {sys_file_path}")
                print(f"[DEBUG ALERT] Đã lưu chính xác User Prompt vào: {user_file_path}")
                print(f"[DEBUG ALERT] Bạn có thể mở 2 file trên để xem toàn bộ text gửi lên LLM.")
                print("-" * 60)

            n_tokens = count_tokens_qwen(tokenizer, messages)

            if n_tokens > MAX_CONTEXT_TOKENS and reference_text is not None:
                messages[1]["content"] = build_llm_prompt(
                    video_context_text,
                    "Reference context truncated — prompt exceeded context window.",
                    evidence_mode,
                )
                n_tokens = count_tokens_qwen(tokenizer, messages)

            llm_response = run_qwen_inference(tokenizer, llm_model, messages, model_type=args.model_type)
            verdict      = extract_and_parse_json(llm_response)
            pred_label   = verdict.get("predicted_label", -1)

            final_prediction = compute_ensemble_label(pred_label, reference_lean, reference_conf)

            ground_truth = int(
                sample_data['y'].item() if isinstance(sample_data['y'], torch.Tensor) else sample_data['y']
            )

            status = "✓" if ground_truth == pred_label else "✗"
            preview_parts = []
            if reference_hits_record:
                preview_parts.append(f"reference={reference_lean}({reference_conf})")
            preview_parts.append(f"llm={pred_label}")
            preview_parts.append(f"final={final_prediction}")
            print(f"| GT={ground_truth} llm={pred_label} {status} | tokens={n_tokens:,} | " + " | ".join(preview_parts))

            evaluation_results.append({
                "folder_name":              folder,
                "evidence_mode":            evidence_mode,
                "ground_truth":             ground_truth,
                "prediction":               pred_label,
                "reference_lean":           reference_lean,
                "reference_confidence":     reference_conf,
                "final_prediction":         final_prediction,
                "token_count":              n_tokens,
                "explanation":              verdict.get("explanation", ""),
                "improvement_suggestions":  verdict.get("improvement_suggestions", []),
                "raw_llm_output":           llm_response,
                "input_scene_captions":     captions[:20],
                "reference_evidence_hits":  reference_hits_record,
            })

        except Exception as e:
            print(f"| ERROR: {e}")
            evaluation_results.append({
                "folder_name":              folder,
                "evidence_mode":            evidence_mode,
                "ground_truth":             None,
                "prediction":               -1,
                "reference_lean":           None,
                "reference_confidence":     None,
                "final_prediction":         -1,
                "token_count":              0,
                "explanation":              f"Pipeline error: {e}",
                "improvement_suggestions":  [],
                "raw_llm_output":           "",
                "input_scene_captions":     [],
                "reference_evidence_hits":  [],
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
# 16. ENTRY POINT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VideoRAG Inference Pipeline — Cross-Modal Self-Querying (V_video) + "
                    "Nuclearity-Weighted RAG with Discourse Path Serialization (DPS).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_root",         type=str, required=True)
    parser.add_argument("--split_file",        type=str, required=True)
    parser.add_argument("--checkpoint_path",   type=str, required=True)
    parser.add_argument("--evidence_mode",     type=str, default="milvus", choices=EVIDENCE_MODE_CHOICES,
                        help="'none': Không dùng thư viện tham chiếu, giữ nguyên thứ tự video tuyến tính. "
                             "'milvus': Kích hoạt DPS cho cả input video và reference videos tìm kiếm được qua Milvus.")
    parser.add_argument("--collection_name",   type=str, default=None,
                        help="Milvus collection. Mặc định đọc biến môi trường MILVUS_COLLECTION_NAME.")
    parser.add_argument("--video_reps_dir",    type=str, default=None,
                        help="Thư mục chứa file video_representations.pt (Bắt buộc khi chọn mode 'milvus').")
    parser.add_argument("--model_name",        type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model_type",        type=str, default="instruct", choices=["instruct", "thinking"])

    # --- Hyperparameters cho mô hình tìm kiếm ---
    parser.add_argument("--top_k",             type=int, default=5, help="Số lượng video làm bằng chứng tham chiếu sau khi dedupe.")
    parser.add_argument("--search_limit",      type=int, default=50, help="Số lượng scene thô truy vấn ban đầu từ Milvus.")
    parser.add_argument("--alpha",             type=float, default=0.7, help="Trọng số blend: alpha*dense_sim + (1-alpha)*topology_sim.")

    # --- Retrieval cache (precompute trên CPU, đọc lại khi chạy LLM trên GPU) ---
    parser.add_argument("--precompute_retrieval", action="store_true",
                        help="Chỉ chạy retrieval (Cross-Modal Self-Querying + Nuclearity re-rank + DPS) cho toàn bộ video test, "
                             "lưu ra --retrieval_cache_path rồi thoát. KHÔNG load model LLM, chạy trên CPU "
                             "để tiết kiệm chi phí GPU.")
    parser.add_argument("--retrieval_cache_path", type=str, default=None,
                        help="Đường dẫn file .json lưu/đọc kết quả retrieval theo từng video "
                             "(đặt trong folder indexing/, vd: .../indexing/retrieval_cache_dps.json). "
                             "Ghi khi --precompute_retrieval bật; đọc khi tắt (nếu file tồn tại).")

    args = parser.parse_args()
    if args.precompute_retrieval:
        run_precompute_retrieval(args)
    else:
        main(args)