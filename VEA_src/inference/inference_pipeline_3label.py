import os
import re
import gc
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import collections
from collections import Counter, defaultdict, deque
from pymilvus import MilvusClient
from neo4j import GraphDatabase
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

load_dotenv()

DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MAX_CONTEXT_TOKENS  = 32768
EVIDENCE_MODE_CHOICES = ["none", "milvus", "graph", "full"]


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
    "MANNER_MEANS"
]


def rst_to_natural(rst_type: str) -> str:
    return RST_DESCRIPTIONS.get(rst_type.upper(), rst_type.lower().replace('_', ' '))


# ==========================================
# 2. LABEL DEFINITIONS
# ==========================================

LABEL_DEFINITIONS = {
    0: (
        "Low Engaging — The video is purely observational. It presents objects, facts, or events "
        "one after another without ever engaging the viewer directly, and without building toward "
        "any specific point, outcome, or resolution. This holds true regardless of genre or how "
        "much detail or emotion the content itself contains — what matters is whether the video "
        "reaches out to an audience and goes somewhere, not how rich the content is."
    ),
    1: (
        "Neutral — The video has some structure, personality, or forward motion, but is only "
        "partially engaging: it may lightly acknowledge the viewer or move toward a point without "
        "fully committing to either, may proceed through its content without any real tension or "
        "payoff, or may introduce several ideas that never quite converge."
    ),
    2: (
        "High Engaging — The video actively reaches out to the viewer (speaking to them directly, "
        "inviting a response, or showing a real audience responding) AND has a clear sense of "
        "direction — it is working toward answering a question, proving a claim, or resolving a "
        "situation, and it follows through on that by the end."
    ),
}

LABEL_SHORT = {
    0: "LOW ENGAGING",
    1: "NEUTRAL",
    2: "HIGH ENGAGING",
}

_label_def_lines  = "\n".join(
    f"  Label {lbl}: {desc}" for lbl, desc in sorted(LABEL_DEFINITIONS.items())
)
_valid_labels_str = " or ".join(str(k) for k in sorted(LABEL_DEFINITIONS.keys()))
_valid_labels_set = set(LABEL_DEFINITIONS.keys())


# ==========================================
# 3. SYSTEM PROMPT
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
      A dominant distribution (>70% one label) is a powerful signal.

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
    "and compare it with the input video's own structure to assess structural similarity."
)

CALIBRATION_NOTE = (
    "- Judge the video on its own specific merits. Low Engaging, Neutral, and High Engaging "
    "are equally valid default-free outcomes — each requires you to point to a concrete, "
    "identifiable reason in the video (or in the references) rather than an assumption in "
    "either direction."
)

CONTENT_PATTERN_NOTES = """
# CORE DISTINCTION CRITERIA

These criteria are genre-agnostic — apply them the same way to a product review, a vlog, a
tutorial, or a dialogue scene. Judge how the video relates to its audience and where it is
going, not what topic it happens to cover.

| Dimension | Low Engaging (0) | Neutral (1) | High Engaging (2) |
| :--- | :--- | :--- | :--- |
| **Audience Connection** — does the video acknowledge a viewer at all? | Never speaks to, invites, or shows a reaction from an audience — plays out as something merely observed. | Occasionally acknowledges an audience, but only in passing. | Clearly speaks to the viewer, invites a response, or shows a real audience reacting. |
| **Direction** — is the video working toward something? | Simply narrates or lists things as they appear, with no question, goal, or claim being pursued. | Moves through content step by step, but without any real tension, stakes, or payoff along the way. | Pursues a clear question, goal, or claim, with some real stakes or payoff along the way (a test, a comparison, a build-up). |
| **Follow-through** — does it land somewhere? | Never arrives anywhere in particular — it just stops. | Touches on an idea or two but doesn't fully tie them together by the end. | Consistently builds toward and reaches one clear takeaway by the end. |

Important calibration notes:
- A narrator or host simply describing, presenting, or explaining something is NOT by itself
  audience connection — narration ABOUT the content is different from speaking TO the viewer.
- Perfunctory or ritual gestures — an opening greeting, a channel logo/jingle, background music,
  a rhetorical question nobody is meant to answer — do NOT count as audience connection either.
  Only count it when there is a genuine, deliberate moment: direct second-person address ("you"),
  an explicit call-to-action or invitation, or a visible audience actually reacting.
- Most real videos will have at most one or two of these three rows lean positive, not all three
  — do not require a perfect match on every row, but also do not round a video up to Label 1 or 2
  just because it clears one row weakly. A confident, detailed, well-produced video with zero
  genuine audience connection anywhere is still Label 0, however articulate it is.
"""


def build_system_prompt(evidence_mode: str) -> str:
    active_docs = []
    intro_parts = []

    if evidence_mode in ("milvus", "full"):
        active_docs.append(REFERENCE_SOURCE_DOCS["milvus"])
        intro_parts.append("similar contexts retrieved from a reference library")

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
IMPORTANT: Only three labels exist — 0 (Low Engaging), 1 (Neutral), and 2 (High Engaging).

{CONTENT_PATTERN_NOTES}

{reference_block}

Reasoning instructions:
{reasoning_intro}
{CALIBRATION_NOTE}
- Be specific: reference scene numbers and explain why the narrative works or doesn't.
- Write your explanation in plain language — no system names, no technical scores.
"""


# ==========================================
# 4. EVIDENCE LEAN + CONFIDENCE
# ==========================================

def evidence_lean_and_confidence(label_counts: Counter, n_total: int):
    """
    Tổng quát hóa cho 3 nhãn (0=Low Engaging, 1=Neutral, 2=High Engaging), và cũng
    hoạt động đúng với graph evidence (có thể có nhiều hơn 2 nhãn tham gia vote).

    Trả về (lean, confidence_label).
    lean: nhãn chiếm đa số phiếu trong label_counts, None nếu hòa ở vị trí dẫn đầu.
    confidence_label:
        "strong" nếu (số phiếu top1 - số phiếu top2) > 1,
        "weak"   nếu chênh lệch == 1,
        "mixed"  nếu hòa tuyệt đối ở vị trí dẫn đầu,
        "none"   nếu không có evidence nào (n_total == 0).
    """
    if n_total == 0 or not label_counts:
        return None, "none"

    ranked = label_counts.most_common()
    top_label, top_count = ranked[0]
    second_count = ranked[1][1] if len(ranked) > 1 else 0

    if len(ranked) > 1 and ranked[1][1] == top_count:
        return None, "mixed"

    diff = top_count - second_count
    confidence = "weak" if diff <= 1 else "strong"
    return top_label, confidence


def label_distribution_str(label_counts: Counter) -> str:
    return ", ".join(f"Label {lbl}: {cnt}" for lbl, cnt in sorted(label_counts.items()))


# ==========================================
# 5. CONTEXT BUILDER — CONTENT SIMILARITY
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
    n_total = sum(label_counts.values())
    lean, confidence = evidence_lean_and_confidence(label_counts, n_total)

    lines = [
        f"Label distribution across {len(top_hits)} distinct reference videos: "
        f"{label_distribution_str(label_counts)}  [agreement: {confidence}]",
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


def build_discourse_context(query_dps_captions: list, ranked_top: list) -> tuple:
    """Trả về (context_text, lean, confidence)."""
    lines = []

    if not ranked_top:
        lines.append("No matching discourse patterns found in the reference library.")
        return "\n".join(lines), None, "none"

    label_counts = Counter(cand['video_label'] for _, cand in ranked_top)
    lean, confidence = evidence_lean_and_confidence(label_counts, len(ranked_top))

    lines.append(f"Top {len(ranked_top)} matching discourse patterns from reference library:")
    lines.append("")
    for rank, (vid, cand) in enumerate(ranked_top, 1):
        outcome = LABEL_SHORT.get(cand['video_label'], "UNKNOWN")
        lines.append(
            f"[{rank}] Outcome: {outcome}  |  Blended score: {cand['score']:.4f} "
            f"(dense={cand['dense_sim']:.3f}, topology={cand['topology_sim']:.3f})"
        )
        lines.append(f'     "{cand["matched_caption"][:220]}"')
        lines.append("")

    lines.append(
        f"Label distribution across {len(ranked_top)} distinct reference videos: "
        f"{label_distribution_str(label_counts)}  [agreement: {confidence}]"
    )

    return "\n".join(lines), lean, confidence


# ==========================================
# 7. KNOWLEDGE GRAPH (Neo4j)
# ==========================================

def explain_rst_chain(chain: list) -> str:
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
    Trả về structural_matches / similarity_videos / concept_priors / video_details /
    concept_details / concept_ids_set. Các query Cypher không đổi so với bản 2 nhãn —
    đã label-agnostic (chỉ đọc coalesce(video_label, predicted_label)).
    """
    result = {}

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


def get_dominant_label_and_count(label_distribution_json):
    """
    Trả về (dominant_label, dominant_count, total) từ chuỗi JSON label distribution.
    Tổng quát cho N nhãn — vd '{"0": 1, "1": 2, "2": 3}' -> (2, 3, 6).
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
    Dùng evidence_lean_and_confidence tổng quát (Counter đa nhãn) để hoạt động đúng với 3 nhãn.
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
            label_short = LABEL_SHORT.get(label, str(label))

            lines.append(f"  [{idx}] Match Video {idx} | Label: {label} ({label_short})  | Max nodes: {max_nodes}, Total paths: {total_paths}")
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
            label_short = LABEL_SHORT.get(label, str(label))
            avg_score_str = f"{avg_score:.3f}" if isinstance(avg_score, (int, float)) else "N/A"
            lines.append(f"  [{idx}] Video Neighbor '{idx}' (Label: {label} / {label_short}) — Avg similarity {avg_score_str}")
            lines.append("")
        lines.append("")
    else:
        lines.append("** Similarity-based Neighbors**: None found.")
        lines.append("")

    # ----- Concept Details -----
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
                lines.append(f"      Dominant label: {dom_label} ({LABEL_SHORT.get(dom_label, '')})")
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

    n_total = sum(label_counts.values())
    if n_total == 0:
        return "\n".join(lines), None, "none"
    lean, confidence = evidence_lean_and_confidence(label_counts, n_total)
    lines.append(f"** Overall label distribution across all graph evidence: {label_distribution_str(label_counts)}  [agreement: {confidence}]")

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
# 8. USER PROMPT
# ==========================================

def build_reasoning_example(evidence_mode: str) -> str:
    bullets = []
    if evidence_mode in ("milvus", "full"):
        bullets.append("- Content similarity: label distribution across 5 distinct reference videos is 4×Label 0, 1×Label 1. [agreement: strong]")
        bullets.append(
            '- Discourse pattern: label distribution across 5 distinct reference videos is 3×Label 1, 2×Label 0. [agreement: weak]\n'
            '  Top match: Blended score 0.81 (dense=0.85, topology=0.70) | LOW ENGAGING\n'
            '  "A static close-up of an internal component [Discourse: elaborates on Scene 2, eventually leading to the video\'s core scene]"'
        )
    if evidence_mode in ("graph", "full"):
        bullets.append(
            "- Knowledge Graph: RST structural matches show 5 videos, mostly Label 0 (LOW ENGAGING), with chain "
            "ROOT -> TEMPORAL -> TEMPORAL. Concept Details show a dominant Concept with label distribution "
            "{'0': 3, '1': 1}, reinforcing Low Engaging. Similarity neighbors are mixed between Label 0 and Label 1."
        )
    if evidence_mode == "full":
        bullets.append("- Milvus and Knowledge Graph sources broadly agree on Low Engaging, but with different strength; "
                        "content similarity shows 'strong' agreement while graph evidence is more mixed, so content gets "
                        "more weight here per the confidence-based rule — but the video's own content is still the "
                        "primary basis for the final call.")
    if not bullets:
        bullets.append("- No reference library is used in this setting — judged purely from the video's own content and structure.")

    example_bullets = "\n".join(bullets)
    return f"""---
[Example]:
{example_bullets}

[Expected output]:
{{
  "audience_connection": "None found. The captions describe the narrator explaining features scene by scene, but no scene has the narrator speak to 'you', invite a response, or show anyone reacting.",
  "direction_and_followthrough": "The scenes move from one feature to the next in sequence, but nothing is being tested, questioned, or built toward — the video simply stops once the last feature is listed.",
  "predicted_label": "0",
  "explanation": "Despite covering many specific details in a confident, well-organized way, the video never reaches out to a viewer and never works toward a point or payoff — it only narrates what is being shown, scene after scene, which matches Low Engaging rather than Neutral or High Engaging.",
  "improvement_suggestions": [
    "Add a direct line to the viewer (e.g. a question or a call-to-action) near the start or end.",
    "Frame the walkthrough around a specific question or comparison the video will resolve, instead of a plain feature list."
  ]
}}
---"""


def build_reasoning_guidelines(evidence_mode: str) -> str:
    primary_filter = (
        "1. **BALANCED CONTENT EVALUATION**: Carefully read the input video's captions in order and "
        "judge it against the Core Distinction Criteria table above — how it connects with its "
        "audience and whether it goes somewhere. Do not automatically default to Label 0 just because "
        "the text describes everyday or routine actions, and do not default to Label 2 just because "
        "the video is detailed, confident, or well-produced — a good narrator is not the same as "
        "audience connection or follow-through. Base the label on the overall balance across the table."
    )

    if evidence_mode == "none":
        return (
            f"{primary_filter}\n"
            "2. **FINAL DECISION**: Weigh the three dimensions together as a whole, rather than requiring "
            "a perfect match on every row. Base your final label and plain-language explanation strictly "
            "on this objective analysis of the video's own content."
        )

    result = (
        f"{primary_filter}\n"
        "2. **REFERENCE CROSS-EXAMINATION (MANDATORY)**: You MUST explicitly state the majority label "
        "and confidence reported in the reference section(s) before giving your final answer — this is "
        "not optional context, it is a required check. If the reference library reports 'strong' "
        "agreement and your own content analysis is uncertain or borderline, change your prediction to "
        "match the reference majority unless you can cite a specific, concrete detail in the video's "
        "own captions that contradicts it.\n"
        "3. **INTEGRATED JUDGMENT**: Only keep your own independent reading over a 'strong' reference "
        "consensus if you can point to that concrete contradicting detail. If the reference shows 'weak' "
        "or 'mixed' agreement, rely primarily on your own content analysis instead."
    )

    if evidence_mode in ("graph", "full"):
        graph_guidance = (
            "\n4. **GRAPH EVIDENCE INTERPRETATION**: When Knowledge Graph evidence is available:\n"
            "   - **Concept Details**: First, inspect the label distribution and semantic details (Audio/Visual Style, Keywords) "
            "of Concepts found in the matched RST paths. A dominant label distribution that aligns with the input video's "
            "content is a powerful signal.\n"
            "   - **RST Structural Matches**: Evaluate how many matched videos share the same RST chain and their labels. "
            "A majority label among matched videos supports the corresponding prediction.\n"
            "   - **Video RST Summary**: Use this to understand the overall discourse structure of the matched videos. "
            "Compare it with your own analysis of the input video's structure to see if they align.\n"
            "   - **Similarity-based Neighbors**: Consider videos with high average similarity. If they mostly agree on "
            "one label, it reinforces the prediction; if mixed, treat with caution.\n"
            "Use strong consensus across multiple sources as high-confidence evidence."
        )
        result += graph_guidance

    return result


def build_llm_prompt(video_context_text, content_similarity_text, narrative_pattern_text, evidence_mode, graph_text=None) -> str:
    sections = [
        "## 1. Target Video Content (Primary Ground Truth)",
        video_context_text
    ]

    if content_similarity_text:
        sections.extend(["## 2. Content References (Primary Baseline)", content_similarity_text])
    if narrative_pattern_text:
        sections.extend(["## 3. Discourse References (Secondary Supplementary)", narrative_pattern_text])
    if graph_text:
        sections.extend(["## 4. Knowledge Graph References (Structural / RST-Chain Match)", graph_text])

    sections.append(f"## Reasoning Guidelines\n{build_reasoning_guidelines(evidence_mode)}")

    has_reference = bool(content_similarity_text or narrative_pattern_text or graph_text)
    reference_field = (
        '\n  "reference_agreement": "State the majority label and confidence reported in the '
        'reference section(s) above, and explicitly say whether your own reading of the video '
        'content agrees or disagrees with it, and why.",' if has_reference else ""
    )

    sections.append("## Instruction")
    sections.append(f"""Analyze the target video based on the criteria. Output ONLY a valid JSON object.
Do not include any markdown block ticks like ```json outside the braces.

Expected JSON format:
{{
  "audience_connection": "Quote the exact scene and phrase that shows direct address, a call-to-action, or a real audience reacting — or write 'None found' if there is none. Do not paraphrase a vague acknowledgment as a quote.",
  "direction_and_followthrough": "Briefly note what the video is working toward, if anything, and whether it follows through by the end.",{reference_field}
  "predicted_label": "{_valid_labels_str}",
  "explanation": "Concisely justify the final predicted label by tying the observations above together. Mention specific scenes.",
  "improvement_suggestions": [
    "Actionable suggestion 1 to increase retention.",
    "Actionable suggestion 2 to increase retention."
  ]
}}""")

    return "\n\n".join(sections)


# ==========================================
# 9. ENSEMBLE
# ==========================================

def compute_ensemble_label(llm_pred, content_lean, content_conf, narrative_lean, narrative_conf):
    """Ensemble cho mode milvus (2 tín hiệu ngoài LLM: content + narrative), trên tập 3 nhãn."""
    votes = [v for v in [llm_pred, content_lean, narrative_lean] if v in _valid_labels_set]
    if not votes:
        return llm_pred if llm_pred in _valid_labels_set else -1

    count = Counter(votes)
    top = count.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        strong_candidates = []
        if content_lean in _valid_labels_set and content_conf == "strong":
            strong_candidates.append(content_lean)
        if narrative_lean in _valid_labels_set and narrative_conf == "strong":
            strong_candidates.append(narrative_lean)
        if strong_candidates:
            return strong_candidates[0]
        if llm_pred in _valid_labels_set:
            return llm_pred
        for candidate in (content_lean, narrative_lean):
            if candidate in _valid_labels_set:
                return candidate
    return top[0][0]


def compute_ensemble_label_two_signal(llm_pred, ref_lean, ref_conf):
    """Ensemble cho mode graph (2 tín hiệu: LLM + graph), trên tập 3 nhãn."""
    votes = [v for v in [llm_pred, ref_lean] if v in _valid_labels_set]
    if not votes:
        return llm_pred if llm_pred in _valid_labels_set else -1
    count = Counter(votes)
    top = count.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        if ref_lean in _valid_labels_set and ref_conf == "strong":
            return ref_lean
        return llm_pred if llm_pred in _valid_labels_set else ref_lean
    return top[0][0]


def compute_ensemble_label_full(llm_pred, content_lean, content_conf, narrative_lean, narrative_conf, graph_lean, graph_conf):
    """Ensemble cho mode full (4 tín hiệu: LLM + content + narrative + graph), trên tập 3 nhãn."""
    votes = [v for v in [llm_pred, content_lean, narrative_lean, graph_lean] if v in _valid_labels_set]
    if not votes:
        return llm_pred if llm_pred in _valid_labels_set else -1
    count = Counter(votes)
    top = count.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        strong_candidates = []
        if content_lean in _valid_labels_set and content_conf == "strong": strong_candidates.append(content_lean)
        if narrative_lean in _valid_labels_set and narrative_conf == "strong": strong_candidates.append(narrative_lean)
        if graph_lean in _valid_labels_set and graph_conf == "strong": strong_candidates.append(graph_lean)
        if strong_candidates: return strong_candidates[0]
        if llm_pred in _valid_labels_set: return llm_pred
        for candidate in (content_lean, narrative_lean, graph_lean):
            if candidate in _valid_labels_set: return candidate
    return top[0][0]


def compute_final_prediction(evidence_mode, pred_label, content_lean, content_conf, narrative_lean, narrative_conf, graph_lean, graph_conf):
    if evidence_mode == "full":
        return compute_ensemble_label_full(pred_label, content_lean, content_conf, narrative_lean, narrative_conf, graph_lean, graph_conf)
    elif evidence_mode == "milvus":
        return compute_ensemble_label(pred_label, content_lean, content_conf, narrative_lean, narrative_conf)
    elif evidence_mode == "graph":
        return compute_ensemble_label_two_signal(pred_label, graph_lean, graph_conf)
    else:
        return pred_label if pred_label in _valid_labels_set else -1


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

def generate_input_video_context(folder_name: str, data: dict, data_root: Path, evidence_mode: str = "milvus") -> str:
    """Hiển thị caption của video test — đi qua DPS cho milvus/full, caption thô cho none/graph."""
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

    if str(evidence_mode).lower() in ("none", "graph"):
        dps_captions = raw_caption_list
    else:
        dps_captions = serialize_discourse_captions(raw_caption_list, data.get('rst_links', []), n_scenes)

    lines = [f"Total scenes: {n_scenes}", ""]
    for idx, scene_id in enumerate(scene_ids_list[:20]):
        cap = dps_captions[idx] if idx < len(dps_captions) else "No caption available."
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
# 12. QWEN INFERENCE HELPERS
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
            "max_new_tokens": max_new_tokens, "do_sample": True,
            "temperature": 0.6, "top_p": 0.95, "top_k": 20,
        }
    else:
        generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False, "temperature": 0.0}

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    output_ids = outputs[0][input_len:].tolist()
    raw_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    if model_type == "thinking":
        thinking_token_id = 151668
        try:
            idx = len(output_ids) - output_ids[::-1].index(thinking_token_id)
            return tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip("\n")
        except ValueError:
            return raw_output

    return raw_output


def extract_and_parse_json(raw_text: str) -> dict:
    valid_labels = _valid_labels_set
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
# 13. RETRIEVAL CACHE
# ==========================================

def load_retrieval_cache(path) -> dict:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        print(f"[INFO] Loaded retrieval cache: {len(cache)} videos from {path}")
        return cache
    except Exception as e:
        print(f"[WARNING] Failed to load retrieval cache ({e}), starting empty.")
        return {}


def save_retrieval_cache(cache: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ==========================================
# 14. PRECOMPUTE RETRIEVAL
# ==========================================

def run_precompute_retrieval(args: argparse.Namespace) -> None:
    """
    Chạy riêng bước retrieval (Milvus content similarity + discourse re-rank, và/hoặc
    Neo4j graph evidence) cho toàn bộ video test, lưu kết quả vào --retrieval_cache_path,
    KHÔNG load model LLM — dùng để chạy trên CPU rẻ, tách khỏi bước inference cần GPU.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if not args.retrieval_cache_path:
        raise ValueError("--retrieval_cache_path is required when --precompute_retrieval is set.")

    data_root     = Path(args.data_root)
    split_file    = Path(args.split_file)
    cache_path    = Path(args.retrieval_cache_path)
    evidence_mode = args.evidence_mode

    need_milvus = evidence_mode in ("milvus", "full")
    need_graph  = evidence_mode in ("graph", "full")

    if not (need_milvus or need_graph):
        print(f"[INFO] evidence_mode='{evidence_mode}' không cần retrieval nào — không có gì để precompute.")
        return

    milvus_client, collection_name, reps_by_folder = None, None, {}
    if need_milvus:
        milvus_endpoint = os.getenv("MILVUS_CLUSTER_ENDPOINT")
        milvus_token    = os.getenv("MILVUS_TOKEN")
        if not all([milvus_endpoint, milvus_token]):
            raise ValueError("Missing MILVUS_CLUSTER_ENDPOINT / MILVUS_TOKEN env vars.")
        milvus_client = MilvusClient(uri=milvus_endpoint, token=milvus_token)

        collection_name = args.collection_name or os.getenv("MILVUS_COLLECTION_NAME")
        if not collection_name:
            raise ValueError("Missing collection name: pass --collection_name or set MILVUS_COLLECTION_NAME.")

        if not args.video_reps_dir:
            raise ValueError("--video_reps_dir is required when evidence_mode is 'milvus' or 'full'.")
        reps_by_folder = load_video_representations(Path(args.video_reps_dir))
        print(f"[INFO] Loaded {len(reps_by_folder)} video representations from {args.video_reps_dir}.")

    neo4j_driver, neo4j_database = None, None
    if need_graph:
        neo4j_driver, neo4j_database = init_neo4j_driver(args)

    valid_folders, valid_data, invalid_folders = load_valid_videos(
        data_root, split_file, reps_by_folder, require_reps=need_milvus,
    )
    print(f"[INFO] Test split: {len(valid_folders)} valid, {len(invalid_folders)} invalid.")

    cache = load_retrieval_cache(cache_path)

    def is_done(entry):
        if entry is None or entry.get("error"):
            return False
        if need_milvus and "content_similarity_text" not in entry:
            return False
        if need_graph and "graph_hits_record" not in entry:
            return False
        return True

    queue = [f for f in valid_folders if not is_done(cache.get(f))]
    print(f"[PRECOMPUTE] evidence_mode='{evidence_mode}' | {len(queue)} videos left to retrieve "
          f"(CPU only, no LLM loaded).\n")

    for i, folder in enumerate(queue, 1):
        print(f"[{i}/{len(queue)}] {folder}", end=" ", flush=True)
        sample_data = valid_data[folder]
        entry = cache.get(folder, {}) or {}
        try:
            with open(data_root / folder / "segments.json", 'r', encoding='utf-8') as f:
                segments = json.load(f)
            captions = load_captions_by_index(segments, sample_data['scene_ids'])

            if need_milvus:
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
                content_text, content_lean, content_conf = build_content_similarity_context(top_hits)
                entry.update({
                    "content_similarity_text": content_text,
                    "content_lean":             content_lean,
                    "content_confidence":       content_conf,
                    "content_hits_record":      top_hits,
                })

                embeddings_norm = F.normalize(sample_data['embeddings'].float(), p=2, dim=1)
                _, ranked_top = retrieve_discourse_evidence(
                    embeddings_norm, sample_data.get('rst_links', []), captions,
                    current_video_id=folder,
                    milvus_client=milvus_client, collection_name=collection_name, data_root=data_root,
                    top_k=args.discourse_top_k, search_limit=args.discourse_search_limit,
                    alpha=args.alpha,
                )
                narrative_text, narrative_lean, narrative_conf = build_discourse_context([], ranked_top)
                entry.update({
                    "narrative_pattern_text":  narrative_text,
                    "narrative_lean":          narrative_lean,
                    "narrative_confidence":    narrative_conf,
                    "narrative_hits_record":   [{"video_id": vid, **cand} for vid, cand in ranked_top],
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

        if i % 10 == 0 or i == len(queue):
            save_retrieval_cache(cache, cache_path)

    save_retrieval_cache(cache, cache_path)

    if neo4j_driver is not None:
        neo4j_driver.close()

    print(f"\n[DONE] Precompute finished. {len(cache)} videos cached at {cache_path}")


# ==========================================
# 15. MAIN
# ==========================================

def main(args: argparse.Namespace) -> None:

    data_root       = Path(args.data_root)
    split_file      = Path(args.split_file)
    checkpoint_path = Path(args.checkpoint_path)
    evidence_mode   = args.evidence_mode

    need_milvus = evidence_mode in ("milvus", "full")
    need_graph  = evidence_mode in ("graph", "full")

    reps_by_folder = {}
    if need_milvus:
        if not args.video_reps_dir:
            raise ValueError("--video_reps_dir is required when evidence_mode is 'milvus' or 'full'.")
        reps_by_folder = load_video_representations(Path(args.video_reps_dir))
        print(f"[INFO] Loaded {len(reps_by_folder)} video representations from {args.video_reps_dir}.")

    print(f"[INFO] evidence_mode = '{evidence_mode}' "
          f"(milvus={'ON' if need_milvus else 'off'}, graph={'ON' if need_graph else 'off'})")
    if need_milvus:
        print(f"[INFO] content hyperparams: top_k={args.content_top_k}, search_limit={args.content_search_limit}")
        print(f"[INFO] discourse hyperparams: alpha={args.alpha}, top_k={args.discourse_top_k}, "
              f"search_limit={args.discourse_search_limit}")
    if need_graph:
        print(f"[INFO] graph hyperparams: graph_top_k={args.graph_top_k}")
    print()

    print(f"[INFO] Loading LLM from: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    if args.load_in_4bit:
        print("[INFO] Loading model with 4-bit quantization (BitsAndBytes)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        llm_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        ).eval()
    else:
        print("[INFO] Loading model in standard bfloat16...")
        llm_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        ).eval()
    print("[INFO] Model loaded.\n")

    valid_folders, valid_data, invalid_folders = load_valid_videos(
        data_root, split_file, reps_by_folder, require_reps=need_milvus,
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

    retrieval_cache = load_retrieval_cache(args.retrieval_cache_path) if args.retrieval_cache_path else {}

    milvus_state = {"client": None, "collection": None}
    def get_milvus_client():
        if milvus_state["client"] is None:
            milvus_state["client"] = MilvusClient(uri=os.getenv("MILVUS_CLUSTER_ENDPOINT"), token=os.getenv("MILVUS_TOKEN"))
            milvus_state["collection"] = args.collection_name or os.getenv("MILVUS_COLLECTION_NAME")
        return milvus_state["client"], milvus_state["collection"]

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

            video_context_text = generate_input_video_context(folder, sample_data, data_root, evidence_mode)

            cached = retrieval_cache.get(folder)

            # ---------- (A) MILVUS: content similarity + discourse pattern ----------
            content_similarity_text, content_hits_record, content_lean, content_conf = None, [], None, None
            narrative_pattern_text, narrative_hits_record, narrative_lean, narrative_conf = None, [], None, None
            if need_milvus:
                if cached and "content_similarity_text" in cached:
                    content_similarity_text = cached["content_similarity_text"]
                    content_lean            = cached["content_lean"]
                    content_conf            = cached["content_confidence"]
                    content_hits_record     = cached["content_hits_record"]
                else:
                    mc, cn = get_milvus_client()
                    rep_vec = reps_by_folder.get(folder)
                    if rep_vec is None:
                        raise ValueError("Missing video representation")
                    res = mc.search(
                        collection_name=cn, data=[rep_vec.tolist()],
                        limit=args.content_search_limit, output_fields=MILVUS_OUTPUT_FIELDS,
                    )
                    top_hits = dedupe_content_hits_by_video(res, args.content_top_k)
                    content_similarity_text, content_lean, content_conf = build_content_similarity_context(top_hits)
                    content_hits_record = top_hits

                if cached and "narrative_pattern_text" in cached:
                    narrative_pattern_text = cached["narrative_pattern_text"]
                    narrative_lean         = cached["narrative_lean"]
                    narrative_conf         = cached["narrative_confidence"]
                    narrative_hits_record  = cached["narrative_hits_record"]
                else:
                    mc, cn = get_milvus_client()
                    embeddings_norm = F.normalize(sample_data['embeddings'].float(), p=2, dim=1)
                    query_dps_captions, ranked_top = retrieve_discourse_evidence(
                        embeddings_norm, sample_data.get('rst_links', []), captions,
                        current_video_id=folder,
                        milvus_client=mc, collection_name=cn, data_root=data_root,
                        top_k=args.discourse_top_k, search_limit=args.discourse_search_limit,
                        alpha=args.alpha,
                    )
                    narrative_pattern_text, narrative_lean, narrative_conf = build_discourse_context(
                        query_dps_captions, ranked_top,
                    )
                    narrative_hits_record = [{"video_id": vid, **cand} for vid, cand in ranked_top]

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

            llm_prompt = build_llm_prompt(video_context_text, content_similarity_text, narrative_pattern_text, evidence_mode, graph_text)
            system_prompt = build_system_prompt(evidence_mode)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": llm_prompt},
            ]
            n_tokens = count_tokens_qwen(tokenizer, messages)

            if n_tokens > MAX_CONTEXT_TOKENS:
                if graph_text is not None:
                    messages[1]["content"] = build_llm_prompt(
                        video_context_text, content_similarity_text, narrative_pattern_text,
                        evidence_mode, "Knowledge graph context truncated — prompt exceeded context window.",
                    )
                elif narrative_pattern_text is not None:
                    messages[1]["content"] = build_llm_prompt(
                        video_context_text, content_similarity_text,
                        "Discourse pattern context truncated — prompt exceeded context window.",
                        evidence_mode, graph_text,
                    )
                elif content_similarity_text is not None:
                    messages[1]["content"] = build_llm_prompt(
                        video_context_text,
                        "Content similarity context truncated — prompt exceeded context window.",
                        narrative_pattern_text, evidence_mode, graph_text,
                    )
                n_tokens = count_tokens_qwen(tokenizer, messages)

            llm_response = run_qwen_inference(tokenizer, llm_model, messages, model_type=args.model_type)
            verdict      = extract_and_parse_json(llm_response)
            pred_label   = verdict.get("predicted_label", -1)

            final_prediction = compute_final_prediction(
                evidence_mode, pred_label, content_lean, content_conf, narrative_lean, narrative_conf, graph_lean, graph_conf
            )

            ground_truth = int(
                sample_data['y'].item() if isinstance(sample_data['y'], torch.Tensor) else sample_data['y']
            )

            status = "✓" if ground_truth == pred_label else "✗"
            preview_parts = []
            if content_hits_record:
                preview_parts.append(f"content={content_lean}({content_conf})")
            if narrative_hits_record:
                preview_parts.append(f"discourse={narrative_lean}({narrative_conf})")
            if graph_hits_record:
                preview_parts.append(f"graph={graph_lean}({graph_conf})")
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
                "graph_lean":               graph_lean,
                "graph_confidence":         graph_conf,
                "final_prediction":         final_prediction,
                "token_count":              n_tokens,
                "explanation":              verdict.get("explanation", ""),
                "improvement_suggestions":  verdict.get("improvement_suggestions", []),
                "raw_llm_output":           llm_response,
                "input_scene_captions":     captions[:20],
                "content_similarity_hits":  content_hits_record,
                "narrative_evidence_hits":  narrative_hits_record,
                "graph_evidence_hits":      graph_hits_record,
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
                "graph_lean":               None,
                "graph_confidence":         None,
                "final_prediction":         -1,
                "token_count":              0,
                "explanation":              f"Pipeline error: {e}",
                "improvement_suggestions":  [],
                "raw_llm_output":           "",
                "input_scene_captions":     [],
                "content_similarity_hits":  [],
                "narrative_evidence_hits":  [],
                "graph_evidence_hits":      {},
            })

        finally:
            gc.collect()
            torch.cuda.empty_cache()

        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

    if neo4j_state["driver"] is not None:
        neo4j_state["driver"].close()

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",         type=str, required=True)
    parser.add_argument("--split_file",        type=str, required=True)
    parser.add_argument("--checkpoint_path",   type=str, required=True)
    parser.add_argument("--evidence_mode",     type=str, default="milvus", choices=EVIDENCE_MODE_CHOICES)

    # --- milvus ---
    parser.add_argument("--collection_name", type=str, default=None,
                        help="Milvus collection scene-level"
                             "Mặc định env MILVUS_COLLECTION_NAME.")
    parser.add_argument("--video_reps_dir",    type=str, default=None,
                        help="Dir chứa video_representations.pt. CHỈ cần khi evidence_mode ∈ {milvus, full}.")
    parser.add_argument("--content_top_k", type=int, default=5,
                        help="Số VIDEO khác nhau (đã dedupe) lấy làm evidence cho kênh Content Similarity.")
    parser.add_argument("--content_search_limit", type=int, default=50,
                        help="Số scene thô lấy từ Milvus trước khi dedupe theo video (nên >> content_top_k).")
    parser.add_argument("--discourse_top_k", type=int, default=5,
                        help="Số VIDEO khác nhau (đã dedupe) lấy làm evidence cho kênh Discourse Pattern.")
    parser.add_argument("--discourse_search_limit", type=int, default=30,
                        help="Số candidate lấy từ Milvus MỖI scene của video test (ANN search).")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Trọng số blend: final = alpha*dense_sim + (1-alpha)*topology_sim. Mặc định 0.7.")

    # --- graph (neo4j) ---
    parser.add_argument("--graph_top_k", type=int, default=5)
    parser.add_argument("--neo4j_uri", type=str, default=None, help="Defaults to env NEO4J_URI.")
    parser.add_argument("--neo4j_username", type=str, default=None, help="Defaults to env NEO4J_USERNAME.")
    parser.add_argument("--neo4j_password", type=str, default=None, help="Defaults to env NEO4J_PASSWORD.")
    parser.add_argument("--neo4j_database", type=str, default=None, help="Defaults to env NEO4J_DATABASE (or 'neo4j').")

    # --- model / inference ---
    parser.add_argument("--model_name",        type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model_type", type=str, default="instruct",
                        choices=["instruct", "thinking"],
                        help="Loại mô hình: 'instruct' (Qwen3-4B-Instruct) hoặc 'thinking' (Qwen3-4B-Thinking-2507)")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization using bitsandbytes to save VRAM.")
    
    # --- retrieval cache ---
    parser.add_argument("--precompute_retrieval", action="store_true",
                        help="Chỉ chạy retrieval (milvus/graph tuỳ evidence_mode), lưu ra "
                             "--retrieval_cache_path rồi thoát. KHÔNG load model LLM, chạy trên CPU để tiết kiệm chi phí GPU.")
    parser.add_argument("--retrieval_cache_path", type=str, default=None,
                        help="Đường dẫn file .json lưu/đọc kết quả retrieval theo từng video "
                             "(đặt trong folder indexing/, vd: .../indexing/retrieval_cache.json). Dùng chung cho cả milvus và graph.")

    args = parser.parse_args()
    if args.precompute_retrieval:
        run_precompute_retrieval(args)
    else:
        main(args)