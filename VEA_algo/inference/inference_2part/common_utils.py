"""
common_utils.py
--------------------------------------------------------------------------------------------------
Các hàm dùng chung cho cả 2 quy trình:
    - stage1_retrieval.py  : thực hiện truy vấn Milvus/Neo4j, KHÔNG load LLM.
    - stage2_inference.py  : build lại prompt từ evidence đã lưu, load LLM, sinh dự đoán.

File này KHÔNG import pymilvus / neo4j / transformers — chỉ chứa logic thuần (string building,
RST parsing, ensemble voting, ...) để cả 2 stage đều dùng lại được mà không phải cài thêm
dependency không cần thiết.
"""

import json
import re
import collections
from pathlib import Path
from collections import Counter

import torch


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
# 3. ROLLBACK SYSTEM PROMPT CONSTANTS
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

GRAPH_SOURCE_DOC = """3. KNOWLEDGE GRAPH EVIDENCE (from Neo4j):
   Three sources from training corpus:

   a) **RST Structural Matches** — videos with same RST discourse chain. Shows label, max nodes, total paths, RST chain,
      and Video RST Summary (total relations, dominant type, distribution). Use to compare narrative structure.

   b) **Similarity-based Neighbors** — videos with semantically similar scenes (cosine similarity).
      Shows label and avg similarity score. Indicates thematic resemblance.

   c) **Concept Details** — semantic clusters from matched scenes. Shows label distribution (prior probability), Audio/Visual Style, and Keywords.
      Strong distribution (>70% one label) is a powerful signal.

   Use strong consensus across sources as high-confidence evidence. Video RST Summary connects structure to content.
"""

CONFLICT_RESOLUTION_NOTE = (
    "\nYou must give significantly more weight and priority to the CONTENT SIMILARITY REFERENCE "
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
    intro_parts = []

    if evidence_mode in ("milvus", "full"):
        active_docs.append(REFERENCE_SOURCE_DOCS["content"])
        active_docs.append(REFERENCE_SOURCE_DOCS["edge"])
        intro_parts.append("similar contexts retrieved from a reference library")

    if evidence_mode in ("graph", "full"):
        active_docs.append(GRAPH_SOURCE_DOC)
        intro_parts.append("structural evidence from a Knowledge Graph")

    if active_docs:
        intro = " cross-reference it with " + " and ".join(intro_parts) + ","
        reference_block = "How to use the reference source(s) below:\n\n" + "\n\n".join(active_docs)
        if evidence_mode == "full" or evidence_mode == "milvus":
            reference_block += "\n" + CONFLICT_RESOLUTION_NOTE

        if evidence_mode == "graph":
            reference_block += (
                "\nWhen using the Knowledge Graph evidence, prioritize Concept Details first "
                "(they give you a prior distribution of engagement labels from semantically similar clusters), "
                "then RST structural matches (to confirm narrative logic), and finally similarity-based neighbors "
                "(as supplementary thematic cues). "
                "Use the Video RST Summary to understand the overall discourse structure of matched videos, "
                "and compare it with the input video's own structure (Section 1) to assess structural similarity."
            )
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
# 4. EVIDENCE LEAN + CONFIDENCE
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
# 5. CONTEXT BUILDERS & HELPERS (MILVUS ROLLBACK)
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
    embeddings_norm, rst_links: list, captions: list,
    current_video_id: str, milvus_client, collection_name: str, data_root: Path,
    top_k: int, search_limit: int, alpha: float,
) -> tuple:
    """
    Thực hiện truy xuất riêng biệt theo từng scene-level embedding.
    NOTE: gọi milvus_client.search(...) -> chỉ dùng ở stage1_retrieval.py.
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
                try:
                    matched_scene_id = int(s_uid.rsplit("_", 1)[-1])
                    matched_idx = cand_data["scene_ids"].index(matched_scene_id)
                    matched_caption = cand_data["dps_captions"][matched_idx]
                except Exception:
                    matched_caption = h["entity"].get("caption", "")

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
# 7. NEO4J KNOWLEDGE GRAPH STRUCTURAL MATCHING
#    (các hàm nhận sẵn `tx`/kết quả -> chỉ gọi thực sự ở stage1_retrieval.py)
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


def format_concept_distribution(label_distribution_json: str) -> str:
    if not label_distribution_json:
        return "No distribution data"
    try:
        dist = json.loads(label_distribution_json)
        return ", ".join(f"L{k}={v}" for k, v in sorted(dist.items(), key=lambda x: int(x[0])))
    except Exception:
        return label_distribution_json

def get_dominant_label_and_count(label_distribution_json: str):
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
    except:
        return None, 0, 0


def build_graph_context_v2(evidence: dict) -> tuple:
    """
    evidence: dict trả về từ get_graph_evidence (đã convert sang list/dict thuần Python).
    Trả về (context_text, lean, confidence)
    """
    lines = []
    label_counts = Counter()

    structural = evidence.get("structural_matches", [])
    video_details = evidence.get("video_details", {})

    # ====== RST Structural Matches ======
    if structural:
        lines.append("** RST Structural Matches** (exact chain matches):")
        for idx, rec in enumerate(structural[:5], 1):
            vid = rec.get("video_id")
            label = rec.get("label")
            if label is not None:
                label_counts[label] += 1
            max_nodes = rec.get("max_nodes_matched", 0)
            total_paths = rec.get("total_matched_sequences", 0)

            match_details = rec.get("match_details", [])
            rst_chain = match_details[0].get("relation_chain", []) if match_details else []
            # rst_chain_str = " -> ".join(rst_chain) if rst_chain else "N/A"
            rst_chain_str = explain_rst_chain(rst_chain) if rst_chain else "N/A"

            vinfo = video_details.get(vid, {})
            rst_summary = vinfo.get("rst_summary", "")

            lines.append(f"  [{idx}] Match Video {idx} | Label: {label}  | Max nodes: {max_nodes}, Total paths: {total_paths}")
            lines.append(f"      RST Chain: [{rst_chain_str}]")
            if rst_summary:
                lines.append(f"      Video RST Summary: {rst_summary[:300]}...")
            lines.append("")
        lines.append("")
    else:
        lines.append("** RST Structural Matches**: None found.")
        lines.append("")

    # ====== Similarity-based Neighbors ======
    sim_videos = evidence.get("similarity_videos", [])
    if sim_videos:
        lines.append("** Similarity-based Neighbors** (videos with similar scenes):")
        for idx, rec in enumerate(sim_videos[:5], 1):
            label = rec.get("label")
            avg_score = rec.get("avg_score")

            if label is not None:
                label_counts[label] += 1
            lines.append(f"  [{idx}] Video Neighbor '{idx}' (Label: {label}) — Avg similarity {avg_score:.3f}")
            lines.append("")
        lines.append("")
    else:
        lines.append("** Similarity-based Neighbors**: None found.")
        lines.append("")

    # ====== Concept Details ======
    concept_ids_set = evidence.get("concept_ids_set", set())
    concept_details = evidence.get("concept_details", {})

    if concept_ids_set:
        concept_list = list(concept_ids_set)
        def sort_key(cid):
            cdetail = concept_details.get(cid, {})
            label_dist = cdetail.get("label_distribution", "{}")
            try:
                dist = json.loads(label_dist) if isinstance(label_dist, str) else label_dist
                return len(dist)   # số lượng label khác nhau
            except:
                return 0
        concept_list.sort(key=sort_key, reverse=True)

        lines.append("** Concept Details** (from matched scenes):")
        # for cid in sorted(concept_ids_set)[:5]:
        for cid in concept_list[:5]:
            cdetail = concept_details.get(cid, {})
            if not cdetail:
                continue
            label_dist = cdetail.get("label_distribution", "N/A")
            audio_style = cdetail.get("audio_style", "")
            visual_style = cdetail.get("visual_style", "")
            keywords = cdetail.get("keywords", [])

            lines.append(f"  [Concept] {cid}")
            # lines.append(f"      Label distribution: {label_dist}")
            dom_label, dom_count, total = get_dominant_label_and_count(label_dist)
            if dom_label is not None:
                # lines.append(f"      Dominant label: {dom_label} ({dom_count}/{total})")
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

    # ====== Tổng hợp label distribution ======
    n_total = sum(label_counts.values())
    if n_total == 0:
        return "\n".join(lines), None, "none"
    n_pos = label_counts.get(1, 0)
    lean, confidence = evidence_lean_and_confidence(n_pos, n_total)
    label_summary = ", ".join(f"Label {lbl}: {cnt}" for lbl, cnt in sorted(label_counts.items()))
    lines.append(f"** Overall label distribution across all graph evidence: {label_summary}  [agreement: {confidence}]")

    return "\n".join(lines), lean, confidence


def build_graph_text_from_hits(graph_hits_record: dict) -> tuple:
    """
    Tái tạo graph_text từ graph_hits_record (không truy vấn Neo4j).
    Trả về (graph_text, lean, confidence).
    """
    lines = []
    label_counts = Counter()

    structural = graph_hits_record.get("structural_matches", [])
    similarity = graph_hits_record.get("similarity_videos", [])
    concept_details = graph_hits_record.get("concept_details", {})

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
            lines.append(f"  [{idx}] Video Neighbor '{idx}' (Label: {label}) — Avg similarity {avg_score:.3f}")
            lines.append("")
        lines.append("")
    else:
        lines.append("** Similarity-based Neighbors**: None found.")
        lines.append("")

    # ----- Concept Details (sắp xếp theo số lượng nhãn khác nhau) -----
    if concept_details:
        # Sắp xếp concept theo số lượng label khác nhau (ưu tiên nhiều hơn)
        concept_list = list(concept_details.keys())
        def sort_key(cid):
            cdetail = concept_details.get(cid, {})
            label_dist = cdetail.get("label_distribution", "{}")
            try:
                dist = json.loads(label_dist) if isinstance(label_dist, str) else label_dist
                return len(dist)   # số lượng label khác nhau
            except:
                return 0
        concept_list.sort(key=sort_key, reverse=True)

        lines.append("** Concept Details** (from matched scenes):")
        for cid in concept_list[:5]:   # chỉ lấy 5 concept đầu sau khi sắp xếp
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
                # lines.append(f"      Domain Label: {dom_label} (dominant count: {dom_count}/{total})")
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


# ==========================================
# 8. LLM PROMPT BUILDERS
#    (thuần string building -> dùng ở CẢ 2 stage, nhưng chỉ thực sự gọi LLM ở stage2)
# ==========================================

def build_reasoning_example(evidence_mode: str) -> str:
    bullets = []
    if evidence_mode in ("milvus", "full"):
        bullets.append("- Content similarity: label distribution across 5 distinct reference videos is 4×Label 1, 1×Label 0. [agreement: strong]")
        bullets.append(
            '- Discourse pattern: label distribution across 5 distinct reference videos is 3×Label 0, 2×Label 1. [agreement: weak]\n'
            '  Top match: Blended score 0.81 (dense=0.85, topology=0.70) | LOW ENGAGEMENT\n'
            '  "A close-up product reveal [Discourse: elaborates on Scene 2, eventually leading to the video\'s core scene]"'
        )
    if evidence_mode in ("graph", "full"):
        bullets.append(
            "- Knowledge Graph: RST structural matches show 5 videos, all Label 1, with chain ROOT -> TEMPORAL -> TEMPORAL. "
            "Concept Details show Concept_G_C127 with label distribution {'1': 3}, suggesting high engagement. "
            "Similarity neighbors are also mostly Label 1."
        )
    if not bullets:
        bullets.append("- No reference library is used in this setting — judged purely from the video's own content and structure.")

    bullet_text = "\n".join(bullets)
    return f"""---
[Example]:
{bullet_text}

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
        return f"{primary_filter}\n2. **FINAL DECISION**: Synthesize your observations across all four core dimensions with equal probability. Base your final label and plain-language explanation strictly on this objective structural analysis."

    result = (
        f"{primary_filter}\n"
        "2. **REFERENCE CROSS-EXAMINATION**: Examine the provided reference examples as contextual anchors. "
        "Look for similarities in structural dynamics, theme progression, or reaction patterns to help calibrate your judgment, "
        "especially for borderline cases.\n"
        "3. **INTEGRATED JUDGMENT**: Combine your independent content analysis with the evidence from the references. "
        "If the reference library shows a strong consensus (agreement: strong), give that structural signal significant weight in your final prediction."
    )

    if evidence_mode in ("graph", "full"):
        graph_guidance = (
            "\n4. **GRAPH EVIDENCE INTERPRETATION**: When Knowledge Graph evidence is available:\n"
            "   - **Concept Details**: First, inspect the label distribution and semantic details (Audio/Visual Style, Keywords) "
            "of Concepts found in the matched RST paths. A strong label distribution (e.g., >70% Label 1) that aligns "
            "with the input video's content is a powerful signal.\n"
            "   - **RST Structural Matches**: Evaluate how many matched videos share the same RST chain and their labels. "
            "A majority of Label 1 among matched videos supports High Engagement.\n"
            "   - **Video RST Summary**: Use this to understand the overall discourse structure of the matched videos. "
            "Compare it with your own analysis of the input video's structure (based on the captions) to see if they align.\n"
            "   - **Similarity-based Neighbors**: Consider videos with high average similarity. If they are mostly Label 1, "
            "it reinforces the prediction; if mixed, treat with caution.\n"
            "Use strong consensus across multiple sources as high-confidence evidence."
        )
        result += graph_guidance

    return result


def build_llm_prompt(video_context_text, evidence_mode, content_text=None, discourse_text=None, graph_text=None) -> str:
    sections = [f"=== 1. INPUT VIDEO CONTENT ===\n{video_context_text}"]
    n = 2
    if content_text is not None:
        sections.append(f"=== {n}. CONTENT SIMILARITY REFERENCE ===\n{content_text}")
        n += 1
    if discourse_text is not None:
        sections.append(f"=== {n}. DISCOURSE PATTERN REFERENCE ===\n{discourse_text}")
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
  "improvement_suggestions": [
    "Actionable suggestion 1.",
    "Actionable suggestion 2."
  ]
}}""")
    return "\n\n".join(sections)


# ==========================================
# 9. ENSEMBLE VOTING STRATEGIES
# ==========================================

def compute_ensemble_label_two_signal(llm_pred, ref_lean, ref_conf):
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


def compute_ensemble_label_hybrid(llm_pred, content_lean, content_conf, narrative_lean, narrative_conf):
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
        return llm_pred if llm_pred in (0, 1) else content_lean
    return top[0][0]


def compute_ensemble_label_full(llm_pred, content_lean, content_conf, narrative_lean, narrative_conf, graph_lean, graph_conf):
    votes = [v for v in [llm_pred, content_lean, narrative_lean, graph_lean] if v in (0, 1)]
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
        if graph_lean in (0, 1) and graph_conf == "strong":
            strong_candidates.append(graph_lean)
        if strong_candidates:
            return strong_candidates[0]
        return llm_pred if llm_pred in (0, 1) else content_lean
    return top[0][0]


# ==========================================
# 10. CONSTANTS & VALIDATION
# ==========================================

REQUIRED_DATA_KEYS = [
    'embeddings', 'scene_ids', 'metadata',
    'edge_index', 'edge_attr', 'rst_links', 'y',
]


# ==========================================
# 11. GENERAL HELPERS
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
                if idx >= len(scene_ids_list): break
                captions_dict[int(scene_ids_list[idx])] = seg.get('caption', '')
        elif isinstance(segments, dict):
            for k, v in segments.items():
                captions_dict[int(k)] = v.get('caption', str(v)) if isinstance(v, dict) else str(v)
    except Exception:
        pass

    n_scenes = len(scene_ids_list)
    raw_caption_list = [captions_dict.get(int(sid), "No caption available.") for sid in scene_ids_list]

    if mode in ["graph", "none"]:
        display_captions = raw_caption_list
    else:
        display_captions = serialize_discourse_captions(raw_caption_list, data.get('rst_links', []), n_scenes)

    lines = [f"Total scenes: {n_scenes}", ""]
    for idx, scene_id in enumerate(scene_ids_list[:20]):
        cap = display_captions[idx] if idx < len(display_captions) else "No caption available."
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

        if not emb_path.exists() or not seg_path.exists():
            continue
        if require_reps and folder_name not in reps_by_folder:
            continue

        try:
            data = torch.load(emb_path, map_location='cpu')
            if any(k not in data for k in REQUIRED_DATA_KEYS): continue
            valid_folders.append(folder_name)
            valid_data[folder_name] = data
        except Exception:
            pass

    return valid_folders, valid_data, invalid_folders


# ==========================================
# 12. LLM RESPONSE PARSING (thuần logic, không cần model -> dùng ở stage2)
# ==========================================

def extract_and_parse_json(raw_text: str) -> dict:
    if not raw_text:
        return {"predicted_label": -1, "explanation": "Empty LLM response"}
    clean = re.sub(r'^```json\s*|^```\s*|\s*```$', '', raw_text.strip(), flags=re.IGNORECASE).strip()
    try:
        match  = re.search(r'\{.*\}', clean, re.DOTALL)
        parsed = json.loads(match.group(0) if match else clean)
        pred   = parsed.get("predicted_label")
        if pred is not None:
            parsed["predicted_label"] = int(str(pred).strip())
            return parsed
    except Exception:
        pass
    return {"predicted_label": -1, "explanation": "Parsing failed"}
