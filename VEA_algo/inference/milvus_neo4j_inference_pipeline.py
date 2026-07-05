"""
milvus_neo4j_inference_pipeline.py
---------------------------------
VideoRAG Inference Pipeline — Content Library (Milvus) + Structural Pattern Retrieval (ConceptRAG/Neo4j)
+ Qwen3-4B-Instruct, với hỗ trợ ABLATION STUDY qua flag --evidence_mode.

Bốn chế độ (dùng chung 1 pipeline, cùng 1 đường code, chỉ khác việc có gọi
2 khối retrieval hay không — đảm bảo so sánh ablation "táo với táo"):

    --evidence_mode none     : chỉ dùng nội dung video test (không retrieval).
    --evidence_mode content  : + Content Similarity (Cross-Modal Self-Querying, Milvus).
    --evidence_mode concept  : + Structural Pattern Retrieval (ConceptRAG/Neo4j).
    --evidence_mode full     : cả 2 nguồn (mặc định, tương đương pipeline đầy đủ).

Usage — ví dụ chạy đủ 4 ablation:
    python milvus_neo4j_inference_pipeline.py --evidence_mode none  --checkpoint_path .../ablation_none.json  ...
    python milvus_neo4j_inference_pipeline.py --evidence_mode content --video_reps_dir .../video_representations --checkpoint_path .../ablation_content.json ...
    python milvus_neo4j_inference_pipeline.py --evidence_mode concept --concept_index_dir .../concept_rag --checkpoint_path .../ablation_concept.json ...
    python milvus_neo4j_inference_pipeline.py --evidence_mode full    --video_reps_dir .../video_representations --concept_index_dir .../concept_rag --checkpoint_path .../ablation_full.json ...

    Đủ tham số (mode = full):
        python milvus_neo4j_inference_pipeline.py \\
            --data_root         /path/to/All_Videos \\
            --split_file        /path/to/dataset_splits.json \\
            --checkpoint_path   /path/to/results_full.json \\
            --video_reps_dir    /path/to/video_representations \\
            --concept_index_dir /path/to/concept_rag \\
            --evidence_mode     full \\
            [--model_name       Qwen/Qwen3-4B-Instruct-2507] \\
            [--top_k_evidence   5] \\
            [--candidate_limit  200]

Environment variables (KHÔNG qua args):
    MILVUS_CLUSTER_ENDPOINT, MILVUS_TOKEN, MILVUS_COLLECTION_NAME (chỉ khi evidence_mode ∈ {content, full})
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE (chỉ khi evidence_mode ∈ {concept, full})

Ghi chú:
    - Đã bỏ chế độ --count_tokens_only.
    - Mỗi record trong checkpoint JSON giờ lưu thêm: evidence_mode,
      input_scene_captions, content_similarity_hits, structural_evidence_hits
      — đủ để audit lại evidence đã dùng.
    - Với evidence_mode='none', KHÔNG cần kết nối Milvus / Neo4j / --video_reps_dir /
      --concept_index_dir — pipeline chạy độc lập, nhẹ hơn.
"""

import os
import re
import gc
import time
import json
import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from pymilvus import MilvusClient
from neo4j import GraphDatabase
from dotenv import load_dotenv
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MAX_CONTEXT_TOKENS  = 32768
EVIDENCE_MODE_CHOICES = ["none", "content", "concept", "full"]


# ==========================================
# 0. RST NORMALIZATION & HELPERS
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
# 1. LABEL DEFINITIONS
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
# 2. SYSTEM PROMPT — ĐỘNG theo evidence_mode (ablation)
# ==========================================

REFERENCE_SOURCE_DOCS = {
    "content": """1. CONTENT SIMILARITY REFERENCE:
   - These are the most visually and thematically similar video segments found in the library.
   - The similarity score reflects overall content alignment — the higher the score, the more
     similar the video is in terms of topic, visual style, and scene content.
   - Pay attention to the label distribution (how many are Label 0 vs Label 1) as a prior
     about whether this type of content tends to engage viewers.""",
    "concept": """2. STRUCTURAL PATTERN REFERENCE:
   - These results come from matching the narrative structure of the input video against a library
     of training videos, using a learned ConceptRAG system that captures rhetorical patterns.
   - Each match shows a reference video that used a similar structural transition pattern
     (e.g., one type of scene followed by another type of scene, connected the same way)
     and its known engagement outcome.
   - The "Relevance" score (0–1) indicates the strength of the structural analogy.
   - When the top matches consistently point to one label, that's a strong signal.
   - "Structural diversity" measures how many distinct discourse relation types a video uses.
     Low diversity often indicates a repetitive narrative structure, more common among
     Low Engagement videos.""",
}


def build_system_prompt(evidence_mode: str) -> str:
    active_docs = []
    if evidence_mode in ("content", "full"):
        active_docs.append(REFERENCE_SOURCE_DOCS["content"])
    if evidence_mode in ("concept", "full"):
        active_docs.append(REFERENCE_SOURCE_DOCS["concept"])

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
# 3. CONTEXT BUILDER — CONTENT SIMILARITY
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
# 4. CONCEPT-RAG RETRIEVAL (Neo4j) — thay thế Dense Edge Retrieval
# ==========================================

# --- MLP Scorer (phải khớp với neo4j_concept_indexing.py) ---
class ConceptTripleScorer(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_concept_index(index_dir: Path) -> dict:
    """Load concept index artifacts (centroids, pca, concept_dict, rst_vocab, mlp, prior_scores)."""
    print(f"[INFO] Loading ConceptRAG index from {index_dir}")
    centroids = torch.load(index_dir / "centroids.pt", map_location="cpu").numpy()

    pca, centroids_pca = None, None
    pca_path = index_dir / "pca.pkl"
    if pca_path.exists():
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        c_norm        = normalize(centroids, norm='l2')
        centroids_pca = normalize(pca.transform(c_norm), norm='l2')
        print(f"  pca.pkl loaded (dim={pca.n_components_})")
    else:
        print("  [INFO] pca.pkl not found → using direct cosine on full 2048-dim")

    with open(index_dir / "concept_dict.json", 'r', encoding='utf-8') as f:
        concept_dict = {int(k): v for k, v in json.load(f).items()}
    print(f"  concept_dict.json: {len(concept_dict)} concepts")

    rst_vocab_path = index_dir / "rst_vocab.json"
    if not rst_vocab_path.exists():
        raise FileNotFoundError(f"rst_vocab.json not found in {index_dir}.")
    with open(rst_vocab_path, 'r') as f:
        rst_type_to_idx = json.load(f)

    ckpt = torch.load(index_dir / "mlp_scorer.pt", map_location="cpu")
    feature_dim = ckpt['feature_dim']
    hidden_dim  = ckpt.get('hidden_dim', 32)

    mlp = ConceptTripleScorer(feature_dim=feature_dim, hidden_dim=hidden_dim)
    mlp.load_state_dict(ckpt['model_state_dict'])
    mlp.eval()

    prior_scores_raw = ckpt.get('prior_scores', {})
    prior_scores = {}
    for k_str, v in prior_scores_raw.items():
        try:
            import ast
            parts = ast.literal_eval(k_str)
            prior_scores[(int(parts[0]), str(parts[1]), int(parts[2]))] = float(v)
        except Exception:
            pass

    print(f"  Loaded: {len(concept_dict)} concepts, {len(rst_type_to_idx)} RST types, "
          f"{len(prior_scores)} priors, mlp feature_dim={feature_dim}")

    return {
        'centroids':     centroids,
        'centroids_pca': centroids_pca,
        'pca':           pca,
        'concept_dict':  concept_dict,
        'mlp':           mlp,
        'rst_vocab':     rst_type_to_idx,
        'prior_scores':  prior_scores,
        'K':             centroids.shape[0],
    }


def assign_concepts_to_video(embeddings: torch.Tensor, centroids: np.ndarray,
                              pca=None, centroids_pca: np.ndarray = None) -> list:
    embs_np = embeddings.numpy().astype(np.float32)
    if pca is not None and centroids_pca is not None:
        embs_norm   = normalize(embs_np, norm='l2')
        embs_pca    = normalize(pca.transform(embs_norm), norm='l2')
        query_t     = torch.from_numpy(embs_pca).float()
        centroids_t = torch.from_numpy(centroids_pca).float()
    else:
        query_t     = torch.from_numpy(normalize(embs_np, norm='l2')).float()
        centroids_t = torch.from_numpy(centroids).float()
    return (query_t @ centroids_t.T).argmax(dim=1).tolist()


def extract_query_triples(scene_ids: list, concept_ids: list, rst_links: list, rst_vocab: dict) -> list:
    triples = []
    for src, tgt, rst_type in rst_links:
        src_int = int(src) - 1
        tgt_int = int(tgt) - 1
        try:
            src_idx = scene_ids.index(src_int)
            tgt_idx = scene_ids.index(tgt_int)
        except ValueError:
            continue
        rst_norm = normalize_rst_type(rst_type)
        triples.append({
            'c_src':        concept_ids[src_idx],
            'rst_type':     rst_norm,
            'c_tgt':        concept_ids[tgt_idx],
            'rst_in_vocab': rst_norm in rst_vocab,
        })
    return triples


def retrieve_concept_candidates(driver, neo4j_database: str, query_triples: list, candidate_limit: int = 200) -> list:
    valid_triples = [t for t in query_triples if t.get('rst_in_vocab', True)]
    if not valid_triples:
        return []

    triple_params = [
        {
            'c_src':      f"concept_{t['c_src']}",
            'rst_type':   t['rst_type'],
            'c_tgt':      f"concept_{t['c_tgt']}",
            'triple_idx': i,
        }
        for i, t in enumerate(valid_triples)
    ]

    cypher = """
    UNWIND $triples AS qt
    MATCH (c1:Concept {id: qt.c_src})<-[:HAS_CONCEPT]-(s1:Scene)
    MATCH (s1)-[r]->(s2:Scene)
    WHERE type(r) = qt.rst_type
    MATCH (s2)-[:HAS_CONCEPT]->(c2:Concept {id: qt.c_tgt})
    MATCH (v:Video)-[:HAS_SCENE]->(s1)
    WHERE v.is_test IS NULL AND v.video_label IS NOT NULL
    RETURN
        v.id           AS video_id,
        v.video_label  AS video_label,
        qt.c_src       AS c_src,
        qt.rst_type    AS rst_type,
        qt.c_tgt       AS c_tgt
    LIMIT $limit
    """

    candidates = []
    with driver.session(database=neo4j_database) as session:
        result = session.run(cypher, triples=triple_params, limit=candidate_limit)
        for rec in result:
            candidates.append({
                'video_id':    rec['video_id'],
                'video_label': rec['video_label'],
                'c_src':       rec['c_src'],
                'rst_type':    rec['rst_type'],
                'c_tgt':       rec['c_tgt'],
            })
    return candidates


def score_concept_candidates(candidates: list, centroids: np.ndarray, mlp: ConceptTripleScorer,
                             rst_vocab: dict, prior_scores: dict, K: int) -> list:
    if not candidates:
        return []
    n_rst = len(rst_vocab)
    feats, valid_cands = [], []

    for cand in candidates:
        try:
            c_src = int(cand['c_src'].replace("concept_", ""))
            c_tgt = int(cand['c_tgt'].replace("concept_", ""))
        except (ValueError, AttributeError):
            continue
        rst_idx = rst_vocab.get(cand['rst_type'])
        if rst_idx is None:
            continue
        cos_sim    = float(np.dot(centroids[c_src], centroids[c_tgt]))
        rst_onehot = np.zeros(n_rst, dtype=np.float32)
        rst_onehot[rst_idx] = 1.0
        prior = prior_scores.get((c_src, cand['rst_type'], c_tgt), 0.5)

        feats.append(np.concatenate([[cos_sim], rst_onehot, [prior], [c_src / K], [c_tgt / K]]))
        cand['c_src_idx'] = c_src
        cand['c_tgt_idx'] = c_tgt
        valid_cands.append(cand)

    if not feats:
        return []

    X = torch.from_numpy(np.stack(feats).astype(np.float32))
    with torch.no_grad():
        scores = mlp(X).tolist()

    for cand, score in zip(valid_cands, scores):
        cand['mlp_score'] = score
    return valid_cands


def retrieve_concept_evidence(
    sample_data: dict,
    concept_index: dict,
    neo4j_driver,
    neo4j_database: str,
    top_k: int,
    candidate_limit: int,
) -> tuple:
    """Trả về (context_text, evidence_hits_list)."""
    scene_ids_list = [int(s) for s in sample_data['scene_ids']]
    concept_ids    = assign_concepts_to_video(
        sample_data['embeddings'],
        concept_index['centroids'],
        concept_index.get('pca'),
        concept_index.get('centroids_pca'),
    )
    query_triples = extract_query_triples(
        scene_ids_list, concept_ids,
        sample_data.get('rst_links', []), concept_index['rst_vocab'],
    )
    candidates = retrieve_concept_candidates(
        neo4j_driver, neo4j_database, query_triples, candidate_limit,
    )
    scored_candidates = score_concept_candidates(
        candidates, concept_index['centroids'], concept_index['mlp'],
        concept_index['rst_vocab'], concept_index['prior_scores'], concept_index['K'],
    )
    # Build context
    context_text = build_concept_rag_context(
        concept_ids, query_triples, scored_candidates,
        concept_index['concept_dict'], top_k,
    )
    # Prepare hits for logging
    seen_videos = {}
    for c in sorted(scored_candidates, key=lambda x: x['mlp_score'], reverse=True):
        if c['video_id'] not in seen_videos:
            seen_videos[c['video_id']] = c
    top_unique = sorted(seen_videos.values(), key=lambda x: x['mlp_score'], reverse=True)[:top_k]
    hits = [{
        'video_id':    c['video_id'],
        'video_label': c['video_label'],
        'score':       c['mlp_score'],
        'rst_type':    c['rst_type'],
        'c_src':       c['c_src_idx'],
        'c_tgt':       c['c_tgt_idx'],
    } for c in top_unique]
    return context_text, hits


def build_concept_rag_context(
    concept_ids: list,
    query_triples: list,
    scored_candidates: list,
    concept_dict: dict,
    top_k: int = 5,
) -> str:
    """Build context block for structural pattern evidence (ConceptRAG)."""
    def cname(cid: int) -> str:
        return concept_dict.get(int(cid), {}).get('concept_name', f"Theme {cid}")

    lines = []
    # Diversity
    diversity = len(set(concept_ids)) / len(concept_ids) if concept_ids else 0.0
    div_note  = " (LOW — may indicate repetitive narrative structure)" if diversity < 0.3 else ""
    lines.append(
        f"Structural diversity: {len(set(concept_ids))}/{len(concept_ids)} scenes "
        f"cover distinct structural themes ({diversity:.0%}){div_note}."
    )
    lines.append("")

    # Input video transition sequence (concept-level)
    valid_t = [qt for qt in query_triples if qt.get('rst_in_vocab', True)]
    if valid_t:
        parts = []
        for qt in valid_t[:10]:
            parts.append(
                f"[{cname(qt['c_src'])}] → ({rst_to_natural(qt['rst_type'])}) → [{cname(qt['c_tgt'])}]"
            )
        lines.append("Input video transition sequence:")
        lines.append("  " + "  |  ".join(parts))
    else:
        lines.append("Input video transition sequence: none extracted.")
    lines.append("")

    # Evidence from reference library
    if not scored_candidates:
        lines.append("No matching structural patterns found in the reference library.")
        lines.append("These narrative patterns appear to be novel relative to the training data.")
        return "\n".join(lines)

    seen = {}
    for c in sorted(scored_candidates, key=lambda x: x['mlp_score'], reverse=True):
        if c['video_id'] not in seen:
            seen[c['video_id']] = c
    top_unique = sorted(seen.values(), key=lambda x: x['mlp_score'], reverse=True)[:top_k]

    lines.append(f"Top {len(top_unique)} matching structural patterns from reference library:")
    lines.append("")

    for rank, cand in enumerate(top_unique, 1):
        outcome = "HIGH ENGAGEMENT" if cand['video_label'] == 1 else "LOW ENGAGEMENT"
        src_name = cname(cand['c_src_idx'])
        tgt_name = cname(cand['c_tgt_idx'])
        lines.append(f"[{rank}] Reference pattern: [{src_name}] → ({rst_to_natural(cand['rst_type'])}) → [{tgt_name}]")
        lines.append(f"     Outcome: {outcome}  |  Relevance: {cand['mlp_score']:.4f}")
        # Thêm định nghĩa concept (tối giản)
        src_def = concept_dict.get(cand['c_src_idx'], {}).get('definition', '')
        tgt_def = concept_dict.get(cand['c_tgt_idx'], {}).get('definition', '')
        if src_def:
            lines.append(f"     {src_name}: {src_def[:120]}")
        if tgt_def and cand['c_tgt_idx'] != cand['c_src_idx']:
            lines.append(f"     {tgt_name}: {tgt_def[:120]}")
        lines.append("")

    n_eng  = sum(1 for c in top_unique if c['video_label'] == 1)
    n_neng = len(top_unique) - n_eng
    if n_eng > n_neng and n_eng >= 2:
        lines.append(f"Note: {n_eng}/{len(top_unique)} reference patterns are from High Engagement videos.")
    elif n_neng > n_eng and n_neng >= 2:
        lines.append(f"Note: {n_neng}/{len(top_unique)} reference patterns are from Low Engagement videos.")
    else:
        lines.append("Note: reference patterns are mixed — no clear lean.")

    return "\n".join(lines)


# ==========================================
# 5. USER PROMPT — ĐỘNG theo evidence_mode
# ==========================================

def build_reasoning_example(evidence_mode: str) -> str:
    bullets = []
    if evidence_mode in ("content", "full"):
        bullets.append("- Content similarity: 4/5 reference videos are Label 1 (High Engagement).")
    if evidence_mode in ("concept", "full"):
        bullets.append(
            '- Structural patterns: 3/5 reference patterns are from Low Engagement videos.\n'
            '  Top pattern: [Product Reveal] → (contrasts with) → [Casual Reaction] | LOW ENGAGEMENT | Relevance: 0.88'
        )
        bullets.append("- Structural diversity: 2/9 transitions (22%) — LOW.")
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
    structural_pattern_text: str,
    evidence_mode: str,
) -> str:
    sections = [f"=== 1. INPUT VIDEO CONTENT ===\n{video_context_text}"]
    n = 2
    if content_similarity_text is not None:
        sections.append(f"=== {n}. CONTENT SIMILARITY REFERENCE ===\n{content_similarity_text}")
        n += 1
    if structural_pattern_text is not None:
        sections.append(f"=== {n}. STRUCTURAL PATTERN REFERENCE ===\n{structural_pattern_text}")
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
# 6. CONSTANTS & HELPERS
# ==========================================

REQUIRED_DATA_KEYS = [
    'embeddings', 'scene_ids', 'metadata',
    'edge_index', 'edge_attr', 'rst_links', 'y',
]
MILVUS_OUTPUT_FIELDS = ["scene_uid", "video_id", "video_label", "caption"]


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
# 7. QWEN INFERENCE HELPERS
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
# 8. MAIN
# ==========================================

def main(args: argparse.Namespace) -> None:
    data_root       = Path(args.data_root)
    split_file      = Path(args.split_file)
    checkpoint_path = Path(args.checkpoint_path)
    evidence_mode   = args.evidence_mode

    need_content = evidence_mode in ("content", "full")
    need_concept = evidence_mode in ("concept", "full")

    milvus_client    = None
    collection_name  = None
    neo4j_driver     = None
    neo4j_database   = None
    reps_by_folder   = {}
    concept_index    = None

    # --- Content similarity: Milvus ---
    if need_content:
        milvus_endpoint = os.getenv("MILVUS_CLUSTER_ENDPOINT")
        milvus_token    = os.getenv("MILVUS_TOKEN")
        collection_name = os.getenv("MILVUS_COLLECTION_NAME")
        if not all([milvus_endpoint, milvus_token, collection_name]):
            raise ValueError("Missing Milvus env vars (required for content/full).")
        milvus_client = MilvusClient(uri=milvus_endpoint, token=milvus_token)
        if not args.video_reps_dir:
            raise ValueError("--video_reps_dir required for content/full.")
        reps_by_folder = load_video_representations(Path(args.video_reps_dir))
        print(f"[INFO] Loaded {len(reps_by_folder)} video representations.")

    # --- Structural pattern: ConceptRAG (Neo4j) ---
    if need_concept:
        neo4j_uri      = os.getenv("NEO4J_URI")
        neo4j_username = os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
        if not all([neo4j_uri, neo4j_username, neo4j_password]):
            raise ValueError("Missing Neo4j env vars (required for concept/full).")
        if not args.concept_index_dir:
            raise ValueError("--concept_index_dir required for concept/full.")
        concept_index = load_concept_index(Path(args.concept_index_dir))
        neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
        neo4j_driver.verify_connectivity()
        print("[INFO] Neo4j connection established.")

    print(f"[INFO] evidence_mode = '{evidence_mode}' "
          f"(content={'ON' if need_content else 'off'}, concept={'ON' if need_concept else 'off'})\n")

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

    # --- Main inference loop ---
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

            # STEP B: Content similarity (Milvus)
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

            # STEP C: Structural pattern (ConceptRAG)
            structural_pattern_text = None
            structural_hits_record  = []
            if need_concept:
                structural_pattern_text, structural_hits_record = retrieve_concept_evidence(
                    sample_data, concept_index, neo4j_driver, neo4j_database,
                    top_k=args.top_k_evidence, candidate_limit=args.candidate_limit,
                )

            # STEP D: Build prompt + token check
            llm_prompt = build_llm_prompt(
                video_context_text, content_similarity_text, structural_pattern_text, evidence_mode,
            )
            system_prompt = build_system_prompt(evidence_mode)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": llm_prompt},
            ]
            n_tokens = count_tokens_qwen(tokenizer, messages)

            if n_tokens > MAX_CONTEXT_TOKENS:
                if structural_pattern_text is not None:
                    messages[1]["content"] = build_llm_prompt(
                        video_context_text, content_similarity_text,
                        "Structural pattern context truncated — prompt exceeded context window.",
                        evidence_mode,
                    )
                elif content_similarity_text is not None:
                    messages[1]["content"] = build_llm_prompt(
                        video_context_text,
                        "Content similarity context truncated — prompt exceeded context window.",
                        structural_pattern_text, evidence_mode,
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
            if structural_hits_record:
                t1 = structural_hits_record[0]
                preview_parts.append(f"struct_top1(L{t1['video_label']},{t1['score']:.2f},{t1['rst_type']})")
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
                "structural_evidence_hits": structural_hits_record,
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
                "structural_evidence_hits": [],
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

    if neo4j_driver:
        try:
            neo4j_driver.close()
        except Exception:
            pass


# ==========================================
# 9. HELPER: load video representations
# ==========================================

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


# ==========================================
# 10. ENTRY POINT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VideoRAG Inference Pipeline — content (Milvus) + structural pattern (ConceptRAG/Neo4j) + Qwen, with ablation flag.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_root",         type=str, required=True)
    parser.add_argument("--split_file",        type=str, required=True)
    parser.add_argument("--checkpoint_path",   type=str, required=True)
    parser.add_argument("--evidence_mode",     type=str, default="full", choices=EVIDENCE_MODE_CHOICES,
                        help="Ablation mode: none | content | concept | full.")
    parser.add_argument("--video_reps_dir",    type=str, default=None,
                        help="Dir with video_representations.pt (from milvus_compute_video_query.py). "
                             "Required when evidence_mode ∈ {content, full}.")
    parser.add_argument("--concept_index_dir", type=str, default=None,
                        help="Dir with ConceptRAG artifacts (centroids.pt, pca.pkl, concept_dict.json, "
                             "rst_vocab.json, mlp_scorer.pt) from neo4j_concept_indexing.py. "
                             "Required when evidence_mode ∈ {concept, full}.")
    parser.add_argument("--model_name",        type=str, default=DEFAULT_MODEL_NAME,
                        help=f"Qwen model ID. Default: {DEFAULT_MODEL_NAME}")
    parser.add_argument("--top_k_evidence",    type=int, default=5,
                        help="Top-K unique reference patterns in structural evidence.")
    parser.add_argument("--candidate_limit",   type=int, default=200,
                        help="Max candidates lấy từ Neo4j cho mỗi query triple (ConceptRAG).")
    args = parser.parse_args()
    main(args)