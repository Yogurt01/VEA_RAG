"""
neo4j_retrieve_evidence.py
--------------------
Phase 2 — Online retrieval (chạy cho từng video test) cho ConceptRAG
phiên bản sử dụng GMM + soft assignment (neo4j_concept_indexing.py).

Pipeline:
    Step A : Load centroids.pt + pca.pkl + concept_dict.json + rst_vocab.json + mlp_scorer.pt
    Step B : Assign hard concept_id cho từng scene của video test (cosine, dùng PCA nếu có)
    Step C : Extract query triples (c_src, RST_type, c_tgt) có hướng
    Step D : Cypher query Neo4j — tìm candidate triples khớp từng query triple
    Step E : Build feature vector + MLP score song song
    Step F : In top-K evidence theo format tự nhiên (không jargon kỹ thuật)

Lưu ý: Phiên bản này sử dụng hard concept assignment cho truy vấn (vì Neo4j lưu
HAS_CONCEPT với 1 concept/scene). Soft assignment (TOP-P) được dùng ở Phase 1
để tăng triple coverage, nhưng retrieval vẫn dùng hard concept để đảm bảo
truy vấn Neo4j nhanh và chính xác.

Usage — single video:
    neo4j_retrieve_evidence.py \
        --index_dir       /path/to/concept_index \
        --data_root       /path/to/All_Videos \
        --video_id        b0f9d31bde3573c94ba3580a36af6b70 \
        [--top_k          5] \
        [--candidate_limit 200]

Usage — batch (đọc từ split file, lấy ngẫu nhiên N video test):
    neo4j_retrieve_evidence.py \
        --index_dir  /path/to/concept_index \
        --data_root  /path/to/All_Videos \
        --split_file /path/to/dataset_splits.json \
        --n_samples  10 \
        [--top_k     5]

Environment variables (Neo4j):
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE
"""

import os
import json
import pickle
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.preprocessing import normalize
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


# ==========================================
# 0. RST TYPE NORMALIZATION
# ==========================================
def normalize_rst_type(rst_type: str) -> str:
    return str(rst_type).strip().upper().replace(" ", "_").replace("-", "_")


# ==========================================
# 1. MLP SCORER (định nghĩa lại để load checkpoint)
# ==========================================

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


# ==========================================
# 2. LOAD CONCEPT INDEX
# ==========================================

def load_concept_index(index_dir: Path) -> dict:
    """
    Load tất cả artifacts từ Phase 1 (neo4j_concept_indexing.py):
        centroids     : np.ndarray (K, 2048)
        centroids_pca : np.ndarray (K, pca_dim) hoặc None
        pca           : fitted PCA object hoặc None
        concept_dict  : dict {int(concept_id): {concept_name, definition}}
        mlp           : ConceptTripleScorer (eval mode)
        rst_vocab     : dict {rst_type: idx} — CỐ ĐỊNH, load trực tiếp từ
                        rst_vocab.json (không lấy lại từ mlp checkpoint, vì
                        checkpoint có thể được train từ một vocab cũ hơn nếu
                        EDGE_TYPES thay đổi giữa các lần build).
        prior_scores  : dict {(c_src, rst_type, c_tgt): float}
    """
    print("[INFO] Loading concept index...")

    # Centroids
    centroids = torch.load(index_dir / "centroids.pt", map_location="cpu").numpy()

    # PCA (tuỳ chọn — nếu build_concept_index.py chạy với PCA)
    pca = None
    centroids_pca = None
    pca_path = index_dir / "pca.pkl"
    if pca_path.exists():
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        centroids_norm = normalize(centroids, norm='l2')
        centroids_pca  = pca.transform(centroids_norm)
        centroids_pca  = normalize(centroids_pca, norm='l2')
        print(f"  pca.pkl loaded (dim={pca.n_components_})")
        print(f"  centroids_pca shape: {centroids_pca.shape}")
    else:
        print("  [INFO] pca.pkl not found → using direct cosine on full 2048-dim")

    # Concept dictionary
    with open(index_dir / "concept_dict.json", 'r', encoding='utf-8') as f:
        concept_dict_raw = json.load(f)
    concept_dict = {int(k): v for k, v in concept_dict_raw.items()}
    print(f"  concept_dict.json: {len(concept_dict)} concepts")

    # ------------------------------------------------------------------
    # RST vocab — load TRỰC TIẾP từ rst_vocab.json (nguồn sự thật duy
    # nhất, cố định theo EDGE_TYPES). KHÔNG dùng rst_type_to_idx bên
    # trong mlp checkpoint để tránh lệch vocab nếu build lại MLP mà
    # quên đồng bộ vocab, hoặc khi rst_vocab.json được cập nhật riêng.
    # ------------------------------------------------------------------
    rst_vocab_path = index_dir / "rst_vocab.json"
    if not rst_vocab_path.exists():
        raise FileNotFoundError(
            f"rst_vocab.json not found in {index_dir}. "
            f"Run neo4j_concept_indexing.py first to generate the fixed RST vocabulary."
        )
    with open(rst_vocab_path, 'r') as f:
        rst_type_to_idx = json.load(f)
    print(f"  rst_vocab.json: {len(rst_type_to_idx)} RST types (fixed vocab)")

    # MLP checkpoint
    ckpt        = torch.load(index_dir / "mlp_scorer.pt", map_location="cpu")
    feature_dim = ckpt['feature_dim']
    n_rst_types = ckpt['n_rst_types']
    hidden_dim  = ckpt.get('hidden_dim', 32)

    # Sanity check: vocab size trong checkpoint phải khớp rst_vocab.json hiện tại.
    # Nếu lệch, nghĩa là MLP được train với một vocab khác — one-hot sẽ sai chiều
    # hoặc sai ý nghĩa nếu cứ load đè lên nhau.
    if n_rst_types != len(rst_type_to_idx):
        raise ValueError(
            f"RST vocab size mismatch: mlp_scorer.pt was trained with "
            f"n_rst_types={n_rst_types}, but rst_vocab.json currently has "
            f"{len(rst_type_to_idx)} types. Re-run neo4j_concept_indexing.py "
            f"to retrain the MLP with the current EDGE_TYPES vocabulary."
        )

    mlp = ConceptTripleScorer(feature_dim=feature_dim, hidden_dim=hidden_dim)
    mlp.load_state_dict(ckpt['model_state_dict'])
    mlp.eval()
    print(f"  mlp_scorer.pt loaded (feature_dim={feature_dim}, hidden_dim={hidden_dim})")

    # Prior scores
    prior_scores_raw = ckpt.get('prior_scores', {})
    prior_scores = {}
    for k_str, v in prior_scores_raw.items():
        try:
            import ast
            parts = ast.literal_eval(k_str)
            prior_scores[(int(parts[0]), str(parts[1]), int(parts[2]))] = float(v)
        except Exception:
            pass
    print(f"  prior_scores: {len(prior_scores)} entries")

    return {
        'centroids':      centroids,
        'centroids_pca':  centroids_pca,
        'pca':            pca,
        'concept_dict':   concept_dict,
        'mlp':            mlp,
        'rst_vocab':      rst_type_to_idx,
        'prior_scores':   prior_scores,
        'n_rst_types':    n_rst_types,
    }


# ==========================================
# 3. LOAD VIDEO TEST DATA
# ==========================================

def load_video_test(video_id: str, data_root: Path):
    """Load scene_embeddings.pt + segments.json của một video test."""
    video_dir = data_root / video_id
    emb_path  = video_dir / "scene_embeddings.pt"
    seg_path  = video_dir / "segments.json"

    if not emb_path.exists():
        print(f"  [ERROR] scene_embeddings.pt not found for {video_id}")
        return None
    if not seg_path.exists():
        print(f"  [ERROR] segments.json not found for {video_id}")
        return None

    data = torch.load(emb_path, map_location="cpu")

    with open(seg_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    scene_ids = [int(s) for s in data['scene_ids']]
    captions  = []
    if isinstance(segments, list):
        for idx in range(len(scene_ids)):
            cap = segments[idx].get('caption', '') if idx < len(segments) else ''
            captions.append(cap)
    elif isinstance(segments, dict):
        for sid in scene_ids:
            entry = segments.get(str(sid)) or segments.get(sid, {})
            cap   = entry.get('caption', '') if isinstance(entry, dict) else str(entry)
            captions.append(cap)

    label = None
    if 'y' in data:
        label = int(data['y'].item() if isinstance(data['y'], torch.Tensor) else data['y'])

    return {
        'video_id':    video_id,
        'video_label': label,
        'embeddings':  data['embeddings'],    # Tensor (T, 2048)
        'scene_ids':   scene_ids,
        'rst_links':   data.get('rst_links', []),
        'captions':    captions,
    }


# ==========================================
# 4. STEP B: ASSIGN CONCEPT IDs TO VIDEO TEST
# ==========================================

def assign_concepts_to_video(
    video: dict,
    centroids: np.ndarray,
    pca=None,
    centroids_pca: np.ndarray = None,
) -> dict:
    """
    Assign concept_id cho từng scene của video test.
    - Nếu có PCA: transform scene embeddings → PCA space → cosine với centroids_pca
    - Nếu không:  cosine trực tiếp với centroids full-dim
    Hai không gian so sánh PHẢI nhất quán với nhau (cùng được transform
    hoặc cùng không transform), nếu không kết quả argmax sẽ vô nghĩa.
    """
    embs_np = video['embeddings'].numpy().astype(np.float32)   # (T, 2048)

    if pca is not None and centroids_pca is not None:
        embs_norm = normalize(embs_np, norm='l2')
        embs_pca  = pca.transform(embs_norm)
        embs_pca  = normalize(embs_pca, norm='l2')

        query_t     = torch.from_numpy(embs_pca).float()         # (T, pca_dim)
        centroids_t = torch.from_numpy(centroids_pca).float()    # (K, pca_dim)
    else:
        embs_norm   = normalize(embs_np, norm='l2')
        query_t     = torch.from_numpy(embs_norm).float()        # (T, 2048)
        centroids_t = torch.from_numpy(centroids).float()        # (K, 2048)

    scores = query_t @ centroids_t.T                             # (T, K)
    c_ids  = scores.argmax(dim=1).tolist()

    video['concept_ids'] = c_ids
    return video


# ==========================================
# 5. STEP C: EXTRACT QUERY TRIPLES (có hướng)
# ==========================================

def extract_query_triples(video: dict, rst_vocab: dict) -> list:
    """
    Tạo danh sách query triples từ RST links của video test.
    Giữ đúng hướng (src → tgt).

    rst_links dùng 1-based index trong file .pt → trừ 1 để khớp với scene_ids,
    nhất quán với neo4j_concept_indexing.py và upload_neo4j.py.

    Nếu RST type của test KHÔNG có trong rst_vocab (cố định, từ EDGE_TYPES):
        - Đây là tình huống RST type hoàn toàn lạ, ngoài cả 20 loại chuẩn.
        - Triple vẫn được giữ lại (không loại bỏ) vì video test vẫn cần được
          phân tích đầy đủ, nhưng được đánh dấu 'rst_in_vocab': False để
          các bước sau (Cypher query, MLP scoring) biết mà xử lý phù hợp
          (Cypher sẽ tự nhiên không tìm thấy match, MLP sẽ không được gọi
          cho riêng triple này).
    """
    c_ids     = video['concept_ids']
    scene_ids = video['scene_ids']
    triples   = []
    unknown_rst = set()

    for src, tgt, rst_type in video['rst_links']:
        src_int = int(src) - 1
        tgt_int = int(tgt) - 1
        try:
            src_idx = scene_ids.index(src_int)
            tgt_idx = scene_ids.index(tgt_int)
        except ValueError:
            continue

        rst_norm    = normalize_rst_type(rst_type)
        in_vocab    = rst_norm in rst_vocab
        if not in_vocab:
            unknown_rst.add(rst_norm)

        triples.append({
            'c_src':        c_ids[src_idx],
            'rst_type':     rst_norm,
            'c_tgt':        c_ids[tgt_idx],
            'src_scene':    src_int,
            'tgt_scene':    tgt_int,
            'src_caption':  video['captions'][src_idx] if src_idx < len(video['captions']) else '',
            'tgt_caption':  video['captions'][tgt_idx] if tgt_idx < len(video['captions']) else '',
            'rst_in_vocab': in_vocab,
        })

    if unknown_rst:
        print(f"  [WARNING] Video has {len(unknown_rst)} RST type(s) outside the fixed "
              f"vocabulary (no training evidence possible for these): {sorted(unknown_rst)}")

    return triples


# ==========================================
# 6. STEP D: NEO4J CANDIDATE RETRIEVAL
# ==========================================

def retrieve_candidates_from_neo4j(
    driver, query_triples: list, candidate_limit: int = 200,
) -> list:
    """
    Với mỗi query triple (c_src, RST_type, c_tgt) có rst_in_vocab=True,
    tìm các video training có cùng pattern trong Neo4j — đúng hướng,
    đúng RST type.

    Triples có rst_in_vocab=False bị loại khỏi truy vấn ngay từ đây —
    không có ý nghĩa gì khi tìm match cho một RST type chưa từng được
    train, và tránh lãng phí một Neo4j round-trip chắc chắn trả về rỗng.
    """
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
    WHERE v.is_test IS NULL
      AND (v.video_label IS NOT NULL OR v.predicted_label IS NOT NULL)
    RETURN
        qt.triple_idx       AS triple_idx,
        v.id                AS video_id,
        coalesce(v.video_label, v.predicted_label) AS video_label,
        qt.c_src            AS c_src,
        qt.rst_type         AS rst_type,
        qt.c_tgt            AS c_tgt,
        coalesce(s1.caption, "") AS src_caption,
        coalesce(s2.caption, "") AS tgt_caption
    LIMIT $limit
    """

    candidates = []
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(cypher, triples=triple_params, limit=candidate_limit)
        for rec in result:
            candidates.append({
                'triple_idx':  rec['triple_idx'],
                'video_id':    rec['video_id'],
                'video_label': rec['video_label'],
                'c_src':       rec['c_src'],
                'rst_type':    rec['rst_type'],
                'c_tgt':       rec['c_tgt'],
                'src_caption': rec['src_caption'],
                'tgt_caption': rec['tgt_caption'],
            })

    return candidates


# ==========================================
# 7. STEP E: MLP SCORING (parallel)
# ==========================================

def score_candidates(
    candidates: list, centroids: np.ndarray, mlp: ConceptTripleScorer,
    rst_vocab: dict, prior_scores: dict,
) -> list:
    """
    Build feature vectors + score tất cả candidates song song bằng MLP.
    Mọi candidate ở đây đều có rst_type nằm trong rst_vocab (đã lọc ở
    Step D), nên rst_vocab.get(...) luôn trả về index hợp lệ — không
    còn fallback âm thầm về 0 như trước khi cố định vocab.
    """
    if not candidates:
        return []
    K = centroids.shape[0]
    n_rst = len(rst_vocab)
    feats = []
    for cand in candidates:
        try:
            c_src_idx = int(cand['c_src'].replace("concept_", ""))
            c_tgt_idx = int(cand['c_tgt'].replace("concept_", ""))
        except ValueError:
            c_src_idx, c_tgt_idx = 0, 0

        cos_sim = float(np.dot(centroids[c_src_idx], centroids[c_tgt_idx]))

        rst_onehot = np.zeros(n_rst, dtype=np.float32)
        rst_idx = rst_vocab.get(cand['rst_type'])
        if rst_idx is None:
            continue
        rst_onehot[rst_idx] = 1.0

        prior = prior_scores.get((c_src_idx, cand['rst_type'], c_tgt_idx), 0.5)

        feat = np.concatenate([[cos_sim], rst_onehot, [prior], [c_src_idx / K], [c_tgt_idx / K]])
        feats.append((cand, feat))

    if not feats:
        return []
    X = torch.from_numpy(np.stack([f for _, f in feats]).astype(np.float32))
    with torch.no_grad():
        scores = mlp(X).tolist()
    scored = []
    for (cand, _), score in zip(feats, scores):
        cand['mlp_score'] = score
        scored.append(cand)
    return scored


# ==========================================
# 8. STEP F: FORMAT & PRINT TOP-K EVIDENCE
# ==========================================

def format_evidence_output(
    video: dict, query_triples: list, scored_candidates: list,
    concept_dict: dict, top_k: int,
) -> tuple[str, dict]:
    """
    Format kết quả top-K evidence theo ngôn ngữ tự nhiên.
    Trả về (formatted_string, metrics_dict)
    """
    sorted_cands = sorted(scored_candidates, key=lambda x: x['mlp_score'], reverse=True)

    seen_videos = {}
    for cand in sorted_cands:
        vid = cand['video_id']
        if vid not in seen_videos:
            seen_videos[vid] = cand

    top_unique = sorted(seen_videos.values(), key=lambda x: x['mlp_score'], reverse=True)[:top_k]
    
    ground_truth = video['video_label']
    engaging_count = sum(1 for c in top_unique if c['video_label'] == 1)
    not_engaging_count = sum(1 for c in top_unique if c['video_label'] == 0)
    if engaging_count > not_engaging_count:
        leans = 'engaging'
    elif not_engaging_count > engaging_count:
        leans = 'not_engaging'
    else:
        leans = 'mixed'

    raw_candidates = len(scored_candidates)
    unique_count = len(top_unique)
    avg_score = sum(c['mlp_score'] for c in top_unique) / unique_count if unique_count else 0.0

    metrics = {
        'video_id': video['video_id'],
        'ground_truth': ground_truth,
        'engaging_count': engaging_count,
        'not_engaging_count': not_engaging_count,
        'leans': leans,
        'raw_candidates': raw_candidates,
        'unique_top_k': unique_count,
        'avg_score': avg_score,
    }

    lines = []
    lines.append("=" * 70)
    lines.append(f" VIDEO TEST: {video['video_id']}")
    if video['video_label'] is not None:
        label_str = "Engaging (Label 1)" if video['video_label'] == 1 else "Not Engaging (Label 0)"
        lines.append(f" Ground Truth: {label_str}")

    n_out_of_vocab = sum(1 for t in query_triples if not t.get('rst_in_vocab', True))
    lines.append(f" Total scenes: {len(video['scene_ids'])} | "
                 f"RST links: {len(video['rst_links'])} | "
                 f"Query triples extracted: {len(query_triples)}"
                 + (f" ({n_out_of_vocab} with out-of-vocab RST type)" if n_out_of_vocab else ""))
    lines.append("=" * 70)

    lines.append("\n--- VIDEO CONTENT (scene by scene) ---")
    for i, (sid, cap) in enumerate(zip(video['scene_ids'][:15], video['captions'][:15])):
        c_id   = video['concept_ids'][i] if i < len(video.get('concept_ids', [])) else '?'
        c_info = concept_dict.get(int(c_id), {})
        c_name = c_info.get('concept_name', f'Concept {c_id}')
        lines.append(f"  Scene {sid} [{c_name}]: \"{cap[:120]}{'...' if len(cap) > 120 else ''}\"")

    lines.append(f"\n--- NARRATIVE PATTERNS EXTRACTED ({len(query_triples)} transitions) ---")
    for i, qt in enumerate(query_triples[:8], 1):
        src_name = concept_dict.get(qt['c_src'], {}).get('concept_name', f"Concept {qt['c_src']}")
        tgt_name = concept_dict.get(qt['c_tgt'], {}).get('concept_name', f"Concept {qt['c_tgt']}")
        vocab_flag = "" if qt.get('rst_in_vocab', True) else " [OUT-OF-VOCAB]"
        lines.append(
            f"  {i}. Scene {qt['src_scene']} [{src_name}]"
            f" ──{qt['rst_type']}{vocab_flag}──▶ "
            f"Scene {qt['tgt_scene']} [{tgt_name}]"
        )

    lines.append(f"\n--- TOP-{top_k} EVIDENCE FROM LIBRARY (by relevance score) ---")

    if not top_unique:
        lines.append("  No matching evidence found in the knowledge graph.")
        lines.append("  This may indicate the RST patterns in this video do not exist in the training set.")
    else:
        for rank, cand in enumerate(top_unique, 1):
            label_val = cand['video_label']
            outcome = "ENGAGING ✓" if isinstance(label_val, int) and label_val == 1 \
                else ("NOT ENGAGING ✗" if isinstance(label_val, int) else str(label_val))

            c_src_id = int(cand['c_src'].replace("concept_", ""))
            c_tgt_id = int(cand['c_tgt'].replace("concept_", ""))
            src_name = concept_dict.get(c_src_id, {}).get('concept_name', cand['c_src'])
            tgt_name = concept_dict.get(c_tgt_id, {}).get('concept_name', cand['c_tgt'])
            src_def  = concept_dict.get(c_src_id, {}).get('definition', '')
            tgt_def  = concept_dict.get(c_tgt_id, {}).get('definition', '')

            src_cap = cand['src_caption'][:100] + '...' if len(cand['src_caption']) > 100 else cand['src_caption']
            tgt_cap = cand['tgt_caption'][:100] + '...' if len(cand['tgt_caption']) > 100 else cand['tgt_caption']

            lines.append(f"\n  [{rank}] Score: {cand['mlp_score']:.4f} | Outcome: {outcome}")
            lines.append(f"       Pattern: [{src_name}] ──{cand['rst_type']}──▶ [{tgt_name}]")
            if src_def:
                lines.append(f"       {src_name}: {src_def}")
            if tgt_def:
                lines.append(f"       {tgt_name}: {tgt_def}")
            if src_cap:
                lines.append(f"       Scene excerpt (from): \"{src_cap}\"")
            if tgt_cap:
                lines.append(f"       Scene excerpt (to):   \"{tgt_cap}\"")

    if top_unique:
        n_engaging     = sum(1 for c in top_unique if c['video_label'] == 1)
        n_not_engaging = sum(1 for c in top_unique if c['video_label'] == 0)
        avg_score      = sum(c['mlp_score'] for c in top_unique) / len(top_unique)
        lines.append(f"\n--- SUMMARY ---")
        lines.append(f"  Retrieved: {len(scored_candidates)} raw candidates → {len(top_unique)} unique videos in top-{top_k}")
        lines.append(f"  Engaging evidence    : {n_engaging}/{len(top_unique)}")
        lines.append(f"  Not-engaging evidence: {n_not_engaging}/{len(top_unique)}")
        lines.append(f"  Average relevance score: {avg_score:.4f}")

        if n_engaging > n_not_engaging:
            hint = "→ Evidence leans ENGAGING"
        elif n_not_engaging > n_engaging:
            hint = "→ Evidence leans NOT ENGAGING"
        else:
            hint = "→ Evidence is mixed"
        lines.append(f"  {hint}")

    lines.append("=" * 70 + "\n")
    return "\n".join(lines), metrics


# ==========================================
# 9. SINGLE VIDEO PIPELINE
# ==========================================

def process_single_video(
    video_id: str, data_root: Path, index: dict, driver, top_k: int, candidate_limit: int,
) -> tuple[str, dict]:
    """Chạy toàn bộ retrieval pipeline cho một video test."""
    print(f"\n[INFO] Processing video: {video_id}")
    
    video = load_video_test(video_id, data_root)
    if video is None:
        return f"[ERROR] Cannot load video {video_id}\n", None

    video = assign_concepts_to_video(
        video,
        centroids=index['centroids'],
        pca=index.get('pca'),
        centroids_pca=index.get('centroids_pca'),
    )
    print(f"  Scenes: {len(video['scene_ids'])} | "
          f"Concepts assigned: {len(set(video['concept_ids']))} unique")

    query_triples = extract_query_triples(video, index['rst_vocab'])
    print(f"  Query triples extracted: {len(query_triples)}")

    if not query_triples:
        return (f"[WARN] Video {video_id} has no RST links → no query triples.\n"
                f"Check rst_links in scene_embeddings.pt.\n"), None

    print(f"  Querying Neo4j (limit={candidate_limit})...")
    candidates = retrieve_candidates_from_neo4j(driver, query_triples, candidate_limit)
    print(f"  Raw candidates retrieved: {len(candidates)}")

    if not candidates:
        missing = set()
        for qt in query_triples:
            if qt.get('rst_in_vocab', True):
                missing.add((f"concept_{qt['c_src']}", qt['rst_type'], f"concept_{qt['c_tgt']}"))
        diag = "\n".join(f"    {a} ─{r}─▶ {b}" for a, r, b in list(missing)[:5])
        return (f"[WARN] No candidates found for video {video_id}.\n"
                f"  Query triples attempted (sample):\n{diag}\n"
                f"  Possible causes:\n"
                f"  1. These concept-pair/RST patterns don't exist in training data\n"
                f"  2. Concept IDs don't match training concepts (centroids/PCA mismatch)\n"
                f"  3. Neo4j missing [:HAS_CONCEPT] relationships — re-run neo4j_concept_indexing.py\n"), None

    candidates = score_candidates(
        candidates, index['centroids'], index['mlp'], index['rst_vocab'], index['prior_scores'],
    )

    output, metrics = format_evidence_output(video, query_triples, candidates, index['concept_dict'], top_k)
    return output, metrics


# ==========================================
# 10. MAIN
# ==========================================

def main(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir)
    data_root = Path(args.data_root)
    
    # Xác định đường dẫn summary file
    if args.summary_file:
        summary_path = Path(args.summary_file)
    else:
        summary_path = index_dir / "video_summaries.csv"

    index = load_concept_index(index_dir)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
        print("[INFO] Neo4j connection established.\n")
    except Exception as e:
        raise RuntimeError(f"Cannot connect to Neo4j: {e}")

    if args.video_id:
        video_ids = [args.video_id]
    elif args.split_file:
        with open(args.split_file, 'r') as f:
            splits = json.load(f)
        test_ids = splits.get("test", [])

        if args.n_samples and args.n_samples < len(test_ids):
            random.seed(42)
            video_ids = random.sample(test_ids, args.n_samples)
            print(f"[INFO] Sampled {len(video_ids)} videos from {len(test_ids)} test IDs.")
        else:
            video_ids = test_ids
            print(f"[INFO] Processing all {len(video_ids)} test videos.")
    else:
        raise ValueError("Provide either --video_id or --split_file.")

    all_metrics = []
    out_file = open(args.output_file, 'w', encoding='utf-8') if args.output_file else None

    # Tạo file CSV header nếu chưa có
    if not summary_path.exists():
        import csv
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['video_id', 'ground_truth', 'engaging_count', 'not_engaging_count',
                             'leans', 'raw_candidates', 'unique_top_k', 'avg_score'])

    for vid in video_ids:
        try:
            output, metrics = process_single_video(
                video_id=vid, data_root=data_root, index=index, driver=driver,
                top_k=args.top_k, candidate_limit=args.candidate_limit,
            )
            print(output)

            if metrics is not None:
                all_metrics.append(metrics)
                # Ghi ngay vào file CSV sau mỗi video
                import csv
                with open(summary_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        metrics['video_id'],
                        metrics['ground_truth'],
                        metrics['engaging_count'],
                        metrics['not_engaging_count'],
                        metrics['leans'],
                        metrics['raw_candidates'],
                        metrics['unique_top_k'],
                        f"{metrics['avg_score']:.4f}"
                    ])

            if out_file:
                out_file.write(output)
                out_file.flush()
        except Exception as e:
            msg = f"[CRITICAL ERROR] {vid}: {e}\n"
            print(msg)
            if out_file:
                out_file.write(msg)

    # Báo cáo metrics tổng hợp
    if all_metrics and args.split_file:
        print("\n" + "=" * 70)
        print(" BATCH METRICS SUMMARY")
        print("=" * 70)
        valid = [m for m in all_metrics if m['ground_truth'] is not None]
        if not valid:
            print("No ground truth available for metrics.")
        else:
            correct = 0
            total = 0
            y_true = []
            y_pred = []
            for m in valid:
                gt = m['ground_truth']
                leans = m['leans']
                if leans == 'mixed':
                    continue
                pred = 1 if leans == 'engaging' else 0
                y_true.append(gt)
                y_pred.append(pred)
                if gt == pred:
                    correct += 1
                total += 1
            print(f"Total videos (exclude mixed): {total}")
            if total > 0:
                print(f"Accuracy: {correct}/{total} = {correct/total:.4f}")
                from sklearn.metrics import confusion_matrix, classification_report
                cm = confusion_matrix(y_true, y_pred)
                print("Confusion Matrix:")
                print("          Predicted")
                print("          NotEng  Eng")
                print(f"Actual 0   {cm[0,0]:5d}  {cm[0,1]:5d}")
                print(f"       1   {cm[1,0]:5d}  {cm[1,1]:5d}")
                print("\nClassification Report:")
                print(classification_report(y_true, y_pred, target_names=['Not Engaging', 'Engaging']))
            else:
                print("No non-mixed predictions.")
        print("=" * 70 + "\n")
    
    # In thông báo summary file
    if all_metrics:
        print(f"\n[INFO] Video summaries saved to: {summary_path}")

    if out_file:
        out_file.close()
        print(f"\n[INFO] Full results saved to: {args.output_file}")

    driver.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "ConceptRAG — retrieve top-K evidence triples for video test(s).\n"
            "Neo4j credentials are read from env vars."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--index_dir",       type=str, required=True,
                        help="Directory containing centroids.pt, pca.pkl, concept_dict.json, "
                             "rst_vocab.json, mlp_scorer.pt")
    parser.add_argument("--data_root",       type=str, required=True,
                        help="Root directory containing per-video folders.")
    parser.add_argument("--video_id",        type=str, default=None,
                        help="Single video ID to process.")
    parser.add_argument("--split_file",      type=str, default=None,
                        help="Path to dataset_splits.json — sử dụng split 'test'.")
    parser.add_argument("--n_samples",       type=int, default=10,
                        help="Số video test lấy ngẫu nhiên từ split file. Default: 10")
    parser.add_argument("--top_k",           type=int, default=5,
                        help="Số evidence triples (unique videos) muốn xem. Default: 5")
    parser.add_argument("--candidate_limit", type=int, default=200,
                        help="Số candidates tối đa lấy từ Neo4j trước khi score. Default: 200")
    parser.add_argument("--output_file",     type=str, default=None,
                        help="Lưu kết quả ra file text (tuỳ chọn).")
    parser.add_argument("--summary_file", type=str, default=None,
                    help="Lưu summary metrics của từng video vào file CSV. "
                         "Mặc định: <index_dir>/video_summaries.csv")
    
    args = parser.parse_args()
    main(args)