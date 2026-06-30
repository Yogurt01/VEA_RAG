"""
retrieve_evidence.py
--------------------
Phase 2 — Online retrieval (chạy cho từng video test).

Pipeline:
    Step A : Load centroids.pt + concept_dict.json + mlp_scorer.pt
    Step B : Assign concept_id cho từng scene của video test
             (cosine với centroids, không cần Neo4j)
    Step C : Extract query triples (c_src, RST_type, c_tgt) có hướng
    Step D : Cypher query Neo4j — tìm candidate triples khớp từng query triple
    Step E : Build feature vector + MLP score song song
    Step F : In top-K evidence theo format tự nhiên (không jargon kỹ thuật)

Usage — single video:
    python retrieve_evidence.py \
        --index_dir       /path/to/concept_index \
        --data_root       /path/to/All_Videos \
        --video_id        b0f9d31bde3573c94ba3580a36af6b70 \
        [--top_k          5] \
        [--candidate_limit 200]

Usage — batch (đọc từ split file, lấy ngẫu nhiên N video test):
    python retrieve_evidence.py \
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
# 1. MLP SCORER (định nghĩa lại để load checkpoint)
# ==========================================

class ConceptTripleScorer(nn.Module):
    def __init__(self, embed_dim: int, n_rst_types: int, hidden_dim: int = 256):
        super().__init__()
        in_dim = 2 * embed_dim + n_rst_types + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
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
    Load tất cả artifacts từ Phase 1:
        centroids   : np.ndarray (K, 2048)
        concept_dict: dict {str(concept_id): {concept_name, definition}}
        mlp         : ConceptTripleScorer (eval mode)
        rst_vocab   : dict {rst_type: idx}
        prior_scores: dict {(c_src, rst_type, c_tgt): float}
    """
    print("[INFO] Loading concept index...")

    # Centroids
    centroids = torch.load(index_dir / "centroids.pt", map_location="cpu").numpy()
    pca = None
    centroids_pca = None
    pca_path = index_dir / "pca.pkl"
    if pca_path.exists():
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        # Transform centroids sang không gian PCA
        centroids_norm = normalize(centroids, norm='l2')
        centroids_pca = pca.transform(centroids_norm)
        centroids_pca = normalize(centroids_pca, norm='l2')
        print(f"  pca.pkl loaded (dim={pca.n_components_})")
        print(f"  centroids_pca shape: {centroids_pca.shape}")
    else:
        print("  [WARN] pca.pkl not found → using direct cosine on full dim")
        centroids_pca = None

    
    # Concept dictionary
    with open(index_dir / "concept_dict.json", 'r', encoding='utf-8') as f:
        concept_dict_raw = json.load(f)
    # Key có thể là string int → normalize về int key
    concept_dict = {int(k): v for k, v in concept_dict_raw.items()}
    print(f"  concept_dict.json: {len(concept_dict)} concepts")

    # MLP checkpoint
    ckpt           = torch.load(index_dir / "mlp_scorer.pt", map_location="cpu")
    embed_dim      = ckpt['embed_dim']
    n_rst_types    = ckpt['n_rst_types']
    hidden_dim     = ckpt.get('hidden_dim', 256)
    rst_type_to_idx = ckpt['rst_type_to_idx']

    mlp = ConceptTripleScorer(embed_dim=embed_dim, n_rst_types=n_rst_types, hidden_dim=hidden_dim)
    mlp.load_state_dict(ckpt['model_state_dict'])
    mlp.eval()
    print(f"  mlp_scorer.pt loaded (embed_dim={embed_dim}, n_rst={n_rst_types})")

    # Prior scores — key lưu dạng str tuple "('3', 'CONTRAST', '84')" → parse lại
    prior_scores_raw = ckpt.get('prior_scores', {})
    prior_scores = {}
    for k_str, v in prior_scores_raw.items():
        try:
            # Key dạng "(c_src, rst_type, c_tgt)"
            import ast
            parts = ast.literal_eval(k_str)
            prior_scores[(int(parts[0]), str(parts[1]), int(parts[2]))] = float(v)
        except Exception:
            pass
    print(f"  prior_scores: {len(prior_scores)} entries")

    return {
        'centroids':      centroids,       # (K, 2048) — dùng cho MLP features
        'centroids_pca':  centroids_pca,   # (K, 128) hoặc None — dùng để assign concept
        'concept_dict':   concept_dict,
        'mlp':            mlp,
        'rst_vocab':      rst_type_to_idx,
        'prior_scores':   prior_scores,
        'embed_dim':      embed_dim,
        'n_rst_types':    n_rst_types,
        'pca':            pca,
    }

# ==========================================
# 3. LOAD VIDEO TEST DATA
# ==========================================

def load_video_test(video_id: str, data_root: Path) -> dict | None:
    """
    Load scene_embeddings.pt + segments.json của một video test.
    """
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
    Hai không gian so sánh phải nhất quán với nhau.
    """
    embs_np = video['embeddings'].numpy().astype(np.float32)   # (T, 2048)

    if pca is not None and centroids_pca is not None:
        # Transform scene embeddings về cùng không gian PCA với centroids_pca
        embs_norm = normalize(embs_np, norm='l2')              # (T, 2048)
        embs_pca  = pca.transform(embs_norm)                   # (T, 128)
        embs_pca  = normalize(embs_pca, norm='l2')             # (T, 128)

        query_t       = torch.from_numpy(embs_pca).float()     # (T, 128)
        centroids_t   = torch.from_numpy(centroids_pca).float()  # (K, 128)
    else:
        # Fallback: cosine trực tiếp trong không gian 2048-dim
        embs_norm     = normalize(embs_np, norm='l2')
        query_t       = torch.from_numpy(embs_norm).float()    # (T, 2048)
        centroids_t   = torch.from_numpy(centroids).float()    # (K, 2048)

    scores = query_t @ centroids_t.T                           # (T, K)
    c_ids  = scores.argmax(dim=1).tolist()                     # (T,)

    video['concept_ids'] = c_ids
    return video


# ==========================================
# 5. STEP C: EXTRACT QUERY TRIPLES (có hướng)
# ==========================================

def extract_query_triples(video: dict) -> list[dict]:
    """
    Tạo danh sách query triples từ RST links của video test.
    Giữ đúng hướng (src → tgt) — khác với thiết kế cũ không kiểm tra hướng.
    """
    c_ids     = video['concept_ids']
    scene_ids = video['scene_ids']
    triples   = []

    for src, tgt, rst_type in video['rst_links']:
        src_int = int(src) - 1
        tgt_int = int(tgt) - 1
        try:
            src_idx = scene_ids.index(src_int)
            tgt_idx = scene_ids.index(tgt_int)
        except ValueError:
            continue

        rst_upper = str(rst_type).strip().upper().replace(" ", "_").replace("-", "_")
        triples.append({
            'c_src':      c_ids[src_idx],
            'rst_type':   rst_upper,
            'c_tgt':      c_ids[tgt_idx],
            'src_scene':  src_int,
            'tgt_scene':  tgt_int,
            'src_caption': video['captions'][src_idx] if src_idx < len(video['captions']) else '',
            'tgt_caption': video['captions'][tgt_idx] if tgt_idx < len(video['captions']) else '',
        })

    return triples


# ==========================================
# 6. STEP D: NEO4J CANDIDATE RETRIEVAL
# ==========================================

def retrieve_candidates_from_neo4j(
    driver,
    query_triples: list[dict],
    candidate_limit: int = 200,
) -> list[dict]:
    """
    Với mỗi query triple (c_src, RST_type, c_tgt), tìm các video training
    có cùng pattern trong Neo4j — đúng hướng, đúng RST type.

    Trả về list[dict] candidates, mỗi item gồm:
        video_id, video_label, c_src, rst_type, c_tgt,
        src_caption, tgt_caption, query_triple_idx
    """
    if not query_triples:
        return []

    # Chuẩn bị params cho UNWIND
    triple_params = [
        {
            'c_src':     f"concept_{t['c_src']}",
            'rst_type':  t['rst_type'],
            'c_tgt':     f"concept_{t['c_tgt']}",
            'triple_idx': i,
        }
        for i, t in enumerate(query_triples)
    ]

    # Query siết chặt: đúng hướng, đúng concept, đúng RST type
    # Không lọc is_test IS NULL vì video test chưa có flag này trong retrieval
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
    candidates: list[dict],
    centroids: np.ndarray,
    mlp: ConceptTripleScorer,
    rst_vocab: dict[str, int],
    prior_scores: dict[tuple, float],
) -> list[dict]:
    """
    Build feature vectors + score tất cả candidates song song bằng MLP.
    Thêm key 'mlp_score' vào mỗi candidate.
    """
    if not candidates:
        return []

    n_rst  = len(rst_vocab)
    D      = centroids.shape[1]
    feats  = []

    for cand in candidates:
        # Parse concept id số từ string "concept_3" → 3
        try:
            c_src_idx = int(cand['c_src'].replace("concept_", ""))
            c_tgt_idx = int(cand['c_tgt'].replace("concept_", ""))
        except ValueError:
            c_src_idx, c_tgt_idx = 0, 0

        emb_src    = centroids[c_src_idx]
        emb_tgt    = centroids[c_tgt_idx]

        rst_onehot = np.zeros(n_rst, dtype=np.float32)
        rst_idx    = rst_vocab.get(cand['rst_type'], 0)
        rst_onehot[rst_idx] = 1.0

        prior_key  = (c_src_idx, cand['rst_type'], c_tgt_idx)
        prior      = prior_scores.get(prior_key, 0.5)

        feat = np.concatenate([emb_src, emb_tgt, rst_onehot, [prior]])
        feats.append(feat)

    X = torch.from_numpy(np.stack(feats).astype(np.float32))

    with torch.no_grad():
        scores = mlp(X).tolist()

    for cand, score in zip(candidates, scores):
        cand['mlp_score'] = score

    return candidates


# ==========================================
# 8. STEP F: FORMAT & PRINT TOP-K EVIDENCE
# ==========================================

def format_evidence_output(
    video: dict,
    query_triples: list[dict],
    scored_candidates: list[dict],
    concept_dict: dict[int, dict],
    top_k: int,
) -> str:
    """
    Format kết quả top-K evidence theo ngôn ngữ tự nhiên —
    không dùng jargon kỹ thuật (không có RST type, không có concept_id số).
    """
    # Sắp xếp theo MLP score giảm dần, lấy top-K
    sorted_cands = sorted(scored_candidates, key=lambda x: x['mlp_score'], reverse=True)
    top_cands    = sorted_cands[:top_k]

    # Loại trùng video_id (chỉ giữ score cao nhất mỗi video)
    seen_videos = {}
    for cand in sorted_cands:
        vid = cand['video_id']
        if vid not in seen_videos:
            seen_videos[vid] = cand

    top_unique = sorted(seen_videos.values(), key=lambda x: x['mlp_score'], reverse=True)[:top_k]

    # Header
    lines = []
    lines.append("=" * 70)
    lines.append(f" VIDEO TEST: {video['video_id']}")
    if video['video_label'] is not None:
        label_str = "Engaging (Label 1)" if video['video_label'] == 1 else "Not Engaging (Label 0)"
        lines.append(f" Ground Truth: {label_str}")
    lines.append(f" Total scenes: {len(video['scene_ids'])} | "
                 f"RST links: {len(video['rst_links'])} | "
                 f"Query triples extracted: {len(query_triples)}")
    lines.append("=" * 70)

    # Scene content của video test
    lines.append("\n--- VIDEO CONTENT (scene by scene) ---")
    for i, (sid, cap) in enumerate(zip(video['scene_ids'][:15], video['captions'][:15])):
        c_id   = video['concept_ids'][i] if i < len(video.get('concept_ids', [])) else '?'
        c_info = concept_dict.get(int(c_id), {})
        c_name = c_info.get('concept_name', f'Concept {c_id}')
        lines.append(f"  Scene {sid} [{c_name}]: \"{cap[:120]}{'...' if len(cap) > 120 else ''}\"")

    # Query triples của video test
    lines.append(f"\n--- NARRATIVE PATTERNS EXTRACTED ({len(query_triples)} transitions) ---")
    for i, qt in enumerate(query_triples[:8], 1):
        src_name = concept_dict.get(qt['c_src'], {}).get('concept_name', f"Concept {qt['c_src']}")
        tgt_name = concept_dict.get(qt['c_tgt'], {}).get('concept_name', f"Concept {qt['c_tgt']}")
        lines.append(
            f"  {i}. Scene {qt['src_scene']} [{src_name}]"
            f" ──{qt['rst_type']}──▶ "
            f"Scene {qt['tgt_scene']} [{tgt_name}]"
        )

    # Top-K evidence
    lines.append(f"\n--- TOP-{top_k} EVIDENCE FROM LIBRARY (by relevance score) ---")

    if not top_unique:
        lines.append("  No matching evidence found in the knowledge graph.")
        lines.append("  This may indicate the RST types in this video do not exist in the training set.")
    else:
        for rank, cand in enumerate(top_unique, 1):
            label_val = cand['video_label']
            if isinstance(label_val, int):
                outcome = "ENGAGING ✓" if label_val == 1 else "NOT ENGAGING ✗"
            else:
                outcome = str(label_val)

            c_src_id   = int(cand['c_src'].replace("concept_", ""))
            c_tgt_id   = int(cand['c_tgt'].replace("concept_", ""))
            src_name   = concept_dict.get(c_src_id, {}).get('concept_name', cand['c_src'])
            tgt_name   = concept_dict.get(c_tgt_id, {}).get('concept_name', cand['c_tgt'])
            src_def    = concept_dict.get(c_src_id, {}).get('definition', '')
            tgt_def    = concept_dict.get(c_tgt_id, {}).get('definition', '')

            src_cap    = cand['src_caption'][:100] + '...' if len(cand['src_caption']) > 100 else cand['src_caption']
            tgt_cap    = cand['tgt_caption'][:100] + '...' if len(cand['tgt_caption']) > 100 else cand['tgt_caption']

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

    # Tóm tắt distribution trong top candidates
    if top_unique:
        n_engaging     = sum(1 for c in top_unique if c['video_label'] == 1)
        n_not_engaging = sum(1 for c in top_unique if c['video_label'] == 0)
        avg_score      = sum(c['mlp_score'] for c in top_unique) / len(top_unique)
        lines.append(f"\n--- SUMMARY ---")
        lines.append(f"  Retrieved: {len(scored_candidates)} raw candidates → {len(top_unique)} unique videos in top-{top_k}")
        lines.append(f"  Engaging evidence   : {n_engaging}/{len(top_unique)}")
        lines.append(f"  Not-engaging evidence: {n_not_engaging}/{len(top_unique)}")
        lines.append(f"  Average relevance score: {avg_score:.4f}")

        # Sơ bộ dự đoán từ evidence (không cần LLM)
        if n_engaging > n_not_engaging:
            hint = "→ Evidence leans ENGAGING"
        elif n_not_engaging > n_engaging:
            hint = "→ Evidence leans NOT ENGAGING"
        else:
            hint = "→ Evidence is mixed"
        lines.append(f"  {hint}")

    lines.append("=" * 70 + "\n")
    return "\n".join(lines)


# ==========================================
# 9. SINGLE VIDEO PIPELINE
# ==========================================

def process_single_video(
    video_id: str,
    data_root: Path,
    index: dict,
    driver,
    top_k: int,
    candidate_limit: int,
) -> str:
    """
    Chạy toàn bộ retrieval pipeline cho một video test.
    Trả về formatted string để in ra màn hình hoặc lưu file.
    """
    print(f"\n[INFO] Processing video: {video_id}")
    
    # Load video data
    video = load_video_test(video_id, data_root)
    if video is None:
        return f"[ERROR] Cannot load video {video_id}\n"

    # Step B: Assign concepts
    video = assign_concepts_to_video(
        video,
        centroids=index['centroids'],
        pca=index.get('pca'),
        centroids_pca=index.get('centroids_pca'),
    )
    print(f"  Scenes: {len(video['scene_ids'])} | "
          f"Concepts assigned: {len(set(video['concept_ids']))} unique")

    # Step C: Extract query triples
    query_triples = extract_query_triples(video)
    print(f"  Query triples extracted: {len(query_triples)}")

    if not query_triples:
        return (f"[WARN] Video {video_id} has no RST links → no query triples.\n"
                f"Check rst_links in scene_embeddings.pt.\n")

    # Step D: Neo4j candidate retrieval
    print(f"  Querying Neo4j (limit={candidate_limit})...")
    candidates = retrieve_candidates_from_neo4j(driver, query_triples, candidate_limit)
    print(f"  Raw candidates retrieved: {len(candidates)}")

    if not candidates:
        # Diagnose: thử xem concept nào không có match
        missing = set()
        for qt in query_triples:
            missing.add((f"concept_{qt['c_src']}", qt['rst_type'], f"concept_{qt['c_tgt']}"))
        diag = "\n".join(f"    {a} ─{r}─▶ {b}" for a, r, b in list(missing)[:5])
        return (f"[WARN] No candidates found for video {video_id}.\n"
                f"  Query triples attempted (sample):\n{diag}\n"
                f"  Possible causes:\n"
                f"  1. RST types in this video don't exist in training data\n"
                f"  2. Concept IDs don't match training concepts (centroids mismatch)\n"
                f"  3. Neo4j missing [:HAS_CONCEPT] relationships — re-run build_concept_index.py\n")

    # Step E: MLP scoring
    candidates = score_candidates(
        candidates,
        index['centroids'],
        index['mlp'],
        index['rst_vocab'],
        index['prior_scores'],
    )

    # Step F: Format output
    output = format_evidence_output(
        video, query_triples, candidates, index['concept_dict'], top_k
    )
    return output


# ==========================================
# 10. MAIN
# ==========================================

def main(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir)
    data_root = Path(args.data_root)

    # Load concept index (một lần)
    index = load_concept_index(index_dir)

    # Kết nối Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
        print("[INFO] Neo4j connection established.\n")
    except Exception as e:
        raise RuntimeError(f"Cannot connect to Neo4j: {e}")

    # Xác định danh sách video cần xử lý
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

    # Output file (tuỳ chọn)
    out_file = open(args.output_file, 'w', encoding='utf-8') if args.output_file else None

    # Chạy pipeline
    all_outputs = []
    for vid in video_ids:
        try:
            result = process_single_video(
                video_id=vid,
                data_root=data_root,
                index=index,
                driver=driver,
                top_k=args.top_k,
                candidate_limit=args.candidate_limit,
            )
            print(result)
            all_outputs.append(result)

            if out_file:
                out_file.write(result)
                out_file.flush()

        except Exception as e:
            msg = f"[CRITICAL ERROR] {vid}: {e}\n"
            print(msg)
            if out_file:
                out_file.write(msg)

    if out_file:
        out_file.close()
        print(f"\n[INFO] Results saved to: {args.output_file}")

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
    parser.add_argument("--index_dir",      type=str, required=True,
                        help="Directory containing centroids.pt, concept_dict.json, mlp_scorer.pt")
    parser.add_argument("--data_root",      type=str, required=True,
                        help="Root directory containing per-video folders.")

    # Chế độ single video
    parser.add_argument("--video_id",       type=str, default=None,
                        help="Single video ID to process.")

    # Chế độ batch
    parser.add_argument("--split_file",     type=str, default=None,
                        help="Path to dataset_splits.json — sử dụng split 'test'.")
    parser.add_argument("--n_samples",      type=int, default=10,
                        help="Số video test lấy ngẫu nhiên từ split file. Default: 10")

    parser.add_argument("--top_k",          type=int, default=5,
                        help="Số evidence triples (unique videos) muốn xem. Default: 5")
    parser.add_argument("--candidate_limit",type=int, default=200,
                        help="Số candidates tối đa lấy từ Neo4j trước khi score. Default: 200")
    parser.add_argument("--output_file",    type=str, default=None,
                        help="Lưu kết quả ra file text (tuỳ chọn).")

    args = parser.parse_args()
    main(args)