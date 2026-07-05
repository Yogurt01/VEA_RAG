"""
neo4j_concept_indexing.py
----------------------
Phase 1 — Offline build (chạy một lần trên train+val).

Cải tiến so với phiên bản cũ:
    [Step 1] GMM thay K-Means — soft assignment, phát hiện overlap tự nhiên
             hơn, không ép scene vào đúng 1 cluster cứng.
             GMM vẫn dùng PCA trước để giảm chiều, nhưng với n_components
             nhỏ hơn (64-dim) vì GMM hoạt động tốt trong không gian thấp chiều.
    [Step 2] Soft concept assignment — mỗi scene có TOP-P concepts (thay vì 1)
             để giảm collapse và cho phép triple matching linh hoạt hơn.
    [Step 3] Concept naming: bỏ log từng concept, chỉ in tổng kết.
    [Step 4–6] Giữ nguyên logic Neo4j upload + MLP training.
    [Output] Tất cả artifacts lưu vào <output_dir>/.
    
Lý do chọn GMM thay K-Means:
    - K-Means ép hard assignment (mỗi scene thuộc đúng 1 cluster) →
      scene nằm ở ranh giới 2 concept bị gán sai → concept collapse.
    - GMM sinh soft probabilities P(concept_k | scene_i) → scene "ranh giới"
      thuộc nhiều concept với trọng số khác nhau → phản ánh đúng hơn bản
      chất liên tục của embedding space (đã xác nhận silhouette 0.10–0.15).
    - Với soft assignment, Triple extraction lấy TOP-P concepts (P=2 hoặc 3)
      cho mỗi scene → số lượng triple tăng lên, coverage cải thiện.
    - GMM vẫn học được "hình dạng" của cluster (covariance), phù hợp với
      embedding không có ranh giới hình cầu cứng như K-Means giả định.

Usage:
    python neo4j_concept_indexing.py \\
        --data_root      /path/to/All_Videos \\
        --split_file     /path/to/dataset_splits.json \\
        --output_dir     /path/to/indexing \\
        --n_concepts     150 \\
        [--pca_dim       64] \\
        [--top_p         2]  \\
        [--model_name    Qwen/Qwen3-4B-Instruct-2507] \\
        [--force]

Environment variables (Neo4j):
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE
"""

import os
import gc
import json
import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from neo4j import GraphDatabase
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


# ==========================================
# 0. FIXED RST VOCABULARY
# ==========================================

EDGE_TYPES = [
    "elaboration", "attribution", "joint", "same-unit", "contrast",
    "explanation", "background", "cause", "enablement", "evaluation",
    "temporal", "condition", "comparison", "topic-change", "summary",
    "manner-means", "textual-organization", "topic-comment", "ROOT", "span",
]


def normalize_rst_type(rst_type: str) -> str:
    return str(rst_type).strip().upper().replace(" ", "_").replace("-", "_")


def build_fixed_rst_vocab() -> dict:
    return {normalize_rst_type(r): i for i, r in enumerate(EDGE_TYPES)}


# ==========================================
# 1. MLP SCORER DEFINITION
# ==========================================

class ConceptTripleScorer(nn.Module):
    """
    MLP scorer với feature vector scalar 24-dim.
    Soft assignment tạo nhiều triple hơn → prior_scores phong phú hơn,
    MLP học được signal tốt hơn so với hard assignment cũ.
    """
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
# 2. DATA LOADING
# ==========================================

def load_split_videos(data_root: Path, split_file: Path) -> list:
    """Load train+val videos — giống phiên bản cũ, không thay đổi."""
    with open(split_file, 'r') as f:
        splits = json.load(f)
    allowed = set(splits.get("train", [])) | set(splits.get("val", []))
    print(f"[INFO] Found {len(allowed)} video IDs in train+val split.")

    videos, skipped = [], 0

    for video_name in sorted(allowed):
        video_dir = data_root / video_name
        emb_path  = video_dir / "scene_embeddings.pt"
        seg_path  = video_dir / "segments.json"

        if not emb_path.exists() or not seg_path.exists():
            skipped += 1
            continue

        try:
            data = torch.load(emb_path, map_location="cpu")
            with open(seg_path, 'r', encoding='utf-8') as f:
                segments = json.load(f)

            scene_ids = data['scene_ids']
            captions  = []
            if isinstance(segments, list):
                for idx in range(len(scene_ids)):
                    cap = segments[idx].get('caption', '') if idx < len(segments) else ''
                    captions.append(cap)
            elif isinstance(segments, dict):
                for sid in scene_ids:
                    entry = segments.get(str(int(sid))) or segments.get(int(sid), {})
                    cap   = entry.get('caption', '') if isinstance(entry, dict) else str(entry)
                    captions.append(cap)

            label = int(data['y'].item() if isinstance(data['y'], torch.Tensor) else data['y'])
            videos.append({
                'video_name':  video_name,
                'video_label': label,
                'embeddings':  data['embeddings'],
                'scene_ids':   [int(s) for s in scene_ids],
                'rst_links':   data.get('rst_links', []),
                'captions':    captions,
            })

        except Exception as e:
            print(f"  [WARN] Skipping {video_name}: {e}")
            skipped += 1

    print(f"[INFO] Loaded {len(videos)} videos ({skipped} skipped).")
    return videos


# ==========================================
# 3. GMM CLUSTERING (thay K-Means)
# ==========================================

def run_gmm(videos: list, n_concepts: int, pca_dim: int = 64) -> tuple:
    """
    PCA (2048 → pca_dim) + GaussianMixture (GMM) thay K-Means.

    Lý do dùng pca_dim=64 thay vì 128:
        GMM ước lượng ma trận covariance (n_concepts × pca_dim × pca_dim)
        → chi phí O(K × D²). Với D=128, K=150: 150 × 128² ≈ 2.5M params.
        Với D=64, K=150: 150 × 64² ≈ 614K params — nhanh hơn 4× và ổn định
        hơn numerically với dataset 12000 scene.

    covariance_type='diag' (default): chỉ học diagonal covariance, trade-off
        tốt giữa biểu đạt và ổn định — full covariance dễ bị singular.

    Trả về:
        gmm          : fitted GaussianMixture object
        pca          : fitted IncrementalPCA object
        centroids    : (K, 2048) mean của mỗi component trong full-dim space,
                       dùng làm anchor cho MLP feature (cosine similarity)
        proba_matrix : (N_scenes_total, K) soft assignment matrix,
                       dùng để assign TOP-P concepts mỗi scene
    """
    print(f"\n[Step 1] Running PCA (2048 → {pca_dim}) + GMM (K={n_concepts})...")

    all_embs = [v['embeddings'].numpy().astype(np.float32) for v in videos]
    X        = np.vstack(all_embs)    # (N, 2048)
    print(f"  Total scenes: {X.shape[0]}")

    X_norm = normalize(X, norm='l2')

    # IncrementalPCA để tránh OOM
    CHUNK = 4096
    pca   = IncrementalPCA(n_components=pca_dim)
    for i in range(0, len(X_norm), CHUNK):
        pca.partial_fit(X_norm[i:i + CHUNK])

    X_pca = normalize(pca.transform(X_norm), norm='l2')    # (N, pca_dim)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA done. Explained variance ratio sum: {var_explained:.3f}")
    if var_explained < 0.70:
        print(f"  [WARNING] Low variance retained ({var_explained:.1%}). "
              f"Consider increasing --pca_dim.")

    # GMM — warm_start=False, n_init=3 để tránh local optima
    print(f"  Fitting GMM ({n_concepts} components, covariance_type=diag)...")
    gmm = GaussianMixture(
        n_components=n_concepts,
        covariance_type='diag',
        n_init=3,
        max_iter=200,
        random_state=42,
        verbose=0,
    )
    gmm.fit(X_pca)
    labels       = gmm.predict(X_pca)          # (N,) hard labels để evaluate
    proba_matrix = gmm.predict_proba(X_pca)    # (N, K) soft assignment

    # Silhouette trên hard labels để so sánh với K-Means baseline
    n = len(labels)
    sample_size = 5000
    if n > sample_size:
        idx_sub = np.random.RandomState(42).choice(n, sample_size, replace=False)
        X_sub, labels_sub = X_pca[idx_sub], labels[idx_sub]
    else:
        X_sub, labels_sub = X_pca, labels
    sil = silhouette_score(X_sub, labels_sub, metric='cosine')

    counts = Counter(labels)
    max_ratio = max(counts.values()) / n
    tiny = sum(1 for c in counts.values() if c < 5)

    print(f"  GMM done.")
    print(f"  Silhouette score (cosine, n={len(X_sub)}): {sil:.4f}  (higher is better)")
    print(f"  Max concept ratio: {max_ratio:.1%}  (lower is better)")
    print(f"  Tiny clusters (<5 scenes): {tiny}/{n_concepts}  (lower is better)")
    print(f"  Concept sizes: min={min(counts.values())}, "
          f"max={max(counts.values())}, avg={n/n_concepts:.1f}")

    # Tính centroid trong full-dim space (mean của scene thuộc cluster, weighted)
    # Dùng hard assignment để tính centroid cho đơn giản và ổn định
    centroids_full = np.zeros((n_concepts, 2048), dtype=np.float32)
    counts_arr     = np.zeros(n_concepts, dtype=np.int32)
    for i, lbl in enumerate(labels):
        centroids_full[lbl] += X_norm[i]
        counts_arr[lbl]     += 1
    counts_arr = np.maximum(counts_arr, 1)
    centroids_full /= counts_arr[:, None]
    centroids_full  = normalize(centroids_full, norm='l2')

    print(f"  Centroid matrix (full-dim): {centroids_full.shape}")
    return gmm, pca, centroids_full, proba_matrix


# ==========================================
# 4. SOFT CONCEPT ASSIGNMENT (TOP-P)
# ==========================================

def assign_concept_ids_soft(videos: list, proba_matrix: np.ndarray, top_p: int = 2) -> list:
    """
    Soft assignment: mỗi scene nhận TOP-P concept IDs có probability cao nhất.
    Khác với hard assignment cũ (chỉ 1 concept), TOP-P giảm collapse risk và
    tăng coverage của triple extraction (nhiều triple hơn từ cùng dữ liệu).

    'concept_ids'      : list[int] — concept chính (argmax, dùng cho Cypher
                         query với filter rst_type)
    'concept_ids_topk' : list[list[int]] — top-P concepts mỗi scene,
                         dùng cho triple extraction đa concept

    Lưu thêm 'concept_proba' để có thể weight triple contribution sau này.
    """
    cursor = 0
    for v in videos:
        n = len(v['scene_ids'])
        p = proba_matrix[cursor:cursor + n]   # (T, K)

        # Hard assignment (primary)
        v['concept_ids'] = p.argmax(axis=1).tolist()

        # Soft assignment TOP-P
        top_p_ids = np.argsort(p, axis=1)[:, -top_p:][:, ::-1].tolist()   # (T, top_p)
        v['concept_ids_topk'] = top_p_ids

        # Probability của hard concept — dùng cho scoring sau này
        v['concept_proba'] = p.max(axis=1).tolist()

        cursor += n

    return videos


def evaluate_per_video_diversity(videos: list, threshold: float = 0.3) -> None:
    """Đo concept collapse ở cấp video — chỉ số quan trọng nhất."""
    ratios = [len(set(v['concept_ids'])) / len(v['concept_ids']) for v in videos]
    ratios = np.array(ratios)
    low    = (ratios < threshold).sum()
    print(f"  Per-video concept diversity: mean={ratios.mean():.3f}, "
          f"median={np.median(ratios):.3f}")
    print(f"  Videos with <{threshold:.0%} diversity (collapse risk): "
          f"{low}/{len(videos)} ({low/len(videos):.1%})")


# ==========================================
# 5. CONCEPT NAMING (Qwen3-4B)
# ==========================================

def get_top_captions_per_concept(videos: list, n_concepts: int, top_n: int) -> dict:
    """
    Lấy top-N captions gần centroid nhất cho mỗi concept.
    Dùng hard concept assignment + probability để rank.
    """
    # {concept_id: [(proba, caption), ...]}
    buckets = defaultdict(list)

    for v in videos:
        for idx, (c_id, prob, cap) in enumerate(zip(
            v['concept_ids'], v['concept_proba'], v['captions']
        )):
            if cap:
                buckets[c_id].append((prob, cap))

    top_captions = {}
    for c_id in range(n_concepts):
        items = sorted(buckets.get(c_id, []), key=lambda x: x[0], reverse=True)
        top_captions[c_id] = [cap for _, cap in items[:top_n]]

    return top_captions


def name_concept_with_qwen(
    concept_id: int, captions: list, tokenizer, model,
) -> dict:
    """Đặt tên concept bằng Qwen3-4B — không log individual output."""
    if not captions:
        return {
            "concept_id":   f"concept_{concept_id}",
            "concept_name": f"Concept {concept_id}",
            "definition":   "No captions available.",
        }

    caption_block = "\n".join(f"- {c}" for c in captions[:10])
    prompt = (
        f"Below are {len(captions)} video scene descriptions from the same "
        f"visual-semantic cluster.\n\nScene descriptions:\n{caption_block}\n\n"
        f"Provide a short concept label and one-sentence definition.\n"
        f'Respond ONLY with valid JSON, no markdown: '
        f'{{"concept_name": "2-4 word label", "definition": "one sentence."}}'
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs    = tokenizer([text], return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128)

    response = tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
    ).strip()

    import re
    try:
        match  = re.search(r'\{.*\}', response, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {}
        return {
            "concept_id":   f"concept_{concept_id}",
            "concept_name": parsed.get("concept_name", f"Concept {concept_id}"),
            "definition":   parsed.get("definition",   ""),
        }
    except Exception:
        return {
            "concept_id":   f"concept_{concept_id}",
            "concept_name": f"Concept {concept_id}",
            "definition":   response[:200],
        }


def build_concept_dictionary(
    videos: list, n_concepts: int, top_n: int,
    tokenizer, model, log_dir: Path,
) -> dict:
    """
    Đặt tên toàn bộ K concepts — không log từng concept ra stdout,
    chỉ in tổng kết cuối cùng.
    """
    print(f"\n[Step 3] Naming {n_concepts} concepts...")
    top_captions = get_top_captions_per_concept(videos, n_concepts, top_n)
    log_dir.mkdir(parents=True, exist_ok=True)
    concept_dict = {}

    for c_id in range(n_concepts):
        caps = top_captions.get(c_id, [])
        info = name_concept_with_qwen(c_id, caps, tokenizer, model)
        concept_dict[c_id] = info

        # Lưu log riêng từng concept vào file, không print ra stdout
        log_path = log_dir / f"concept_{c_id:03d}.txt"
        with open(log_path, 'w', encoding='utf-8') as lf:
            lf.write(f"Captions:\n" + "\n".join(f"  - {c}" for c in caps))
            lf.write(f"\n\nResult: {json.dumps(info, ensure_ascii=False)}\n")

    print(f"  Done naming {len(concept_dict)} concepts.")
    print(f"  Naming logs saved to: {log_dir}")
    return concept_dict


# ==========================================
# 6. NEO4J UPLOAD
# ==========================================

def clear_concepts(tx):
    tx.run("""
        MATCH (c:Concept)
        OPTIONAL MATCH (c)<-[r:HAS_CONCEPT]-()
        DETACH DELETE c
    """)
    print("  All Concept nodes and HAS_CONCEPT relationships removed.")


def upload_concept_graph(driver, videos: list, concept_dict: dict) -> None:
    """Upload Concept nodes + HAS_CONCEPT relationships lên Neo4j."""
    print("\n[Step 4] Uploading concept graph to Neo4j...")

    concept_batch = [
        {"id": info["concept_id"], "name": info["concept_name"],
         "definition": info["definition"]}
        for info in concept_dict.values()
    ]
    with driver.session(database=NEO4J_DATABASE) as session:
        session.run("""
            UNWIND $batch AS c
            MERGE (con:Concept {id: c.id})
            SET con.name = c.name, con.definition = c.definition
        """, batch=concept_batch)
    print(f"  Uploaded {len(concept_batch)} Concept nodes.")

    # Dùng hard concept (concept_ids) để upload HAS_CONCEPT
    link_batch = []
    for v in videos:
        for scene_id, c_id in zip(v['scene_ids'], v['concept_ids']):
            link_batch.append({
                "scene_uid":  f"{v['video_name']}_scene_{scene_id}",
                "concept_id": f"concept_{c_id}",
            })

    CHUNK = 2000
    total = 0
    for i in range(0, len(link_batch), CHUNK):
        chunk = link_batch[i:i + CHUNK]
        with driver.session(database=NEO4J_DATABASE) as session:
            session.run("""
                UNWIND $batch AS item
                MATCH (s:Scene {uid: item.scene_uid})
                MATCH (c:Concept {id: item.concept_id})
                MERGE (s)-[:HAS_CONCEPT]->(c)
            """, batch=chunk)
        total += len(chunk)
    print(f"  Linked {total} Scene→Concept relationships.")


# ==========================================
# 7. EXTRACT CONCEPT TRIPLES (SOFT — TOP-P)
# ==========================================

def extract_concept_triples(videos: list, rst_type_to_idx: dict) -> list:
    """
    Extract concept triples từ RST links.

    Cải tiến: dùng TOP-P concept assignment (concept_ids_topk) thay vì chỉ
    hard concept (concept_ids). Với mỗi RST edge (src, tgt), tạo triple cho
    TỪNG CẶP (c_src, c_tgt) trong top-P × top-P của hai scene.

    Điều này tăng số triple từ N_edges lên N_edges × P² nhưng giúp:
        1. Coverage: scene ở ranh giới 2 concept không còn "chọn sai" 1 concept
        2. Richer prior: prior_scores phản ánh nhiều context hơn
        3. Triple sparsity giảm đáng kể (từ 1.1% → coverage cao hơn)

    Trọng số triple = proba(c_src) × proba(c_tgt) — lưu lại để weight MLP
    feature hoặc prior sau này. Hiện tại extract triples để train MLP.

    rst_links dùng 1-based index → trừ 1.
    """
    print("\n[Step 5] Extracting concept triples (soft TOP-P assignment)...")
    triples       = []
    unknown_rst   = set()
    n_edges_total = 0

    for v in videos:
        scene_ids = v['scene_ids']
        c_ids_topk = v.get('concept_ids_topk', [[c] for c in v['concept_ids']])
        c_proba    = v.get('concept_proba', [1.0] * len(scene_ids))
        label      = v['video_label']

        for src, tgt, rst_type in v['rst_links']:
            src_int = int(src) - 1
            tgt_int = int(tgt) - 1
            try:
                src_idx = scene_ids.index(src_int)
                tgt_idx = scene_ids.index(tgt_int)
            except ValueError:
                continue

            n_edges_total += 1
            rst_norm = normalize_rst_type(rst_type)
            if rst_norm not in rst_type_to_idx:
                unknown_rst.add(rst_norm)
                continue

            # TOP-P × TOP-P combinations
            for c_src in c_ids_topk[src_idx]:
                for c_tgt in c_ids_topk[tgt_idx]:
                    triples.append({
                        'c_src':       c_src,
                        'rst_type':    rst_norm,
                        'c_tgt':       c_tgt,
                        'video_label': label,
                    })

    print(f"  Total RST edges processed: {n_edges_total}")
    print(f"  Total concept triples extracted: {len(triples)}")
    print(f"  Expansion ratio: {len(triples)/max(n_edges_total,1):.1f}× (TOP-P effect)")

    if unknown_rst:
        print(f"  [WARNING] Out-of-vocab RST types (skipped): {sorted(unknown_rst)}")

    rst_counts = Counter(t['rst_type'] for t in triples)
    n_used = len(rst_counts)
    print(f"  RST types used: {n_used}/{len(rst_type_to_idx)} "
          + ", ".join(f"{k}:{v}" for k, v in rst_counts.most_common(6)))

    return triples


def compute_prior_scores(triples: list) -> dict:
    """Prior Laplace-smoothed: P(label=1 | c_src, rst_type, c_tgt)."""
    counts_pos   = defaultdict(int)
    counts_total = defaultdict(int)

    for t in triples:
        key = (t['c_src'], t['rst_type'], t['c_tgt'])
        counts_total[key] += 1
        if t['video_label'] == 1:
            counts_pos[key] += 1

    return {
        key: (counts_pos[key] + 1) / (total + 2)
        for key, total in counts_total.items()
    }


# ==========================================
# 8. MLP TRAINING
# ==========================================

def build_mlp_features(
    triples: list, centroids: np.ndarray,
    rst_type_to_idx: dict, prior_scores: dict,
) -> tuple:
    """
    Feature vector 24-dim (scalar-based):
        [cosine(centroid_src, centroid_tgt),   # (1,)
         one_hot(rst_type),                    # (20,)
         prior_score,                          # (1,)
         c_src / K,                            # (1,)
         c_tgt / K]                            # (1,)
    Total: 24-dim
    """
    K     = centroids.shape[0]
    n_rst = len(rst_type_to_idx)
    X_list, y_list = [], []

    for t in triples:
        c_src, c_tgt = t['c_src'], t['c_tgt']
        cos_sim  = float(np.dot(centroids[c_src], centroids[c_tgt]))

        rst_onehot = np.zeros(n_rst, dtype=np.float32)
        rst_onehot[rst_type_to_idx.get(t['rst_type'], 0)] = 1.0

        prior = prior_scores.get((c_src, t['rst_type'], c_tgt), 0.5)

        feat = np.concatenate([[cos_sim], rst_onehot, [prior], [c_src/K], [c_tgt/K]])
        X_list.append(feat)
        y_list.append(float(t['video_label']))

    return (
        torch.from_numpy(np.stack(X_list).astype(np.float32)),
        torch.tensor(y_list),
    )


def train_mlp_scorer(
    triples: list, centroids: np.ndarray,
    rst_type_to_idx: dict, prior_scores: dict,
    n_epochs: int = 30, batch_size: int = 512, lr: float = 1e-3,
) -> ConceptTripleScorer:
    """Train MLP scorer — với soft triples nhiều hơn nên batch_size lớn hơn."""
    print(f"\n[Step 6] Training MLP scorer ({n_epochs} epochs)...")

    X, y = build_mlp_features(triples, centroids, rst_type_to_idx, prior_scores)
    print(f"  Feature matrix: {X.shape} | Labels: {y.shape} "
          f"| Positive rate: {y.mean():.3f}")

    n     = len(X)
    idx   = torch.randperm(n)
    split = int(n * 0.85)
    X_tr, y_tr = X[idx[:split]], y[idx[:split]]
    X_va, y_va = X[idx[split:]], y[idx[split:]]

    model     = ConceptTripleScorer(feature_dim=X.shape[1], hidden_dim=32)
    optim     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    best_val_loss, best_state = float('inf'), None

    for epoch in range(n_epochs):
        model.train()
        perm    = torch.randperm(len(X_tr))
        ep_loss, steps = 0.0, 0

        for i in range(0, len(X_tr), batch_size):
            xb = X_tr[perm[i:i + batch_size]]
            yb = y_tr[perm[i:i + batch_size]]
            loss = criterion(model(xb), yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            ep_loss += loss.item()
            steps   += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va)
            val_loss = criterion(val_pred, y_va).item()
            val_acc  = ((val_pred > 0.5).float() == y_va).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs} | "
                  f"train_loss={ep_loss/steps:.4f} | "
                  f"val_loss={val_loss:.4f} | val_acc={val_acc:.3f}")

    model.load_state_dict(best_state)
    print(f"  Best val_loss: {best_val_loss:.4f}")
    return model


# ==========================================
# 9. MAIN
# ==========================================

def main(args: argparse.Namespace) -> None:
    data_root  = Path(args.data_root)
    split_file = Path(args.split_file)

    # Tất cả artifacts ConceptRAG lưu vào <output_dir>
    # để tách biệt với các index khác (edge_index, scene_index...)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] ConceptRAG artifacts will be saved to: {output_dir}")

    centroids_path    = output_dir / "centroids.pt"
    gmm_path          = output_dir / "gmm.pkl"
    pca_path          = output_dir / "pca.pkl"
    proba_path        = output_dir / "proba_matrix.npy"
    concept_dict_path = output_dir / "concept_dict.json"
    rst_vocab_path    = output_dir / "rst_vocab.json"
    mlp_path          = output_dir / "mlp_scorer.pt"
    log_dir           = output_dir / "concept_naming_logs"

    # RST vocab cố định — lưu trước toàn bộ các bước
    rst_type_to_idx = build_fixed_rst_vocab()
    with open(rst_vocab_path, 'w') as f:
        json.dump(rst_type_to_idx, f, indent=2)
    print(f"[INFO] Fixed RST vocab ({len(rst_type_to_idx)} types) saved → {rst_vocab_path}")

    # Load data
    videos = load_split_videos(data_root, split_file)

    # ====================== STEP 1: GMM (thay K-Means) ======================
    if args.skip_kmeans and all(p.exists() for p in [gmm_path, pca_path, centroids_path, proba_path]):
        print("[SKIP] GMM + PCA (loading from cache)")
        centroids = torch.load(centroids_path, map_location="cpu").numpy()
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        with open(gmm_path, 'rb') as f:
            gmm = pickle.load(f)
        proba_matrix = np.load(proba_path)
    else:
        gmm, pca, centroids, proba_matrix = run_gmm(
            videos, args.n_concepts, pca_dim=args.pca_dim
        )
        torch.save(torch.from_numpy(centroids), centroids_path)
        with open(pca_path, 'wb') as f:
            pickle.dump(pca, f)
        with open(gmm_path, 'wb') as f:
            pickle.dump(gmm, f)
        np.save(proba_path, proba_matrix)
        print(f"  Saved GMM artifacts → {output_dir}")

    # ====================== STEP 2: Soft Concept Assignment ======================
    print(f"\n[Step 2] Assigning concept IDs (soft TOP-{args.top_p})...")
    videos = assign_concept_ids_soft(videos, proba_matrix, top_p=args.top_p)
    evaluate_per_video_diversity(videos)

    # ====================== STEP 3: Qwen Naming ======================
    if args.skip_naming and concept_dict_path.exists():
        print("[SKIP] Qwen concept naming")
        with open(concept_dict_path, 'r', encoding='utf-8') as f:
            concept_dict = {int(k): v for k, v in json.load(f).items()}
    else:
        print(f"\n[INFO] Loading Qwen3-4B-Instruct from: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        llm = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
        )
        llm.eval()

        concept_dict = build_concept_dictionary(
            videos, args.n_concepts, args.top_n_captions,
            tokenizer, llm, log_dir=log_dir,
        )
        with open(concept_dict_path, 'w', encoding='utf-8') as f:
            json.dump(concept_dict, f, ensure_ascii=False, indent=2)
        print(f"  Saved concept_dict → {concept_dict_path}")

        del llm, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # ====================== STEP 4: Neo4j Upload ======================
    if not args.skip_upload:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()

        if args.force:
            print("=" * 60)
            print("WARNING: --force flag is ACTIVE. Removing existing Concept nodes...")
            print("=" * 60)
            with driver.session(database=NEO4J_DATABASE) as session:
                session.execute_write(clear_concepts)

        upload_concept_graph(driver, videos, concept_dict)
        driver.close()
    else:
        print("[SKIP] Neo4j upload")

    # ====================== STEP 5 & 6: Triples + MLP ======================
    if args.skip_triples:
        print("[SKIP] Extract concept triples")
    else:
        triples      = extract_concept_triples(videos, rst_type_to_idx)
        prior_scores = compute_prior_scores(triples)

        if not args.skip_mlp:
            mlp = train_mlp_scorer(
                triples, centroids, rst_type_to_idx, prior_scores,
                n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr,
            )
            n_rst       = len(rst_type_to_idx)
            feature_dim = n_rst + 4   # cosine + one_hot + prior + c_src/K + c_tgt/K

            torch.save({
                'model_state_dict': mlp.state_dict(),
                'feature_dim':      feature_dim,
                'n_rst_types':      n_rst,
                'hidden_dim':       32,
                'rst_type_to_idx':  rst_type_to_idx,
                'prior_scores':     {str(k): v for k, v in prior_scores.items()},
                'top_p':            args.top_p,
            }, mlp_path)
            print(f"  Saved MLP scorer → {mlp_path}")
        else:
            print("[SKIP] MLP training")

    print(f"\n[DONE] Concept index processing completed.")
    print(f"  All artifacts in: {output_dir}")
    print(f"  Files: centroids.pt | gmm.pkl | pca.pkl | proba_matrix.npy "
          f"| concept_dict.json | rst_vocab.json | mlp_scorer.pt")


# ==========================================
# 10. ENTRY POINT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build ConceptRAG index — GMM-based soft clustering (Phase 1 offline)."
    )
    parser.add_argument("--data_root",       type=str, required=True)
    parser.add_argument("--split_file",      type=str, required=True)
    parser.add_argument("--output_dir",      type=str, required=True,
                        help="Root indexing dir. Artifacts saved to <output_dir>/+")

    parser.add_argument("--skip_kmeans",     action="store_true",
                        help="Skip GMM + PCA (load from <output_dir>/ cache)")
    parser.add_argument("--skip_naming",     action="store_true")
    parser.add_argument("--skip_upload",     action="store_true")
    parser.add_argument("--skip_triples",    action="store_true")
    parser.add_argument("--skip_mlp",        action="store_true")

    parser.add_argument("--pca_dim",         type=int,   default=64,
                        help="PCA target dim before GMM. Default 64 (lower than K-Means "
                             "because GMM covariance is O(K×D²)).")
    parser.add_argument("--n_concepts",      type=int,   default=150)
    parser.add_argument("--top_p",           type=int,   default=2,
                        help="Number of soft concepts per scene for triple expansion. "
                             "Default 2 (each edge → up to 4 triples).")
    parser.add_argument("--top_n_captions",  type=int,   default=10)
    parser.add_argument("--model_name",      type=str,   default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--n_epochs",        type=int,   default=30)
    parser.add_argument("--batch_size",      type=int,   default=512)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--force",           action="store_true",
                        help="Drop & recreate Concept nodes in Neo4j and rebuild GMM.")

    args = parser.parse_args()
    main(args)