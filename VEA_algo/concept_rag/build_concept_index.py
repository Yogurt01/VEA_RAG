"""
build_concept_index.py
----------------------
Phase 1 — Offline build (chạy một lần trên train+val).

Pipeline:
    Step 1 : K-Means trên tất cả scene embeddings → K centroids
    Step 2 : Gemma local đặt tên concept từ Top-N captions gần centroid
    Step 3 : Assign concept_id cho từng scene (cosine với centroids)
    Step 4 : Upload (:Scene)-[:HAS_CONCEPT]->(:Concept) lên Neo4j
    Step 5 : Extract concept triples (c_src, RST_type, c_tgt, video_label)
    Step 6 : Train MLP scorer — video_label 0/1 làm supervision signal
    Step 7 : Lưu centroids.pt, concept_dict.json, mlp_scorer.pt

Usage:
    python build_concept_index.py \
        --data_root      /path/to/All_Videos \
        --split_file     /path/to/dataset_splits.json \
        --output_dir     /path/to/concept_index \
        --n_concepts     200 \
        --top_n_captions 10 \
        [--model_name    /path/to/gemma-4-E2B-it] \
        [--force]

Environment variables (Neo4j):
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE
"""

import os
import json
import argparse
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize
from neo4j import GraphDatabase
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForImageTextToText

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


# ==========================================
# 1. MLP SCORER DEFINITION
# ==========================================

class ConceptTripleScorer(nn.Module):
    """
    Lightweight MLP scorer — mượn ý tưởng triple-scoring từ SubgraphRAG.

    Input features cho mỗi triple (c_src, RST_type, c_tgt):
        - centroid embedding của c_src  : (D,)  = 2048
        - centroid embedding của c_tgt  : (D,)  = 2048
        - one-hot của RST_type          : (R,)
        - prior score (tần suất engaging): (1,)
    Total input dim = 2*D + R + 1

    Output: P(triple thuộc video engaging) ∈ [0, 1]
    """
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
# 2. DATA LOADING
# ==========================================

def load_split_videos(data_root: Path, split_file: Path) -> list[dict]:
    """
    Load tất cả video trong train+val split.
    Trả về list[dict] với keys: video_name, video_label, embeddings, scene_ids,
    rst_links, captions.
    """
    with open(split_file, 'r') as f:
        splits = json.load(f)
    allowed = set(splits.get("train", [])) | set(splits.get("val", []))
    print(f"[INFO] Found {len(allowed)} video IDs in train+val split.")

    videos = []
    skipped = 0

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

            # Đọc captions theo thứ tự scene_ids
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
                'embeddings':  data['embeddings'],   # Tensor (T, 2048)
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
# 3. K-MEANS CLUSTERING
# ==========================================

def run_kmeans(videos: list[dict], n_concepts: int, pca_dim: int = 128) -> tuple[np.ndarray, IncrementalPCA]:
    """
    Gom toàn bộ scene embeddings, giảm chiều bằng IncrementalPCA,
    sau đó chạy MiniBatchKMeans trong không gian thấp chiều.

    Trả về:
        centroids_full : (n_concepts, 2048) — centroid trong không gian gốc
                         (dùng để assign concept cho scene mới bằng cosine)
        pca            : fitted IncrementalPCA object — lưu lại để transform
                         scene embedding của video test khi inference
    """
    print(f"\n[Step 1] Running PCA (2048 → {pca_dim}) + K-Means (K={n_concepts})...")

    all_embs = []
    for v in videos:
        embs = v['embeddings'].numpy().astype(np.float32)
        all_embs.append(embs)

    X = np.vstack(all_embs)     # (N_total_scenes, 2048)
    print(f"  Total scenes: {X.shape[0]}")

    # L2-normalize trước PCA
    X_norm = normalize(X, norm='l2')

    # IncrementalPCA để tránh OOM với dataset lớn
    CHUNK = 4096
    pca   = IncrementalPCA(n_components=pca_dim)
    for i in range(0, len(X_norm), CHUNK):
        pca.partial_fit(X_norm[i:i + CHUNK])

    X_pca = pca.transform(X_norm)      # (N, pca_dim)
    X_pca = normalize(X_pca, norm='l2')
    print(f"  PCA done. Explained variance ratio sum: "
          f"{pca.explained_variance_ratio_.sum():.3f}")

    # K-Means trong không gian PCA
    km = MiniBatchKMeans(
        n_clusters=n_concepts,
        batch_size=min(4096, X_pca.shape[0]),
        n_init=5,
        random_state=42,
        verbose=0,
    )
    km.fit(X_pca)
    print(f"  K-Means done.")

    # Ánh xạ centroids ngược về không gian 2048-dim gốc
    # bằng cách tìm mean embedding của tất cả scene thuộc cụm đó
    labels          = km.labels_                     # (N,)
    centroids_full  = np.zeros((n_concepts, 2048), dtype=np.float32)
    counts          = np.zeros(n_concepts, dtype=np.int32)

    for i, lbl in enumerate(labels):
        centroids_full[lbl] += X_norm[i]
        counts[lbl]         += 1

    # Tránh chia cho 0 nếu có cụm rỗng
    counts = np.maximum(counts, 1)
    centroids_full /= counts[:, None]
    centroids_full  = normalize(centroids_full, norm='l2')

    print(f"  Centroid matrix (full-dim): {centroids_full.shape}")

    from collections import Counter
    counts = Counter(labels)
    print(f"  Concept sizes: min={min(counts.values())}, max={max(counts.values())}, avg={sum(counts.values())/len(counts):.1f}")
    print(f"  Number of empty clusters: {sum(1 for c in counts.values() if c == 0)}")

    return centroids_full, pca


# ==========================================
# 4. CONCEPT NAMING VIA GEMMA
# ==========================================

def assign_concept_ids(videos: list[dict], centroids: np.ndarray) -> list[dict]:
    """
    Với mỗi scene, tìm concept_id gần nhất (cosine) từ centroids.
    Thêm key 'concept_ids' vào mỗi video dict.
    """
    centroids_t = torch.from_numpy(centroids)   # (K, 2048)

    for v in videos:
        embs    = F.normalize(v['embeddings'].float(), p=2, dim=1)   # (T, 2048)
        scores  = embs @ centroids_t.T                               # (T, K)
        c_ids   = scores.argmax(dim=1).tolist()                      # (T,)
        v['concept_ids'] = c_ids

    return videos


def get_top_captions_per_concept(
    videos: list[dict],
    centroids: np.ndarray,
    top_n: int,
) -> dict[int, list[str]]:
    """
    Với mỗi concept, tìm top-N captions của scene gần centroid nhất.
    Trả về dict {concept_id: [caption1, caption2, ...]}.
    """
    K = centroids.shape[0]
    centroids_t = torch.from_numpy(centroids)

    # (concept_id) → list of (cosine_score, caption)
    buckets: dict[int, list] = {k: [] for k in range(K)}

    for v in videos:
        embs  = F.normalize(v['embeddings'].float(), p=2, dim=1)
        scores = (embs @ centroids_t.T)   # (T, K)

        for t, (c_id, cap) in enumerate(zip(v['concept_ids'], v['captions'])):
            if cap:
                cosine_score = scores[t, c_id].item()
                buckets[c_id].append((cosine_score, cap))

    top_captions = {}
    for c_id, items in buckets.items():
        items.sort(key=lambda x: x[0], reverse=True)
        top_captions[c_id] = [cap for _, cap in items[:top_n]]

    return top_captions


def name_concept_with_gemma(
    concept_id: int,
    captions: list[str],
    processor,
    model,
    output_dir: Path = None,   # Thêm để lưu log
) -> dict:
    """
    Dùng Gemma để đặt tên concept + log chi tiết để debug.
    """
    if not captions:
        return {
            "concept_id": f"concept_{concept_id}",
            "concept_name": f"Unknown concept {concept_id}",
            "definition": "No captions available.",
        }

    caption_block = "\n".join(f"- {c}" for c in captions[:10])  # log tối đa 10 captions

    prompt = f"""Below are {len(captions)} video scene descriptions that belong to the same visual-semantic cluster.

Scene descriptions:
{caption_block}

Based on these descriptions, provide a short concept label and definition.
Respond ONLY with a valid JSON object, no markdown, no explanation:
{{"concept_name": "2-4 word label", "definition": "One sentence describing the visual or narrative pattern."}}"""

    # === LOG PROMPT VÀ CAPTIONS ===
    log_file = output_dir / f"concept_{concept_id:03d}_log.txt" if output_dir else None
    if log_file:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== CONCEPT {concept_id} ===\n\n")
            f.write("TOP CAPTIONS:\n")
            f.write(caption_block + "\n\n")
            f.write("PROMPT SENT TO GEMMA:\n")
            f.write("-" * 80 + "\n")
            f.write(prompt + "\n")
            f.write("-" * 80 + "\n\n")

    messages = [{"role": "user", "content": prompt}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128)

    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    # === LOG RESPONSE ===
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"GEMMA RESPONSE:\n{response}\n\n")

    # Parse JSON
    import re
    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {}
        result = {
            "concept_id":   f"concept_{concept_id}",
            "concept_name": parsed.get("concept_name", f"Concept {concept_id}"),
            "definition":   parsed.get("definition",   ""),
        }
    except Exception:
        result = {
            "concept_id":   f"concept_{concept_id}",
            "concept_name": f"Concept {concept_id}",
            "definition":   response[:200],
        }

    # Log final result
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"FINAL PARSED:\n")
            f.write(json.dumps(result, ensure_ascii=False, indent=2))

    print(f"  Concept {concept_id:3d} → {result['concept_name']}")
    return result


def build_concept_dictionary(
    videos: list[dict],
    centroids: np.ndarray,
    top_n: int,
    processor,
    model,
    output_dir: Path,
) -> dict[int, dict]:
    """
    Chạy Gemma naming cho tất cả K concepts.
    Trả về dict {concept_id (int): {concept_id, concept_name, definition}}.
    """
    print(f"\n[Step 3] Naming {centroids.shape[0]} concepts with Gemma...")

    top_captions = get_top_captions_per_concept(videos, centroids, top_n)
    concept_dict = {}

    log_dir = output_dir / "concept_naming_logs"
    log_dir.mkdir(exist_ok=True)

    for c_id in range(centroids.shape[0]):
        caps = top_captions.get(c_id, [])
        info = name_concept_with_gemma(c_id, caps, processor, model, output_dir=log_dir)
        concept_dict[c_id] = info

        if (c_id + 1) % 10 == 0:
            print(f"  Named {c_id + 1}/{centroids.shape[0]} concepts...")

    print(f"  Done naming {len(concept_dict)} concepts.")
    print(f"  Logs saved to: {log_dir}")
    return concept_dict


# ==========================================
# 5. NEO4J UPLOAD
# ==========================================

def clear_concepts(tx):
    """
    Xóa tất cả Concept nodes và HAS_CONCEPT relationships,
    nhưng giữ nguyên Video, Scene, RST.
    """
    tx.run("""
        MATCH (c:Concept)
        OPTIONAL MATCH (c)<-[r:HAS_CONCEPT]-()
        DETACH DELETE c
    """)
    print("All Concept nodes and HAS_CONCEPT relationships removed.")


def upload_concept_graph(driver, videos: list[dict], concept_dict: dict[int, dict]) -> None:
    """
    Upload lên Neo4j:
        (:Concept {id, name, definition}) nodes
        (:Scene)-[:HAS_CONCEPT]->(:Concept) relationships
    """
    print("\n[Step 4] Uploading concept graph to Neo4j...")

    # Upload Concept nodes
    concept_batch = [
        {"id": info["concept_id"], "name": info["concept_name"], "definition": info["definition"]}
        for info in concept_dict.values()
    ]
    with driver.session(database=NEO4J_DATABASE) as session:
        session.run("""
            UNWIND $batch AS c
            MERGE (con:Concept {id: c.id})
            SET con.name       = c.name,
                con.definition = c.definition
        """, batch=concept_batch)
    print(f"  Uploaded {len(concept_batch)} Concept nodes.")

    # Upload HAS_CONCEPT relationships theo batch
    link_batch = []
    for v in videos:
        for scene_id, c_id in zip(v['scene_ids'], v['concept_ids']):
            scene_uid   = f"{v['video_name']}_scene_{scene_id}"
            concept_str = f"concept_{c_id}"
            link_batch.append({"scene_uid": scene_uid, "concept_id": concept_str})

    CHUNK = 2000
    total_linked = 0
    for i in range(0, len(link_batch), CHUNK):
        chunk = link_batch[i:i + CHUNK]
        with driver.session(database=NEO4J_DATABASE) as session:
            session.run("""
                UNWIND $batch AS item
                MATCH (s:Scene {uid: item.scene_uid})
                MATCH (c:Concept {id: item.concept_id})
                MERGE (s)-[:HAS_CONCEPT]->(c)
            """, batch=chunk)
        total_linked += len(chunk)

    print(f"  Linked {total_linked} Scene→Concept relationships.")


# ==========================================
# 6. EXTRACT CONCEPT TRIPLES
# ==========================================

def extract_concept_triples(videos: list[dict]) -> list[dict]:
    """
    Với mỗi RST edge (src_scene, RST_type, tgt_scene) trong một video,
    tạo concept triple:
        (concept_id[src], RST_type, concept_id[tgt], video_label)

    Đây là tập training data cho MLP scorer.
    """
    print("\n[Step 5] Extracting concept triples...")
    triples = []

    for v in videos:
        c_ids  = v['concept_ids']
        label  = v['video_label']

        for src, tgt, rst_type in v['rst_links']:
            src_int = int(src) - 1
            tgt_int = int(tgt) - 1

            # scene_ids là list[int], tìm index trong list đó
            try:
                src_idx = v['scene_ids'].index(src_int)
                tgt_idx = v['scene_ids'].index(tgt_int)
            except ValueError:
                continue

            triples.append({
                'c_src':       c_ids[src_idx],
                'rst_type':    str(rst_type).strip().upper().replace(" ", "_").replace("-", "_"),
                'c_tgt':       c_ids[tgt_idx],
                'video_label': label,
            })

    print(f"  Total concept triples extracted: {len(triples)}")

    # Thống kê nhanh
    from collections import Counter
    rst_counts = Counter(t['rst_type'] for t in triples)
    print(f"  RST types seen ({len(rst_counts)} unique): "
          + ", ".join(f"{k}:{v}" for k, v in rst_counts.most_common(8)))

    return triples


def compute_prior_scores(triples: list[dict]) -> dict[tuple, float]:
    """
    Tần suất Bayesian prior:
        prior(c_src, rst_type, c_tgt) = count(label=1) / count(total)
    Dùng làm feature bổ sung cho MLP.
    """
    from collections import defaultdict
    counts_pos   = defaultdict(int)
    counts_total = defaultdict(int)

    for t in triples:
        key = (t['c_src'], t['rst_type'], t['c_tgt'])
        counts_total[key] += 1
        if t['video_label'] == 1:
            counts_pos[key] += 1

    prior = {}
    for key, total in counts_total.items():
        # Laplace smoothing
        prior[key] = (counts_pos[key] + 1) / (total + 2)

    return prior


# ==========================================
# 7. MLP TRAINING
# ==========================================

def build_mlp_features(
    triples: list[dict],
    centroids: np.ndarray,
    rst_type_to_idx: dict[str, int],
    prior_scores: dict[tuple, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Xây dựng feature matrix X và label vector y để train MLP.

    X[i] = [centroid(c_src) || centroid(c_tgt) || one_hot(rst_type) || prior_score]
    y[i] = video_label (0 hoặc 1)
    """
    n_rst = len(rst_type_to_idx)
    D     = centroids.shape[1]

    X_list = []
    y_list = []

    for t in triples:
        c_src     = t['c_src']
        c_tgt     = t['c_tgt']
        rst_type  = t['rst_type']
        rst_idx   = rst_type_to_idx.get(rst_type, 0)

        emb_src   = centroids[c_src]                        # (2048,)
        emb_tgt   = centroids[c_tgt]                        # (2048,)
        rst_onehot = np.zeros(n_rst, dtype=np.float32)
        rst_onehot[rst_idx] = 1.0

        prior     = prior_scores.get((c_src, rst_type, c_tgt), 0.5)
        prior_arr = np.array([prior], dtype=np.float32)

        feat = np.concatenate([emb_src, emb_tgt, rst_onehot, prior_arr])
        X_list.append(feat)
        y_list.append(float(t['video_label']))

    X = torch.from_numpy(np.stack(X_list))
    y = torch.tensor(y_list)

    return X, y


def train_mlp_scorer(
    triples: list[dict],
    centroids: np.ndarray,
    rst_type_to_idx: dict[str, int],
    prior_scores: dict[tuple, float],
    n_epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
) -> ConceptTripleScorer:
    """
    Train MLP scorer với video_label 0/1 làm supervision.
    """
    print(f"\n[Step 6] Training MLP scorer ({n_epochs} epochs)...")

    X, y = build_mlp_features(triples, centroids, rst_type_to_idx, prior_scores)
    print(f"  Feature matrix: {X.shape} | Labels: {y.shape} "
          f"| Positive rate: {y.mean():.3f}")

    # Train/val split 85/15
    n     = len(X)
    idx   = torch.randperm(n)
    split = int(n * 0.85)
    X_tr, y_tr = X[idx[:split]], y[idx[:split]]
    X_va, y_va = X[idx[split:]], y[idx[split:]]

    n_rst  = len(rst_type_to_idx)
    D      = centroids.shape[1]
    model  = ConceptTripleScorer(embed_dim=D, n_rst_types=n_rst, hidden_dim=256)
    optim  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    best_state    = None

    for epoch in range(n_epochs):
        model.train()
        perm    = torch.randperm(len(X_tr))
        ep_loss = 0.0
        steps   = 0

        for i in range(0, len(X_tr), batch_size):
            batch_idx = perm[i:i + batch_size]
            xb = X_tr[batch_idx]
            yb = y_tr[batch_idx]
            pred = model(xb)
            loss = criterion(pred, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            ep_loss += loss.item()
            steps   += 1

        # Validation
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


def main(args: argparse.Namespace) -> None:
    data_root  = Path(args.data_root)
    split_file = Path(args.split_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    centroids_path    = output_dir / "centroids.pt"
    pca_path          = output_dir / "pca.pkl"
    concept_dict_path = output_dir / "concept_dict.json"
    rst_vocab_path    = output_dir / "rst_vocab.json"
    mlp_path          = output_dir / "mlp_scorer.pt"

    # if args.force:
    #     # Xóa file output cũ để rebuild
    #     for f in [centroids_path, concept_dict_path, rst_vocab_path, mlp_path]:
    #         if f.exists():
    #             f.unlink()
    #             print(f"Removed old file: {f}")
    #     # Xóa log cũ
    #     log_dir = output_dir / "concept_naming_logs"
    #     if log_dir.exists():
    #         import shutil
    #         shutil.rmtree(log_dir)
    #         print("Removed old concept naming logs.")
            
    # Load data
    videos = load_split_videos(data_root, split_file)
    for v in videos[:1]:
        print(v['scene_ids'])
        print(v['rst_links'])

    # ====================== STEP 1: K-Means + PCA ======================
    if args.skip_kmeans and centroids_path.exists() and pca_path.exists():
        print("[SKIP] K-Means + PCA")
        centroids = torch.load(centroids_path, map_location="cpu").numpy()
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
    else:
        centroids, pca = run_kmeans(videos, args.n_concepts, pca_dim=args.pca_dim)
        torch.save(torch.from_numpy(centroids), centroids_path)
        with open(pca_path, 'wb') as f:
            pickle.dump(pca, f)
        print(f"  Saved centroids & pca → {output_dir}")

    # ====================== STEP 2: Assign Concept IDs ======================
    print("\n[Step 2] Assigning concept IDs to scenes...")
    videos = assign_concept_ids(videos, centroids)

    # ====================== STEP 3: Gemma Naming ======================
    if args.skip_naming and concept_dict_path.exists():
        print("[SKIP] Gemma concept naming")
        with open(concept_dict_path, 'r', encoding='utf-8') as f:
            concept_dict_raw = json.load(f)
        concept_dict = {int(k): v for k, v in concept_dict_raw.items()}
    else:
        print(f"\n[INFO] Loading Gemma from: {args.model_name}")
        processor = AutoProcessor.from_pretrained(args.model_name)
        llm = AutoModelForImageTextToText.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        llm.eval()

        concept_dict = build_concept_dictionary(
            videos, centroids, args.top_n_captions, processor, llm, output_dir=output_dir
        )

        with open(concept_dict_path, 'w', encoding='utf-8') as f:
            json.dump(concept_dict, f, ensure_ascii=False, indent=2)
        print(f"  Saved concept_dict → {concept_dict_path}")

        del llm, processor
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # ====================== STEP 4: Neo4j Upload ======================
    if not args.skip_upload:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()

        if args.force:
            # Xóa Concept cũ và relationships, giữ nguyên Video, Scene, RST
            with driver.session(database=NEO4J_DATABASE) as session:
                print("=" * 60)
                print("WARNING: --force flag is ACTIVE. Removing existing Concept nodes...")
                print("=" * 60)
                session.execute_write(clear_concepts)

        upload_concept_graph(driver, videos, concept_dict)
        driver.close()
    else:
        print("[SKIP] Neo4j upload")

    # ====================== STEP 5: Triples + MLP ======================
    if args.skip_triples and rst_vocab_path.exists():
        print("[SKIP] Extract concept triples")
        with open(rst_vocab_path, 'r') as f:
            rst_type_to_idx = json.load(f)
        # Load triples if needed...
    else:
        triples = extract_concept_triples(videos)

        rst_types = sorted(set(t['rst_type'] for t in triples))
        rst_type_to_idx = {r: i for i, r in enumerate(rst_types)}
        with open(rst_vocab_path, 'w') as f:
            json.dump(rst_type_to_idx, f, indent=2)

        prior_scores = compute_prior_scores(triples)

        if not args.skip_mlp:
            mlp = train_mlp_scorer(
                triples, centroids, rst_type_to_idx, prior_scores,
                n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr
            )

            torch.save({
                'model_state_dict': mlp.state_dict(),
                'embed_dim': centroids.shape[1],
                'n_rst_types': len(rst_type_to_idx),
                'hidden_dim': 256,
                'rst_type_to_idx': rst_type_to_idx,
                'prior_scores': {str(k): v for k, v in prior_scores.items()},
            }, mlp_path)
            print(f"  Saved MLP scorer → {mlp_path}")
        else:
            print("[SKIP] MLP training")

    print("\n[DONE] Concept index processing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build concept index for ConceptRAG — offline phase."
    )
    parser.add_argument("--data_root",      type=str, required=True)
    parser.add_argument("--split_file",     type=str, required=True)
    parser.add_argument("--output_dir",     type=str, required=True)
    
    parser.add_argument("--skip_kmeans",    action="store_true", help="Skip K-Means + PCA")
    parser.add_argument("--skip_naming",    action="store_true", help="Skip Gemma concept naming")
    parser.add_argument("--skip_upload",    action="store_true", help="Skip Neo4j upload")
    parser.add_argument("--skip_triples",   action="store_true", help="Skip extracting concept triples")
    parser.add_argument("--skip_mlp",       action="store_true", help="Skip MLP training")
    
    parser.add_argument("--pca_dim",        type=int, default=128)
    parser.add_argument("--n_concepts",     type=int,   default=200)
    parser.add_argument("--top_n_captions", type=int,   default=10)
    parser.add_argument("--model_name",     type=str,   default="google/gemma-4-E2B-it")
    parser.add_argument("--n_epochs",       type=int,   default=30)
    parser.add_argument("--batch_size",     type=int,   default=512)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--force",          action="store_true")

    args = parser.parse_args()
    main(args)