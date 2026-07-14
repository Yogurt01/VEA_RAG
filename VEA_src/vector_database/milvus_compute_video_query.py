import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Dict, Any


script_dir = Path(__file__).resolve().parent
multimodal_to_text_dir = script_dir.parent / "multimodal_to_text"
if multimodal_to_text_dir.exists():
    sys.path.insert(0, str(multimodal_to_text_dir))
else:
    sys.path.insert(0, str(script_dir))

try:
    from qwen3_vl_embedding import Qwen3VLEmbedder
except ImportError as e:
    print(f"[ERROR] Cannot import Qwen3VLEmbedder: {e}")
    print("Please ensure qwen3_vl_embedding.py is in the 'multimodal_to_text' folder.")
    raise


# ==========================================
# 1. CONSTANTS & CONFIG
# ==========================================

REQUIRED_DATA_KEYS = [
    'embeddings', 'scene_ids', 'metadata',
    'edge_index', 'edge_attr', 'rst_links', 'y',
]


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def load_scene_captions(folder_name: str, data_root: Path) -> List[str]:
    """
    Read captions from segments.json in the order of scene_ids.
    Returns a list of caption strings; missing captions are replaced with empty strings.
    """
    seg_path = data_root / folder_name / "segments.json"
    if not seg_path.exists():
        print(f"  [WARNING] segments.json not found for {folder_name}")
        return []

    try:
        with open(seg_path, 'r', encoding='utf-8') as f:
            segments_content = json.load(f)
    except Exception as e:
        print(f"  [WARNING] Cannot parse segments.json for {folder_name}: {e}")
        return []

    if isinstance(segments_content, list):
        return [seg.get('caption', '') for seg in segments_content]

    if isinstance(segments_content, dict):
        captions = []
        for k in sorted(segments_content.keys(), key=lambda x: int(x) if x.isdigit() else 0):
            entry = segments_content[k]
            if isinstance(entry, dict):
                captions.append(entry.get('caption', ''))
            else:
                captions.append(str(entry))
        return captions

    return []


@torch.no_grad()
def cross_modal_self_query(
    scene_visual_embeddings: torch.Tensor,
    scene_captions: List[str],
    embedder,
    temperature: float = 0.01,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a representative vector V_video ∈ R^2048 via query-weighted aggregation.

    Steps:
        1. q = mean( TextEncode(caption_t) for t in T )
        2. s_t = dot(v_t, q)                           
        3. α_t = softmax(s_t / τ)                      
        4. V_video = Σ α_t · v_t                       

    Note: We use dot product (not cosine) as the similarity measure. If you need cosine,
          normalize both v_t and q before computing the dot product.
    """
    T = scene_visual_embeddings.shape[0]
    if len(scene_captions) != T:
        print(f"  [WARNING] Captions count ({len(scene_captions)}) != Embeddings count ({T}). Adjusting...")
        if len(scene_captions) > T:
            scene_captions = scene_captions[:T]
        else:
            scene_captions = scene_captions + [""] * (T - len(scene_captions))

    if T == 0:
        raise ValueError("No scenes to process.")

    if device is None:
        device = scene_visual_embeddings.device

    V = scene_visual_embeddings.to(device).float()   # (T, 2048)

    valid_indices = [i for i, cap in enumerate(scene_captions) if cap.strip()]
    if not valid_indices:
        print("  [WARNING] All captions are empty. Using mean of visual embeddings as query.")
        return V.mean(dim=0)  # (2048,)

    valid_captions = [scene_captions[i] for i in valid_indices]
    text_inputs = [{"text": cap} for cap in valid_captions]

    try:
        raw = embedder.process(text_inputs)
    except Exception as e:
        print(f"  [ERROR] embedder.process failed: {e}")
        return V.mean(dim=0)

    if isinstance(raw, list):
        text_embs = torch.stack([e.cpu().float() for e in raw]).to(device)
    else:
        text_embs = raw.to(device).float()

    q = text_embs.mean(dim=0)  # (2048,)

    scores = torch.matmul(V, q)              # (T,)

    weights = F.softmax(scores / temperature, dim=0)  # (T,)

    V_video = (weights.unsqueeze(1) * V).sum(dim=0)   # (2048,)

    return V_video


def load_valid_videos(
    data_root: Path,
    split_file: Path,
    split_key: str = "test",
) -> tuple[List[str], Dict[str, Any], List[tuple]]:
    """
    Load the list of valid videos from the split file.
    split_key: 'test' (default) or 'val' (used when computing representations for the
               validation set for hyperparameter sweeping without tuning on test).
    """
    with open(split_file, 'r') as f:
        splits = json.load(f)
    target_folders = splits.get(split_key, [])
    if not target_folders:
        print(f"[WARNING] split_file has no entries under key '{split_key}'.")

    valid_folders: List[str] = []
    valid_data: Dict[str, Any] = {}
    invalid_folders: List[tuple] = []

    for folder_name in target_folders:
        folder_path = data_root / folder_name
        emb_path = folder_path / "scene_embeddings.pt"
        seg_path = folder_path / "segments.json"

        if not emb_path.exists():
            invalid_folders.append((folder_name, "Missing scene_embeddings.pt"))
            continue
        if not seg_path.exists():
            invalid_folders.append((folder_name, "Missing segments.json"))
            continue

        try:
            data = torch.load(emb_path, map_location='cpu')
            missing_keys = [k for k in REQUIRED_DATA_KEYS if k not in data]
            if missing_keys:
                invalid_folders.append((folder_name, f"Missing keys: {missing_keys}"))
                continue
            valid_folders.append(folder_name)
            valid_data[folder_name] = data
        except Exception as e:
            invalid_folders.append((folder_name, f"Corrupted .pt: {e}"))

    return valid_folders, valid_data, invalid_folders


# ==========================================
# 3. MAIN
# ==========================================

def main(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    split_file = Path(args.split_file)
    output_dir = Path(args.output_dir)
    device = torch.device(args.device)

    if not data_root.is_dir():
        raise ValueError(f"data_root not found: {data_root}")
    if not split_file.exists():
        raise ValueError(f"split_file not found: {split_file}")

    # 1. Load Qwen3VLEmbedder
    print(f"[INFO] Loading Qwen3VLEmbedder from: {args.embedding_model_name}")
    embedder = Qwen3VLEmbedder(
        model_name_or_path=args.embedding_model_name,
        torch_dtype=torch.bfloat16,
    )
    print("[INFO] Embedder loaded.\n")

    # 2. Load valid videos
    print(f"[INFO] split_key = '{args.split_key}'")
    valid_folders, valid_data, invalid_folders = load_valid_videos(data_root, split_file, args.split_key)
    print(f"Validation: {len(valid_folders)} valid, {len(invalid_folders)} invalid.")
    for f, reason in invalid_folders[:5]:
        print(f"  [INVALID] {f}: {reason}")
    if len(invalid_folders) > 5:
        print(f"  ... and {len(invalid_folders)-5} more")

    if not valid_folders:
        print("No valid videos to process.")
        return

    # 3. Compute V_video for each video
    results = []
    total = len(valid_folders)

    for idx, folder in enumerate(valid_folders, 1):
        print(f"[{idx}/{total}] Processing: {folder}")

        data = valid_data[folder]
        scene_embeddings = data['embeddings']  # (T, 2048)
        scene_captions = load_scene_captions(folder, data_root)

        if not scene_captions or all(not cap.strip() for cap in scene_captions):
            print(f"  [WARNING] No valid captions for {folder}. Using mean of visual embeddings.")
            v_video = scene_embeddings.mean(dim=0)
        else:
            try:
                v_video = cross_modal_self_query(
                    scene_visual_embeddings=scene_embeddings,
                    scene_captions=scene_captions,
                    embedder=embedder,
                    temperature=args.temperature,
                    device=device,
                )
            except Exception as e:
                print(f"  [ERROR] cross_modal_self_query failed: {e}")
                print(f"  [FALLBACK] Using mean of visual embeddings for {folder}.")
                v_video = scene_embeddings.mean(dim=0)

        results.append({
            "folder_name": folder,
            "scene_captions": scene_captions,
            "v_video": v_video.cpu().float(),
            "temperature": args.temperature,
        })

        if args.device == 'cuda':
            torch.cuda.empty_cache()

    # 4. Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    v_video_dict = {r["folder_name"]: r["v_video"] for r in results}
    pt_path = output_dir / "video_representations.pt"
    torch.save(v_video_dict, pt_path)

    meta_path = output_dir / "video_representations_meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "total_videos": len(results),
                "temperature": args.temperature,
                "embedding_model": args.embedding_model_name,
            },
            "videos": [
                {
                    "folder_name": r["folder_name"],
                    "scene_captions": r["scene_captions"],
                    "temperature": r["temperature"],
                }
                for r in results
            ],
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Saved {len(results)} video representations to: {output_dir}")
    print(f"  - split_key used      : {args.split_key}")
    print(f"  - Embeddings (binary) : {pt_path}")
    print(f"  - Metadata (JSON)     : {meta_path}")
    print(f"  - First video: {results[0]['folder_name']}")


# ==========================================
# 4. ENTRY POINT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute video representations using Cross-Modal Self-Querying."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing per-video folders."
    )
    parser.add_argument(
        "--split_file",
        type=str,
        required=True,
        help="Path to dataset_splits.json file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory: will create video_representations.pt + video_representations_meta.json."
    )
    parser.add_argument(
        "--split_key",
        type=str,
        default="test",
        help="Key in split_file to fetch video list."
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="Qwen/Qwen3-VL-Embedding-2B",
        help="Path/ID for Qwen3-VL-Embedding-2B model."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help="Temperature τ for Softmax weighting."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the embedding model on."
    )
    args = parser.parse_args()
    main(args)