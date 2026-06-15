#!/usr/bin/env python3
"""
Run SubgraphX explanation on trained RGCN model and RST graphs.

Usage:
    python run_subgraphx.py --data_root EnTube \
                            --model_path model/YYYYMMDD_HHMMSS/best_rgcn_model.pth \
                            --output_dir EnTube/SubgraphX_Results \
                            --split_json EnTube/dataset_splits.json \
                            --split test \
                            --n_min 5 \
                            --m 15 --t 10 --exp_weight 5.0 --max_children 12
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from typing import Set
from dataclasses import dataclass
from torch_geometric.data import Data
from torch.utils.data import Subset

from train_gnn import VideoEngagementDataset, VideoRGCN

sys.path.insert(0, str(Path(__file__).parent))

from subgraphx.subgraph_x import SubgraphX
from subgraphx.shapley import mc_l_shapley
from subgraphx.task_enum import Task


# ========================
# Hàm tính fidelity và sparsity
# ========================
@torch.inference_mode()
def fidelity(graph: Data, node_set: Set[int], model: nn.Module) -> float:
    """
    Tính fidelity: sự thay đổi xác suất dự đoán khi chỉ giữ lại các node quan trọng.
    Ở đây tính theo tỉ lệ logit của class dự đoán (hoặc xác suất).
    """
    model.eval()
    device = graph.x.device

    # Ensure node_set isn't empty to avoid errors
    if not node_set:
        return 0.0

    batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)

    # Original prediction
    logits = model(graph.x, graph.edge_index, graph.edge_type, batch)
    probs = torch.softmax(logits, dim=-1)

    predicted_class = torch.argmax(probs, dim=-1)
    # Use index select for cleaner code
    original_score = probs[0, predicted_class].item()

    # Occluded Graph
    x_occluded = graph.x.clone()
    # Convert node_set to tensor safely
    node_indices = torch.as_tensor(list(node_set), device=device, dtype=torch.long)
    x_occluded[node_indices] = 0.0

    # Occluded prediction
    logits_occ = model(x_occluded, graph.edge_index, graph.edge_type, batch)
    probs_occ = torch.softmax(logits_occ, dim=-1)
    occluded_score = probs_occ[0, predicted_class].item()

    return original_score - occluded_score


def sparsity(graph, node_set):
    """
    Sparsity = 1 - (|Ei| / |Gi|)
    |Ei| là số lượng nút trong giải thích, |Gi| là tổng số nút trong đồ thị
    """
    return 1 - (len(node_set) / graph.num_nodes)


# ========================
# Hàm chính
# ========================
def run_explanations(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1. Load dataset (dùng class từ train_gnn, đã hardcode num_relations=18)
    full_dataset = VideoEngagementDataset(args.data_root)
    print(f"Loaded {len(full_dataset)} videos.")
    print(f"Input features: {full_dataset.num_features}, Relations: {full_dataset.num_relations}")

    # 2. Load split file
    with open(args.split_json, 'r') as f:
        splits = json.load(f)
    split_set = splits[args.split]
    print(f"Using split '{args.split}' with {len(split_set)} videos.")

    # Lọc indices theo split
    id_to_idx = {s.video_id: i for i, s in enumerate(full_dataset._samples)}
    indices = [id_to_idx[v] for v in split_set if v in id_to_idx]
    dataset = Subset(full_dataset, indices)
    print(f"Actual videos to process: {len(dataset)}")

    # 3. Load trained model (kiến trúc giống hệt train_gnn)
    model = VideoRGCN(
        in_dim=full_dataset.num_features,
        hidden_dim=args.hidden_dim,
        num_relations=full_dataset.num_relations,  # =18
        num_layers=args.num_layers,
        num_classes=2,
        dropout=args.dropout
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    torch.set_grad_enabled(False)

    # 4. Khởi tạo SubgraphX
    subgraphx = SubgraphX(
        model=model,
        num_layers=args.num_layers,
        exp_weight=args.exp_weight,
        m=args.m,
        t=args.t,
        high2low=args.high2low,
        max_children=args.max_children,
        task=Task.GRAPH_CLASSIFICATION,
        value_func=mc_l_shapley,
        experiment=None
    )

    # 5. Tạo thư mục output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 6. Duyệt từng video
    for idx in tqdm(range(len(dataset)), desc="Processing videos"):
        data = dataset[idx]   # data là Data object, đã có sẵn .edge_type (do VideoEngagementDataset tạo)
        video_name = data.video_id
        out_path = out_dir / f"{video_name}_explanation.pt"

        if out_path.exists() and not args.force:
            print(f"Skipping {video_name} (already exists)")
            continue

        # Đưa graph lên device (các tensor đã được định nghĩa đúng kiểu)
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.edge_type = data.edge_type.to(device)
        data.y = data.y.to(device)

        print(f"\nProcessing {video_name} | nodes={data.num_nodes} edges={data.edge_index.size(1)}")

        if data.num_nodes <= args.n_min:
            print(f"  Too few nodes ({data.num_nodes} <= {args.n_min}), using full graph as explanation.")
            important_nodes = set(range(data.num_nodes))
        else:
            try:
                important_nodes, _ = subgraphx(data, n_min=args.n_min, nodes_to_keep=None, exhaustive=True)
                important_nodes = set(important_nodes)
            except Exception as e:
                print(f"  SubgraphX failed: {e}, falling back to full graph.")
                important_nodes = set(range(data.num_nodes))

        fid_score = fidelity(data, important_nodes, model)
        spar_score = sparsity(data, important_nodes)

        print(f"  Important nodes: {sorted(important_nodes)}")
        print(f"  Fidelity: {fid_score:.4f}, Sparsity: {spar_score:.4f}")

        orig_idx = indices[idx]
        orig_sample = full_dataset._samples[orig_idx]
        orig_ckpt = torch.load(orig_sample.scene_pt_path, map_location="cpu")

        result = {
            "video_name": video_name,
            "important_scenes": sorted(list(important_nodes)),
            "fidelity": fid_score,
            "sparsity": spar_score,
            "original_y": data.y.cpu().item(),
            "embeddings": orig_ckpt.get("embeddings").float().cpu() if torch.is_tensor(orig_ckpt.get("embeddings")) else orig_ckpt.get("embeddings"),
            "edge_index": orig_ckpt.get("edge_index").long().cpu() if torch.is_tensor(orig_ckpt.get("edge_index")) else orig_ckpt.get("edge_index"),
            "edge_attr": orig_ckpt.get("edge_attr").long().cpu() if torch.is_tensor(orig_ckpt.get("edge_attr")) else orig_ckpt.get("edge_attr"),
            "rst_links": orig_ckpt.get("rst_links"),
            "scene_ids": orig_ckpt.get("scene_ids")
        }
        torch.save(result, out_path)
        print(f"  Saved to {out_path}")

    print("\nAll done!")


# ========================
# Argument parser
# ========================
def main():
    parser = argparse.ArgumentParser(description="Run SubgraphX explanations on RST graphs")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split_json", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--n_min", type=int, default=5)
    parser.add_argument("--m", type=int, default=15)
    parser.add_argument("--t", type=int, default=10)
    parser.add_argument("--exp_weight", type=float, default=5.0)
    parser.add_argument("--max_children", type=int, default=12)
    parser.add_argument("--high2low", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    run_explanations(args)


if __name__ == "__main__":
    main()