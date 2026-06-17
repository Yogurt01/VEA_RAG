#!/usr/bin/env python3
"""
Run train RGCN and evaluate on split dataset

Usage:
    python train_gnn.py \
        --data_root EnTube \
        --split_dataset_file EnTube/dataset_splits.json \
        --hidden_dim 512 \
        --num_layers 3 \
        --train_batch_size 16 \
        --epochs 50 \
        --lr 1e-4 \
        --dropout 0.2 \
        --device cuda
"""


import os
import os.path as osp
import json
import argparse
from typing import List
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# pyrefly: ignore [missing-import]
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Subset
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool, global_max_pool
from torch_geometric.utils import coalesce

from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


# =====================
# Dataset Preparation
# =====================

def to_undirected_edge_index(edge_index: Tensor,
                              edge_attr: Tensor = None,
                              num_nodes: int = None):
    """Convert directed graph to undirected graph."""
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    row, col = edge_index
    edge_index_undir = torch.cat(
        [edge_index, torch.stack([col, row])], dim=1)

    if edge_attr is not None:
        # Duplicate edge attributes for reverse edges
        edge_attr_undir = torch.cat([edge_attr, edge_attr], dim=0)
    else:
        edge_attr_undir = None

    # Use 'max' instead of 'mean' to preserve integer relation types
    edge_index_undir, edge_attr_undir = coalesce(
        edge_index_undir,
        edge_attr_undir,
        num_nodes=num_nodes,
        reduce='max',
    )
    return edge_index_undir, edge_attr_undir


@dataclass(frozen=True)
class VideoSample:
    video_id: str
    video_dir: str
    scene_pt_path: str


def _find_all_scene_embeddings(dataset_root: str) -> List[VideoSample]:
    """Find all valid scene embedding files in the dataset."""
    samples: List[VideoSample] = []
    if not osp.isdir(dataset_root):
        return samples
    
    for root, dirs, files in os.walk(dataset_root):
        if "scene_embeddings.pt" in files:
            pt_path = osp.join(root, "scene_embeddings.pt")
            video_id = osp.basename(root)
            samples.append(VideoSample(
                video_id=video_id,
                video_dir=root,
                scene_pt_path=pt_path,
            ))
    samples.sort(key=lambda x: x.video_id)
    return samples


class VideoEngagementDataset(Dataset):
    # Hardcoded number of RST edge types according to RST-DT (18 classes)
    NUM_RST_RELATIONS = 18

    def __init__(
        self,
        dataset_path: str,
        transform=None,
        pre_transform=None,
        strict: bool = True,
    ):
        self.dataset_root = osp.abspath(dataset_path)
        self.strict = strict
        self._samples = _find_all_scene_embeddings(self.dataset_root)

        if self.strict and not self._samples:
            raise FileNotFoundError(f"No scene_embeddings.pt found under {self.dataset_root!r}")

        self._n_classes = 2
        self._n_features = self._infer_num_features()
        self._n_relations = self.NUM_RST_RELATIONS

        super().__init__(root=self.dataset_root, transform=transform, pre_transform=pre_transform)

    @property
    def num_features(self) -> int:
        return int(self._n_features)

    @property
    def num_classes(self) -> int:
        return self._n_classes

    @property
    def num_relations(self) -> int:
        return self._n_relations

    def _infer_num_features(self) -> int:
        if not self._samples:
            return 0
        obj = torch.load(self._samples[0].scene_pt_path, map_location="cpu", weights_only=False)
        x = obj.get("embeddings")
        if not isinstance(x, Tensor) or x.dim() != 2:
            raise ValueError(f"Expected `embeddings` [N, D] in {self._samples[0].scene_pt_path}")
        return int(x.size(-1))

    # This method is kept for debugging but not used for num_relations.
    # We now hardcode 18 relations.
    def _infer_num_relations_from_data(self) -> int:
        """Only for diagnostic purposes. Returns the actual max relation id + 1 found in data."""
        if not self._samples:
            return 1
        max_rel = 0
        for sample in self._samples:
            obj = torch.load(sample.scene_pt_path, map_location="cpu", weights_only=False)
            edge_attr = obj.get("edge_attr")
            if isinstance(edge_attr, Tensor) and edge_attr.numel() > 0:
                max_rel = max(max_rel, int(edge_attr.max().item()))
        return max_rel + 1

    def len(self) -> int:
        return len(self._samples)

    def get(self, idx: int) -> Data:
        s = self._samples[idx]
        obj = torch.load(s.scene_pt_path, map_location="cpu", weights_only=False)

        x = obj.get("embeddings")
        edge_index = obj.get("edge_index")
        y_obj = obj.get("y")
        edge_attr = obj.get("edge_attr")

        if not isinstance(x, Tensor) or x.dim() != 2:
            raise ValueError(f"Bad embeddings in {s.scene_pt_path}: expected [N, D].")
        if not isinstance(edge_index, Tensor) or edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"Bad edge_index in {s.scene_pt_path}: expected [2, E].")
        if not isinstance(y_obj, Tensor) or y_obj.numel() != 1:
            raise ValueError(f"Missing/invalid y in {s.scene_pt_path}: expected scalar.")

        # Ensure edge_attr values are within [0, NUM_RST_RELATIONS-1]
        if edge_attr is not None and isinstance(edge_attr, Tensor) and edge_attr.numel() > 0:
            if edge_attr.max() >= self.NUM_RST_RELATIONS:
                print(f"Warning: {s.video_id} has edge_attr max = {edge_attr.max()} >= {self.NUM_RST_RELATIONS}. Clamping to {self.NUM_RST_RELATIONS-1}.")
                edge_attr = edge_attr.clamp(max=self.NUM_RST_RELATIONS - 1)

        edge_index, edge_attr = to_undirected_edge_index(
            edge_index, edge_attr, num_nodes=x.size(0))

        data = Data(
            x=x.to(dtype=torch.float32),
            edge_index=edge_index.to(dtype=torch.long).contiguous(),
            y=torch.tensor([int(y_obj.item())], dtype=torch.long),
        )

        # RGCN requires edge_type to be a 1D tensor of type Long
        if isinstance(edge_attr, Tensor):
            data.edge_type = edge_attr.view(-1).to(torch.long)
        else:
            data.edge_type = torch.zeros(data.edge_index.size(1), dtype=torch.long)

        data.video_id = s.video_id
        data.video_dir = s.video_dir
        return data


# =====================
# GNN Model
# =====================

class VideoRGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_relations, num_layers, num_classes, dropout=0.2):
        super().__init__()

        self.node_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.convs = nn.ModuleList()
        num_bases = min(10, num_relations) # num_bases should be <= num_relations

        for _ in range(num_layers):
            self.convs.append(RGCNConv(
                hidden_dim,
                hidden_dim,
                num_relations=num_relations,
                num_bases=num_bases
            ))

        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, edge_type, batch):
        x = self.node_proj(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_graph = torch.cat([x_mean, x_max], dim=1)

        return self.classifier(x_graph)


# =====================
# Training & Evaluation
# =====================

def train_step(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.edge_type, data.batch)

        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        preds = out.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, acc, f1


def evaluate_step(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_type, data.batch)

            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs

            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, acc, f1


# =====================
# MAIN RUN
# =====================

def run(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Tạo thư mục lưu model theo thời gian
    base_model_dir = Path("models/rgcn")
    base_model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_model_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    # Lưu lại tham số dòng lệnh
    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    best_model_path = run_dir / "best_rgcn_model.pth"
    print(f"Model will be saved to: {best_model_path}")

    # 1. Load Dataset (num_relations is now hardcoded to 18)
    full_dataset = VideoEngagementDataset(args.data_root)
    print(f"Dataset loaded: {len(full_dataset)} samples, {full_dataset.num_features} features, "
          f"relations (hardcoded RST-DT): {full_dataset.num_relations}")

    # (Optional) diagnostic: what is the actual max relation id in data?
    actual_num_rels = full_dataset._infer_num_relations_from_data()
    print(f"[Diagnostic] Actual unique relation indices in data: max+1 = {actual_num_rels} "
          f"(expected <= {full_dataset.num_relations})")

    with open(args.split_dataset_file, 'r') as f:
        splits = json.load(f)

    train_indices = [i for i, s in enumerate(full_dataset._samples) if s.video_id in splits["train"]]
    val_indices = [i for i, s in enumerate(full_dataset._samples) if s.video_id in splits["val"]]
    test_indices = [i for i, s in enumerate(full_dataset._samples) if s.video_id in splits["test"]]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False)

    # 2. Initialize Model and Optimizer
    model = VideoRGCN(
        in_dim=full_dataset.num_features,
        hidden_dim=args.hidden_dim,
        num_relations=full_dataset.num_relations,  # Now equals 18
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # 3. Training Loop
    best_val_f1 = 0.0

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss, train_acc, train_f1 = train_step(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate_step(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:03d} | LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved (Val F1: {best_val_f1:.4f}) to {best_model_path}")

    # 4. Final Test Evaluation
    print(f"\nLoading best model from {best_model_path} for final test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc, test_f1 = evaluate_step(model, test_loader, criterion, device)

    print("\n" + "=" * 55)
    print("FINAL TEST RESULTS")
    print("=" * 55)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
    print("=" * 55)
    print(f"All training artifacts saved in: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VideoRGCN for engagement prediction")

    # Data paths
    parser.add_argument("--data_root", type=str, default="EnTube", help="Root directory of the dataset")
    parser.add_argument("--split_dataset_file", type=str, default="/content/drive/MyDrive/KhoaLuan/EnTube/dataset_splits.json", help="Path to split JSON file")

    # Training hyperparameters
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--val_batch_size", type=int, default=1, help="Batch size for validation/testing")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for AdamW")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")

    # Model architecture
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of RGCN layers")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of output classes")

    # System
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on (cuda or cpu)")

    args = parser.parse_args()
    run(args)