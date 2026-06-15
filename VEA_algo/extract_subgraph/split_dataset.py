#!/usr/bin/env python3
"""
Run split dataset into train, val, test

Usage:
    python split_dataset.py --data_root EnTube \
                            --split_dataset_file EnTube/dataset_splits.json
                            --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
"""

import os
import json
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse


def run(args):
    valid_ids = []
    valid_labels = []

    subdirs = [os.path.join(args.data_root, d) for d in os.listdir(args.data_root)
               if os.path.isdir(os.path.join(args.data_root, d))]

    # 1. Iterate through directories to read labels for Stratified Split
    for subdir in tqdm(subdirs, desc="Reading labels"):
        folder_name = os.path.basename(subdir)
        pt_file_path = os.path.join(subdir, 'scene_embeddings.pt')

        if os.path.exists(pt_file_path):
            try:
                # Load with map_location="cpu" for fast label extraction
                data = torch.load(pt_file_path, map_location="cpu")
                if isinstance(data, dict) and 'y' in data:
                    valid_ids.append(folder_name)
                    # Convert to standard Python int for reliable counting later
                    valid_labels.append(int(data['y'].item())) 
            except Exception:
                pass  # Silently ignore corrupted files

    print(f"\nFound {len(valid_ids)} valid videos for splitting.")

    # 2. Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0. Current sum: {total_ratio}")

    # 3. First Split: Train vs Temp (Val + Test)
    temp_ratio = args.val_ratio + args.test_ratio
    train_ids, temp_ids, train_labels, temp_labels = train_test_split(
        valid_ids, valid_labels,
        test_size=temp_ratio,
        stratify=valid_labels,
        random_state=args.seed
    )

    # 4. Second Split: Temp into Validation and Test
    # test_size here is relative to temp_ids, so we calculate the proportion
    val_test_split_ratio = args.test_ratio / temp_ratio if temp_ratio > 0 else 0.0
    
    val_ids, test_ids, val_labels, test_labels = train_test_split(
        temp_ids, temp_labels,
        test_size=val_test_split_ratio,
        stratify=temp_labels,
        random_state=args.seed
    )

    # 5. Print statistics
    print("\n" + "=" * 55)
    print("DATASET SPLIT RESULTS")
    print("=" * 55)
    print(f"Train set: {len(train_ids)} videos (Label 0: {train_labels.count(0)}, Label 1: {train_labels.count(1)})")
    print(f"Val set  : {len(val_ids)} videos (Label 0: {val_labels.count(0)}, Label 1: {val_labels.count(1)})")
    print(f"Test set : {len(test_ids)} videos (Label 0: {test_labels.count(0)}, Label 1: {test_labels.count(1)})")

    # 6. Save split IDs to JSON file
    split_dict = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }

    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(args.split_dataset_file) or '.', exist_ok=True)
    
    with open(args.split_dataset_file, 'w') as f:
        json.dump(split_dict, f, indent=4, is_safe=True)

    print(f"\nSplit configuration saved successfully at:\n -> {args.split_dataset_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train, val, and test sets.")

    parser.add_argument("--data_root", type=str, default="EnTube", 
                        help="Root directory containing the dataset folders")
    parser.add_argument("--split_dataset_file", type=str, default="EnTube/dataset_splits.json", 
                        help="Path to save the output JSON split file")
    
    # New arguments for flexible splitting
    parser.add_argument("--train_ratio", type=float, default=0.8, 
                        help="Ratio of the dataset to be used for training (default: 0.8)")
    parser.add_argument("--val_ratio", type=float, default=0.1, 
                        help="Ratio of the dataset to be used for validation (default: 0.1)")
    parser.add_argument("--test_ratio", type=float, default=0.1, 
                        help="Ratio of the dataset to be used for testing (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()
    run(args)