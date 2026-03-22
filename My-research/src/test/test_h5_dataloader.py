# -*- coding: utf-8 -*-
"""
Basic test script for unified HDF5 dataloaders.

Usage examples
--------------
python tests/test_h5_dataloader.py \
    --h5_path /path/to/train.h5 \
    --dataset_name phm \
    --feature_mode both

python src/test/test_h5_dataloader.py \
    --h5_path data/PHM_spur/PHM_spur8_T1_200-5_train.h5 \
    --dataset_name cwru \
    --feature_mode tf \
    --batch_size 16 \
    --num_workers 2
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import torch

import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import build_dataloader, build_dataset  # noqa: E402



def pretty_print_dict(title: str, data: Dict[str, Any]) -> None:
    print(f"\n[{title}]")
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))



def inspect_raw_h5(h5_path: str) -> None:
    print("\n========== Raw H5 Inspection ==========")
    with h5py.File(h5_path, "r") as f:
        print(f"File: {h5_path}")
        print(f"Keys: {list(f.keys())}")
        print("Attributes:")
        for key in f.attrs.keys():
            print(f"  - {key}: {f.attrs[key]}")
        for key in f.keys():
            arr = f[key]
            print(f"Dataset '{key}': shape={arr.shape}, dtype={arr.dtype}")



def check_dataset_level(dataset) -> None:
    print("\n========== Dataset-Level Check ==========")
    print(f"Dataset class: {dataset.__class__.__name__}")
    print(f"Length: {len(dataset)}")
    print(f"Num classes: {dataset.get_num_classes()}")
    print(f"Num domains: {dataset.get_num_domains()}")
    print(f"Split name: {dataset.get_split_name()}")
    pretty_print_dict("Summary", dataset.get_summary())
    pretty_print_dict("Label map", dataset.get_label_map())
    pretty_print_dict("Domain map", dataset.get_domain_map())

    sample = dataset[0]
    print("\nFirst sample keys:", list(sample.keys()))
    for k, v in sample.items():
        if torch.is_tensor(v):
            print(f"  - {k}: shape={tuple(v.shape)}, dtype={v.dtype}, min={v.min().item() if v.numel() else 'NA'}, max={v.max().item() if v.numel() else 'NA'}")
        else:
            print(f"  - {k}: type={type(v)}, value={v}")

    labels = dataset.get_all_labels()
    domains = dataset.get_all_domains()
    print("\nLabel distribution:")
    print(dict(sorted(Counter(labels.tolist()).items(), key=lambda x: x[0])))
    print("Domain distribution:")
    print(dict(sorted(Counter(domains.tolist()).items(), key=lambda x: x[0])))



def check_dataloader_level(loader, feature_mode: str) -> None:
    print("\n========== DataLoader-Level Check ==========")
    batch = next(iter(loader))
    print("Batch keys:", list(batch.keys()))

    if feature_mode in {"freq", "both"}:
        x_freq = batch["x_freq"]
        print(f"x_freq batch shape: {tuple(x_freq.shape)}, dtype={x_freq.dtype}")
        assert x_freq.ndim == 3, f"Expected x_freq ndim=3, got {x_freq.ndim}"

    if feature_mode in {"tf", "both"}:
        x_tf = batch["x_tf"]
        print(f"x_tf batch shape: {tuple(x_tf.shape)}, dtype={x_tf.dtype}")
        assert x_tf.ndim == 4, f"Expected x_tf ndim=4, got {x_tf.ndim}"

    y = batch["y"]
    d = batch["domain"]
    print(f"y batch shape: {tuple(y.shape)}, dtype={y.dtype}")
    print(f"domain batch shape: {tuple(d.shape)}, dtype={d.dtype}")

    assert y.ndim == 1, f"Expected y ndim=1, got {y.ndim}"
    assert d.ndim == 1, f"Expected domain ndim=1, got {d.ndim}"
    assert y.dtype == torch.long, f"Expected y dtype=torch.long, got {y.dtype}"
    assert d.dtype == torch.long, f"Expected domain dtype=torch.long, got {d.dtype}"



def check_consistency_with_raw_h5(h5_path: str, dataset, num_checks: int = 3) -> None:
    print("\n========== Sample Consistency Check ==========")
    with h5py.File(h5_path, "r") as f:
        total = len(dataset)
        indices = np.linspace(0, total - 1, num=min(num_checks, total), dtype=int)
        for idx in indices.tolist():
            item = dataset[idx]
            raw_y = int(f["y"][idx])
            raw_d = int(f["domain"][idx])
            ds_y = int(item["y"].item() if torch.is_tensor(item["y"]) else item["y"])
            ds_d = int(item["domain"].item() if torch.is_tensor(item["domain"]) else item["domain"])
            assert raw_y == ds_y, f"Mismatch at idx={idx}: raw_y={raw_y}, ds_y={ds_y}"
            assert raw_d == ds_d, f"Mismatch at idx={idx}: raw_domain={raw_d}, ds_domain={ds_d}"
            print(f"idx={idx}: y OK ({raw_y}), domain OK ({raw_d})")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str, required=True, help="Path to one train.h5 or test.h5 file")
    parser.add_argument("--dataset_name", type=str, default="generic", choices=["generic", "phm", "phm2009", "pu", "cwru"])
    parser.add_argument("--feature_mode", type=str, default="both", choices=["freq", "tf", "both"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--return_index", action="store_true")
    args = parser.parse_args()

    inspect_raw_h5(args.h5_path)

    dataset = build_dataset(
        h5_path=args.h5_path,
        dataset_name=args.dataset_name,
        feature_mode=args.feature_mode,
        return_index=args.return_index,
    )
    check_dataset_level(dataset)
    check_consistency_with_raw_h5(args.h5_path, dataset)

    loader = build_dataloader(
        h5_path=args.h5_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        dataset_kwargs={
            "feature_mode": args.feature_mode,
            "return_index": args.return_index,
        },
    )
    check_dataloader_level(loader, args.feature_mode)

    print("\n✅ DataLoader test passed.")


if __name__ == "__main__":
    main()
