# -*- coding: utf-8 -*-
"""
Test a backbone by connecting it to the unified H5 dataloader.

Usage examples
--------------
python tests/test_backbone_with_loader.py \
    --h5_path /path/to/train.h5 \
    --dataset_name pu \
    --feature_mode freq \
    --backbone_name resnet1d18

python tests/test_backbone_with_loader.py \
    --h5_path /path/to/train.h5 \
    --dataset_name cwru \
    --feature_mode tf \
    --backbone_name resnet18 \
    --pretrained

python tests/test_backbone_with_loader.py \
    --h5_path /path/to/train.h5 \
    --dataset_name phm \
    --feature_mode both \
    --backbone_name_freq resnet1d18 \
    --backbone_name_tf resnet18
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

torch.set_num_threads(1)

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import build_dataloader  # noqa: E402
from src.backbones import build_backbone  # noqa: E402


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)



def infer_input_shape(batch: dict, feature_mode: str) -> Tuple[Tuple[int, ...], str]:
    if feature_mode == "freq":
        return tuple(batch["x_freq"].shape), "x_freq"
    if feature_mode == "tf":
        return tuple(batch["x_tf"].shape), "x_tf"
    raise ValueError("For infer_input_shape, feature_mode must be 'freq' or 'tf'.")



def test_single_backbone(batch: dict, feature_mode: str, backbone_name: str, num_classes: int, pretrained: bool) -> None:
    input_shape, key = infer_input_shape(batch, feature_mode)
    x = batch[key]
    in_channels = int(x.shape[1])

    backbone = build_backbone(backbone_name, in_channels=in_channels, pretrained=pretrained)
    head = LinearHead(in_dim=backbone.out_dim, num_classes=num_classes)

    print(f"\n[Test Single Backbone] key={key}, backbone={backbone_name}, input_shape={input_shape}, out_dim={backbone.out_dim}")

    with torch.no_grad():
        feat = backbone(x)
        logits = head(feat)

    print(f"Feature shape: {tuple(feat.shape)}")
    print(f"Logits shape:  {tuple(logits.shape)}")

    assert feat.ndim == 2, f"Expected feature ndim=2, got {feat.ndim}"
    assert logits.shape[0] == x.shape[0], "Batch size mismatch between logits and input"
    assert logits.shape[1] == num_classes, "Class dimension mismatch"



def test_feature_maps(batch: dict, feature_mode: str, backbone_name: str, pretrained: bool) -> None:
    _, key = infer_input_shape(batch, feature_mode)
    x = batch[key]
    in_channels = int(x.shape[1])
    backbone = build_backbone(backbone_name, in_channels=in_channels, pretrained=pretrained)

    with torch.no_grad():
        feats = backbone(x, return_feature_maps=True)

    print(f"\n[Intermediate Feature Maps] backbone={backbone_name}")
    for name, tensor in feats.items():
        print(f"  - {name}: shape={tuple(tensor.shape)}")

    assert "pooled" in feats, "Missing pooled feature"



def test_both_modalities(batch: dict, num_classes: int, backbone_name_freq: str, backbone_name_tf: str) -> None:
    x_freq = batch["x_freq"]
    x_tf = batch["x_tf"]

    freq_backbone = build_backbone(backbone_name_freq, in_channels=int(x_freq.shape[1]), pretrained=False)
    tf_backbone = build_backbone(backbone_name_tf, in_channels=int(x_tf.shape[1]), pretrained=False)

    fusion_dim = freq_backbone.out_dim + tf_backbone.out_dim
    head = LinearHead(in_dim=fusion_dim, num_classes=num_classes)

    print(
        f"\n[Test Dual Backbones] freq_backbone={backbone_name_freq}, tf_backbone={backbone_name_tf}, "
        f"freq_out_dim={freq_backbone.out_dim}, tf_out_dim={tf_backbone.out_dim}, fusion_dim={fusion_dim}"
    )

    with torch.no_grad():
        freq_feat = freq_backbone(x_freq)
        tf_feat = tf_backbone(x_tf)
        fused_feat = torch.cat([freq_feat, tf_feat], dim=1)
        logits = head(fused_feat)

    print(f"Freq feature shape: {tuple(freq_feat.shape)}")
    print(f"TF feature shape:   {tuple(tf_feat.shape)}")
    print(f"Fused shape:        {tuple(fused_feat.shape)}")
    print(f"Logits shape:       {tuple(logits.shape)}")

    assert fused_feat.shape[1] == fusion_dim
    assert logits.shape[1] == num_classes



def test_one_training_step(batch: dict, feature_mode: str, backbone_name: str, num_classes: int, pretrained: bool) -> None:
    _, key = infer_input_shape(batch, feature_mode)
    x = batch[key]
    y = batch["y"]

    backbone = build_backbone(backbone_name, in_channels=int(x.shape[1]), pretrained=pretrained)
    head = LinearHead(in_dim=backbone.out_dim, num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=1e-3)

    feat = backbone(x)
    logits = head(feat)
    loss = criterion(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"\n[One-Step Train Sanity Check] backbone={backbone_name}, loss={loss.item():.6f}")
    print("Backward and optimizer step: OK")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="generic", choices=["generic", "phm", "phm2009", "pu", "cwru"])
    parser.add_argument("--feature_mode", type=str, default="freq", choices=["freq", "tf", "both"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--backbone_name", type=str, default="resnet1d18")
    parser.add_argument("--backbone_name_freq", type=str, default="resnet1d18")
    parser.add_argument("--backbone_name_tf", type=str, default="resnet18")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    loader = build_dataloader(
        h5_path=args.h5_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        dataset_kwargs={"feature_mode": args.feature_mode},
    )
    dataset = loader.dataset
    num_classes = dataset.get_num_classes()

    batch = next(iter(loader))
    print("========== Loader -> Backbone Test ==========")
    print(f"Dataset class: {dataset.__class__.__name__}")
    print(f"Feature mode:  {args.feature_mode}")
    print(f"Num classes:   {num_classes}")
    print(f"Batch keys:    {list(batch.keys())}")

    if args.feature_mode in {"freq", "tf"}:
        test_single_backbone(
            batch=batch,
            feature_mode=args.feature_mode,
            backbone_name=args.backbone_name,
            num_classes=num_classes,
            pretrained=args.pretrained,
        )
        test_feature_maps(
            batch=batch,
            feature_mode=args.feature_mode,
            backbone_name=args.backbone_name,
            pretrained=args.pretrained,
        )
        test_one_training_step(
            batch=batch,
            feature_mode=args.feature_mode,
            backbone_name=args.backbone_name,
            num_classes=num_classes,
            pretrained=args.pretrained,
        )
    else:
        test_both_modalities(
            batch=batch,
            num_classes=num_classes,
            backbone_name_freq=args.backbone_name_freq,
            backbone_name_tf=args.backbone_name_tf,
        )

    print("\n✅ Backbone test passed.")


if __name__ == "__main__":
    main()
