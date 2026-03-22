# -*- coding: utf-8 -*-
"""Common base classifier for strict DG methods.

This file extracts the reusable model skeleton that was previously living inside
ERMClassifier. It is intentionally method-agnostic: it only knows how to build
backbones/classifier, extract features for freq/tf/both modes, and produce
logits. Concrete methods should subclass BaseDGClassifier and implement their
own compute_loss() when needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List

import torch
import torch.nn as nn

from src.backbones import build_backbone
from src.heads import LinearClassifier


@dataclass
class BaseDGConfig:
    feature_mode: str = "freq"
    num_classes: int = 2
    freq_backbone_name: str = "resnet1d18"
    tf_backbone_name: str = "resnet18"
    freq_in_channels: int = 1
    tf_in_channels: int = 1
    freq_pretrained: bool = False
    tf_pretrained: bool = False
    classifier_dropout: float = 0.0
    backbone_kwargs: Optional[Dict] = None


class BaseDGClassifier(nn.Module):
    def __init__(self, cfg: BaseDGConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.feature_mode = str(cfg.feature_mode).lower()
        if self.feature_mode not in {"freq", "tf", "both"}:
            raise ValueError("feature_mode must be one of {'freq', 'tf', 'both'}")

        kwargs = cfg.backbone_kwargs or {}
        self.freq_backbone = None
        self.tf_backbone = None
        feat_dim = 0

        if self.feature_mode in {"freq", "both"}:
            self.freq_backbone = build_backbone(
                cfg.freq_backbone_name,
                in_channels=cfg.freq_in_channels,
                pretrained=cfg.freq_pretrained,
                **kwargs,
            )
            feat_dim += int(self.freq_backbone.out_dim)

        if self.feature_mode in {"tf", "both"}:
            self.tf_backbone = build_backbone(
                cfg.tf_backbone_name,
                in_channels=cfg.tf_in_channels,
                pretrained=cfg.tf_pretrained,
                **kwargs,
            )
            feat_dim += int(self.tf_backbone.out_dim)

        self.feat_dim = feat_dim
        self.classifier = LinearClassifier(feat_dim, cfg.num_classes, dropout=cfg.classifier_dropout)

    def _extract_freq_feature(self, x_freq: torch.Tensor) -> torch.Tensor:
        if self.freq_backbone is None:
            raise RuntimeError("freq backbone is not initialized for current feature_mode")
        return self.freq_backbone(x_freq)

    def _extract_tf_feature(self, x_tf: torch.Tensor) -> torch.Tensor:
        if self.tf_backbone is None:
            raise RuntimeError("tf backbone is not initialized for current feature_mode")
        return self.tf_backbone(x_tf)

    @staticmethod
    def _merge_features(feats: List[torch.Tensor]) -> torch.Tensor:
        if not feats:
            raise RuntimeError("No modality feature was produced.")
        return feats[0] if len(feats) == 1 else torch.cat(feats, dim=1)

    def extract_features(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feats = []
        out: Dict[str, torch.Tensor] = {}

        if self.feature_mode in {"freq", "both"}:
            freq_feat = self._extract_freq_feature(batch["x_freq"])
            out["feat_freq"] = freq_feat
            feats.append(freq_feat)

        if self.feature_mode in {"tf", "both"}:
            tf_feat = self._extract_tf_feature(batch["x_tf"])
            out["feat_tf"] = tf_feat
            feats.append(tf_feat)

        out["feature"] = self._merge_features(feats)
        return out

    def forward_logits(self, feature: torch.Tensor) -> torch.Tensor:
        return self.classifier(feature)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = self.extract_features(batch)
        out["logits"] = self.forward_logits(out["feature"])
        return out
