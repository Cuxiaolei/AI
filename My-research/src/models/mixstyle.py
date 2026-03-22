# -*- coding: utf-8 -*-
"""MixStyle baseline adapted to the unified backbone interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import BaseDGClassifier, BaseDGConfig


class MixStyle(nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or torch.rand(1, device=x.device).item() > self.p:
            return x
        b = x.size(0)
        if b <= 1:
            return x
        dims = list(range(2, x.dim()))
        mu = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, unbiased=False, keepdim=True)
        sig = (var + self.eps).sqrt()
        x_normed = (x - mu) / sig

        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample((b, 1)).to(x.device)
        for _ in range(x.dim() - 2):
            lmda = lmda.unsqueeze(-1)
        perm = torch.randperm(b, device=x.device)
        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)
        return x_normed * sig_mix + mu_mix


@dataclass
class MixStyleConfig(BaseDGConfig):
    mix_p: float = 0.5
    mix_alpha: float = 0.1
    mix_layer: str = 'layer1'


class MixStyleClassifier(BaseDGClassifier):
    def __init__(self, cfg: MixStyleConfig) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.mixstyle = MixStyle(p=cfg.mix_p, alpha=cfg.mix_alpha)
        self.mix_layer = cfg.mix_layer

    def _forward_backbone_with_mixstyle(self, backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = backbone.stem(x)
        if hasattr(backbone, 'maxpool'):
            x = backbone.maxpool(x)
        x = backbone.layer1(x)
        if self.mix_layer == 'layer1':
            x = self.mixstyle(x)
        x = backbone.layer2(x)
        if self.mix_layer == 'layer2':
            x = self.mixstyle(x)
        x = backbone.layer3(x)
        if self.mix_layer == 'layer3':
            x = self.mixstyle(x)
        x = backbone.layer4(x)
        if self.mix_layer == 'layer4':
            x = self.mixstyle(x)
        x = backbone.pool(x).flatten(1)
        return x

    def _extract_freq_feature(self, x_freq: torch.Tensor) -> torch.Tensor:
        return self._forward_backbone_with_mixstyle(self.freq_backbone, x_freq)

    def _extract_tf_feature(self, x_tf: torch.Tensor) -> torch.Tensor:
        return self._forward_backbone_with_mixstyle(self.tf_backbone, x_tf)

    def compute_loss(self, batch: Dict[str, torch.Tensor], criterion: nn.Module, epoch=None, global_step=None) -> Dict[str, torch.Tensor]:
        out = self.forward(batch)
        ce = criterion(out['logits'], batch['y'])
        out.update({'loss': ce, 'ce_loss': ce.detach()})
        return out
