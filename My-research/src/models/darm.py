# -*- coding: utf-8 -*-
"""Engineering-friendly DARM baseline for strict domain generalization.

Reference:
    Zhenling Mo, Zijun Zhang, Kwok-Leung Tsui,
    "Distance Aware Risk Minimization for Domain Generalization in Machine Fault Diagnosis,"
    IEEE Internet of Things Journal, 2024.

This implementation is a reusable approximation designed to fit the current project:
- It keeps the current backbone / classifier / dataloader stack unchanged.
- It adds two distance-aware terms on top of ERM:
  1) instance-to-instance (ItI) metric regularization on batch features
  2) prototype-to-prototype (PtP) metric regularization on domain-class prototypes

It is NOT a line-by-line reproduction of the paper's exact losses, but it preserves the core
idea of jointly considering ERM classification, instance-wise distances, and prototype-wise distances.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDGClassifier, BaseDGConfig


@dataclass
class DARMConfig(BaseDGConfig):
    darm_iti_weight: float = 0.1
    darm_ptp_weight: float = 0.1
    darm_margin: float = 1.0
    darm_feature_normalize: bool = True


class DARMClassifier(BaseDGClassifier):
    def __init__(self, cfg: DARMConfig) -> None:
        super().__init__(cfg)
        self.darm_iti_weight = float(cfg.darm_iti_weight)
        self.darm_ptp_weight = float(cfg.darm_ptp_weight)
        self.darm_margin = float(cfg.darm_margin)
        self.darm_feature_normalize = bool(cfg.darm_feature_normalize)

    def _prepare_feature(self, feat: torch.Tensor) -> torch.Tensor:
        if self.darm_feature_normalize:
            feat = F.normalize(feat, dim=1)
        return feat

    @staticmethod
    def _pairwise_dist(x: torch.Tensor) -> torch.Tensor:
        if x.size(0) <= 1:
            return x.new_zeros((x.size(0), x.size(0)))
        return torch.cdist(x, x, p=2)

    def _instance_to_instance_loss(self, feat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = feat.size(0)
        if n <= 1:
            return feat.new_tensor(0.0)
        dist = self._pairwise_dist(feat)
        device = feat.device
        eye = torch.eye(n, dtype=torch.bool, device=device)
        same = (y[:, None] == y[None, :]) & (~eye)
        diff = (y[:, None] != y[None, :])

        same_term = dist[same].mean() if same.any() else feat.new_tensor(0.0)
        diff_term = F.relu(self.darm_margin - dist[diff]).mean() if diff.any() else feat.new_tensor(0.0)
        return same_term + diff_term

    def _collect_domain_class_prototypes(
        self,
        feat: torch.Tensor,
        y: torch.Tensor,
        domains: torch.Tensor,
    ) -> List[Tuple[int, int, torch.Tensor]]:
        protos: List[Tuple[int, int, torch.Tensor]] = []
        unique_domains = torch.unique(domains)
        unique_classes = torch.unique(y)
        for d in unique_domains:
            dom_mask = domains == d
            for c in unique_classes:
                mask = dom_mask & (y == c)
                if int(mask.sum().item()) == 0:
                    continue
                proto = feat[mask].mean(dim=0)
                protos.append((int(c.item()), int(d.item()), proto))
        return protos

    def _prototype_to_prototype_loss(
        self,
        feat: torch.Tensor,
        y: torch.Tensor,
        domains: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if domains is None or feat.size(0) <= 1:
            return feat.new_tensor(0.0)

        protos = self._collect_domain_class_prototypes(feat, y, domains)
        if len(protos) <= 1:
            return feat.new_tensor(0.0)

        same_dists = []
        diff_hinges = []
        for i in range(len(protos)):
            ci, di, pi = protos[i]
            for j in range(i + 1, len(protos)):
                cj, dj, pj = protos[j]
                dist = torch.norm(pi - pj, p=2)
                if ci == cj and di != dj:
                    same_dists.append(dist)
                elif ci != cj:
                    diff_hinges.append(F.relu(self.darm_margin - dist))

        same_term = torch.stack(same_dists).mean() if same_dists else feat.new_tensor(0.0)
        diff_term = torch.stack(diff_hinges).mean() if diff_hinges else feat.new_tensor(0.0)
        return same_term + diff_term

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        criterion: nn.Module,
        epoch: Optional[int] = None,
        global_step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.forward(batch)
        logits = out['logits']
        feat = self._prepare_feature(out['feature'])
        y = batch['y']
        domains = batch.get('domain', None)

        ce = criterion(logits, y)
        iti_loss = self._instance_to_instance_loss(feat, y)
        ptp_loss = self._prototype_to_prototype_loss(feat, y, domains)
        loss = ce + self.darm_iti_weight * iti_loss + self.darm_ptp_weight * ptp_loss

        out.update({
            'loss': loss,
            'ce_loss': ce.detach(),
            'darm_iti_loss': iti_loss.detach(),
            'darm_ptp_loss': ptp_loss.detach(),
            'darm_iti_weight': logits.new_tensor(float(self.darm_iti_weight)),
            'darm_ptp_weight': logits.new_tensor(float(self.darm_ptp_weight)),
        })
        return out
