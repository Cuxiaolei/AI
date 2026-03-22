# -*- coding: utf-8 -*-
"""Engineering-friendly DPJDG baseline for strict domain generalization.

Reference:
    Kai Huang, Zhijun Ren, Tantao Lin, Yongsheng Zhu, Linbo Zhu,
    "A Dual-Perspective Joint Domain Generalization Network for Bearing Fault Diagnosis under Unseen Working Conditions,"
    Advanced Engineering Informatics, 2025.

This implementation is a reusable approximation for the current project:
- semantic-consistent augmentation on source-domain inputs
- class-level invariance via feature consistency between original and augmented views
- cross-domain invariance via source-domain MMD on learned features
- lightweight confidence weighting at instance/domain level

It is NOT a paper-exact reproduction of every module in DPJDG, but it keeps the main design
ideas while reusing the current data/backbone/trainer pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDGClassifier, BaseDGConfig


@dataclass
class DPJDGConfig(BaseDGConfig):
    dpjdg_consistency_weight: float = 0.5
    dpjdg_mmd_weight: float = 0.5
    dpjdg_aug_noise_std: float = 0.02
    dpjdg_mask_ratio: float = 0.05
    dpjdg_rbf_gamma: float = 1.0


class DPJDGClassifier(BaseDGClassifier):
    def __init__(self, cfg: DPJDGConfig) -> None:
        super().__init__(cfg)
        self.dpjdg_consistency_weight = float(cfg.dpjdg_consistency_weight)
        self.dpjdg_mmd_weight = float(cfg.dpjdg_mmd_weight)
        self.dpjdg_aug_noise_std = float(cfg.dpjdg_aug_noise_std)
        self.dpjdg_mask_ratio = float(cfg.dpjdg_mask_ratio)
        self.dpjdg_rbf_gamma = float(cfg.dpjdg_rbf_gamma)

    def _augment_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        noise = self.dpjdg_aug_noise_std * torch.randn_like(x)
        out = x + noise
        if self.dpjdg_mask_ratio > 0:
            mask = (torch.rand_like(out) > self.dpjdg_mask_ratio).to(out.dtype)
            out = out * mask
        return out

    def _augment_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        aug = dict(batch)
        if 'x_freq' in aug and torch.is_tensor(aug['x_freq']):
            aug['x_freq'] = self._augment_tensor(aug['x_freq'])
        if 'x_tf' in aug and torch.is_tensor(aug['x_tf']):
            aug['x_tf'] = self._augment_tensor(aug['x_tf'])
        return aug

    @staticmethod
    def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
        dist2 = torch.cdist(x, y, p=2).pow(2)
        return torch.exp(-gamma * dist2)

    def _mmd_rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.size(0) == 0 or y.size(0) == 0:
            return x.new_tensor(0.0)
        kxx = self._rbf_kernel(x, x, self.dpjdg_rbf_gamma)
        kyy = self._rbf_kernel(y, y, self.dpjdg_rbf_gamma)
        kxy = self._rbf_kernel(x, y, self.dpjdg_rbf_gamma)
        return kxx.mean() + kyy.mean() - 2.0 * kxy.mean()

    def _domain_mmd(self, feat: torch.Tensor, domains: Optional[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        if domains is None:
            return feat.new_tensor(0.0)
        unique_domains = torch.unique(domains)
        if int(unique_domains.numel()) <= 1:
            return feat.new_tensor(0.0)
        losses: List[torch.Tensor] = []
        for i in range(len(unique_domains)):
            for j in range(i + 1, len(unique_domains)):
                di = unique_domains[i]
                dj = unique_domains[j]
                mi = domains == di
                mj = domains == dj
                if int(mi.sum().item()) == 0 or int(mj.sum().item()) == 0:
                    continue
                wi = weights[mi].mean().detach() if int(mi.sum().item()) > 0 else feat.new_tensor(1.0)
                wj = weights[mj].mean().detach() if int(mj.sum().item()) > 0 else feat.new_tensor(1.0)
                w = 0.5 * (wi + wj)
                losses.append(w * self._mmd_rbf(feat[mi], feat[mj]))
        if not losses:
            return feat.new_tensor(0.0)
        return torch.stack(losses).mean()

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        criterion: nn.Module,
        epoch: Optional[int] = None,
        global_step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.forward(batch)
        logits = out['logits']
        feat = F.normalize(out['feature'], dim=1)
        y = batch['y']
        domains = batch.get('domain', None)

        aug_batch = self._augment_batch(batch)
        aug_out = self.forward(aug_batch)
        aug_logits = aug_out['logits']
        aug_feat = F.normalize(aug_out['feature'], dim=1)

        per_sample_ce_aug = F.cross_entropy(aug_logits, y, reduction='none')
        ce_orig = criterion(logits, y)

        probs = torch.softmax(logits.detach(), dim=1)
        inst_conf = probs.max(dim=1).values.clamp_min(1e-6)
        ce_aug = (inst_conf * per_sample_ce_aug).mean()
        consistency = (inst_conf * (feat - aug_feat).pow(2).mean(dim=1)).mean()
        mmd_loss = self._domain_mmd(feat, domains, inst_conf)

        loss = 0.5 * ce_orig + 0.5 * ce_aug + self.dpjdg_consistency_weight * consistency + self.dpjdg_mmd_weight * mmd_loss

        out.update({
            'loss': loss,
            'ce_loss': ce_orig.detach(),
            'dpjdg_aug_ce_loss': ce_aug.detach(),
            'dpjdg_consistency_loss': consistency.detach(),
            'dpjdg_mmd_loss': mmd_loss.detach(),
            'dpjdg_consistency_weight': logits.new_tensor(float(self.dpjdg_consistency_weight)),
            'dpjdg_mmd_weight': logits.new_tensor(float(self.dpjdg_mmd_weight)),
        })
        return out
