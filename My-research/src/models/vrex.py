# -*- coding: utf-8 -*-
"""VREx baseline for strict domain generalization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import BaseDGClassifier, BaseDGConfig


@dataclass
class VRExConfig(BaseDGConfig):
    vrex_lambda: float = 1.0
    vrex_penalty_anneal_iters: int = 0


class VRExClassifier(BaseDGClassifier):
    def __init__(self, cfg: VRExConfig) -> None:
        super().__init__(cfg)
        self.vrex_lambda = float(cfg.vrex_lambda)
        self.vrex_penalty_anneal_iters = int(cfg.vrex_penalty_anneal_iters)

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        criterion: nn.Module,
        epoch: Optional[int] = None,
        global_step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.forward(batch)
        logits = out['logits']
        y = batch['y']
        domains = batch.get('domain', None)

        if domains is None:
            ce = criterion(logits, y)
            penalty = logits.new_tensor(0.0)
            loss = ce
            out.update({'loss': loss, 'ce_loss': ce.detach(), 'vrex_penalty': penalty.detach(), 'vrex_lambda': logits.new_tensor(0.0)})
            return out

        unique_domains = torch.unique(domains)
        env_losses = []
        for dom in unique_domains:
            mask = domains == dom
            if int(mask.sum().item()) == 0:
                continue
            env_losses.append(criterion(logits[mask], y[mask]))

        if not env_losses:
            ce = criterion(logits, y)
            penalty = logits.new_tensor(0.0)
        else:
            env_losses = torch.stack(env_losses)
            ce = env_losses.mean()
            penalty = env_losses.var(unbiased=False) if env_losses.numel() > 1 else logits.new_tensor(0.0)

        apply_lambda = self.vrex_lambda
        if self.vrex_penalty_anneal_iters > 0 and global_step is not None and global_step < self.vrex_penalty_anneal_iters:
            apply_lambda = 1.0

        loss = ce + apply_lambda * penalty
        out.update({
            'loss': loss,
            'ce_loss': ce.detach(),
            'vrex_penalty': penalty.detach(),
            'vrex_lambda': logits.new_tensor(float(apply_lambda)),
        })
        return out
