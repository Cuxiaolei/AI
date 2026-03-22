# -*- coding: utf-8 -*-
"""IRM baseline for strict domain generalization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDGClassifier, BaseDGConfig


@dataclass
class IRMConfig(BaseDGConfig):
    irm_lambda: float = 1.0
    irm_penalty_anneal_iters: int = 0


class IRMClassifier(BaseDGClassifier):
    def __init__(self, cfg: IRMConfig) -> None:
        super().__init__(cfg)
        self.irm_lambda = float(cfg.irm_lambda)
        self.irm_penalty_anneal_iters = int(cfg.irm_penalty_anneal_iters)

    @staticmethod
    def irm_penalty(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if logits.numel() == 0:
            return logits.new_tensor(0.0)
        scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
        loss = F.cross_entropy(logits * scale, y)
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)

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
            loss = ce + penalty
            out.update({'loss': loss, 'ce_loss': ce.detach(), 'irm_penalty': penalty.detach(), 'irm_lambda': logits.new_tensor(0.0)})
            return out

        unique_domains = torch.unique(domains)
        env_losses = []
        penalties = []
        for dom in unique_domains:
            mask = domains == dom
            if int(mask.sum().item()) == 0:
                continue
            env_logits = logits[mask]
            env_y = y[mask]
            env_losses.append(criterion(env_logits, env_y))
            penalties.append(self.irm_penalty(env_logits, env_y))

        if not env_losses:
            ce = criterion(logits, y)
            penalty = logits.new_tensor(0.0)
        else:
            ce = torch.stack(env_losses).mean()
            penalty = torch.stack(penalties).mean() if penalties else logits.new_tensor(0.0)

        apply_lambda = self.irm_lambda
        if self.irm_penalty_anneal_iters > 0 and global_step is not None and global_step < self.irm_penalty_anneal_iters:
            apply_lambda = 1.0

        loss = ce + apply_lambda * penalty
        out.update({
            'loss': loss,
            'ce_loss': ce.detach(),
            'irm_penalty': penalty.detach(),
            'irm_lambda': logits.new_tensor(float(apply_lambda)),
        })
        return out
