# -*- coding: utf-8 -*-
"""GroupDRO baseline for strict domain generalization.

Reference:
    Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, Percy Liang,
    "Distributionally Robust Neural Networks for Group Shifts:
    On the Importance of Regularization for Worst-Case Generalization," ICLR 2020.

This implementation uses source-domain labels as groups. For each mini-batch:
- compute one loss per source domain
- update adversarial group weights q on the domains appearing in the batch
- optimize the weighted sum of domain losses

It fits the current strict DG pipeline because only source-domain training data are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import BaseDGClassifier, BaseDGConfig


@dataclass
class GroupDROConfig(BaseDGConfig):
    groupdro_eta: float = 0.01


class GroupDROClassifier(BaseDGClassifier):
    def __init__(self, cfg: GroupDROConfig) -> None:
        super().__init__(cfg)
        self.groupdro_eta = float(cfg.groupdro_eta)
        # lazily initialized because domain ids come from the h5 file dynamically
        self.register_buffer('q', torch.empty(0), persistent=False)

    def _ensure_q_size(self, max_domain_id: int, device: torch.device) -> None:
        needed = int(max_domain_id) + 1
        if self.q.numel() >= needed:
            if self.q.device != device:
                self.q = self.q.to(device)
            return
        old_q = self.q.detach().to(device) if self.q.numel() > 0 else torch.empty(0, device=device)
        new_q = torch.ones(needed, dtype=torch.float32, device=device)
        if old_q.numel() > 0:
            new_q[: old_q.numel()] = old_q
        new_q = new_q / new_q.sum()
        self.q = new_q

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
            out.update({
                'loss': ce,
                'ce_loss': ce.detach(),
                'groupdro_worst_loss': ce.detach(),
                'groupdro_num_domains': logits.new_tensor(0.0),
            })
            return out

        unique_domains = torch.unique(domains)
        if int(unique_domains.numel()) == 0:
            ce = criterion(logits, y)
            out.update({
                'loss': ce,
                'ce_loss': ce.detach(),
                'groupdro_worst_loss': ce.detach(),
                'groupdro_num_domains': logits.new_tensor(0.0),
            })
            return out

        self._ensure_q_size(int(unique_domains.max().item()), logits.device)

        env_losses = []
        dom_ids = []
        for dom in unique_domains:
            mask = domains == dom
            if int(mask.sum().item()) == 0:
                continue
            env_loss = criterion(logits[mask], y[mask])
            env_losses.append(env_loss)
            dom_ids.append(int(dom.item()))

        if not env_losses:
            ce = criterion(logits, y)
            out.update({
                'loss': ce,
                'ce_loss': ce.detach(),
                'groupdro_worst_loss': ce.detach(),
                'groupdro_num_domains': logits.new_tensor(0.0),
            })
            return out

        env_losses_t = torch.stack(env_losses)
        dom_ids_t = torch.tensor(dom_ids, dtype=torch.long, device=logits.device)

        # Update adversarial weights only on the groups present in the current batch.
        with torch.no_grad():
            self.q[dom_ids_t] = self.q[dom_ids_t] * torch.exp(self.groupdro_eta * env_losses_t.detach())
            self.q = self.q / self.q.sum().clamp_min(1e-12)

        q_batch = self.q[dom_ids_t]
        q_batch = q_batch / q_batch.sum().clamp_min(1e-12)
        robust_loss = torch.sum(q_batch * env_losses_t)
        ce_mean = env_losses_t.mean()
        worst_loss = env_losses_t.max()

        out.update({
            'loss': robust_loss,
            'ce_loss': ce_mean.detach(),
            'groupdro_worst_loss': worst_loss.detach(),
            'groupdro_num_domains': logits.new_tensor(float(len(dom_ids))),
            'groupdro_eta': logits.new_tensor(float(self.groupdro_eta)),
        })
        return out
