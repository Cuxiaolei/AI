# -*- coding: utf-8 -*-
"""MLDG baseline for strict domain generalization.

Reference:
    Da Li, Yongxin Yang, Yi-Zhe Song, Timothy M. Hospedales,
    "Learning to Generalize: Meta-Learning for Domain Generalization," AAAI 2018.

This implementation follows a practical source-only MLDG style:
- domains inside a source-only training batch are split into meta-train and meta-test domains
- one differentiable inner gradient step is taken on meta-train domains
- meta-test loss is evaluated with fast weights
- final objective = meta_train_loss + beta * meta_test_loss

It reuses the existing backbone/classifier/data pipeline and only changes the per-step loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

try:
    from torch.func import functional_call
except Exception:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call  # type: ignore

from .base import BaseDGClassifier, BaseDGConfig


@dataclass
class MLDGConfig(BaseDGConfig):
    mldg_beta: float = 1.0
    mldg_inner_lr: float = 1e-2
    mldg_meta_test_domains: int = 1
    mldg_first_order: bool = False


class MLDGClassifier(BaseDGClassifier):
    def __init__(self, cfg: MLDGConfig) -> None:
        super().__init__(cfg)
        self.mldg_beta = float(cfg.mldg_beta)
        self.mldg_inner_lr = float(cfg.mldg_inner_lr)
        self.mldg_meta_test_domains = int(cfg.mldg_meta_test_domains)
        self.mldg_first_order = bool(cfg.mldg_first_order)

    @staticmethod
    def _subset_batch(batch: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            if torch.is_tensor(v) and v.size(0) == mask.size(0):
                out[k] = v[mask]
            else:
                out[k] = v
        return out

    def _forward_functional(self, batch: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return functional_call(self, params, (batch,))

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        criterion: nn.Module,
        epoch: Optional[int] = None,
        global_step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        # Normal forward for logging/prediction on the current parameters.
        out = self.forward(batch)
        logits = out['logits']
        y = batch['y']
        domains = batch.get('domain', None)

        # Fallback to ERM if there are not enough domains in this batch.
        if domains is None:
            ce = criterion(logits, y)
            out.update({
                'loss': ce,
                'meta_train_loss': ce.detach(),
                'meta_test_loss': logits.new_tensor(0.0),
                'mldg_beta': logits.new_tensor(0.0),
            })
            return out

        unique_domains = torch.unique(domains)
        if int(unique_domains.numel()) < 2:
            ce = criterion(logits, y)
            out.update({
                'loss': ce,
                'meta_train_loss': ce.detach(),
                'meta_test_loss': logits.new_tensor(0.0),
                'mldg_beta': logits.new_tensor(0.0),
            })
            return out

        # Randomly choose meta-test domains inside the source-only batch.
        n_meta_test = max(1, min(self.mldg_meta_test_domains, int(unique_domains.numel()) - 1))
        perm = torch.randperm(int(unique_domains.numel()), device=unique_domains.device)
        meta_test_domains = unique_domains[perm[:n_meta_test]]
        meta_train_domains = unique_domains[perm[n_meta_test:]]

        meta_test_mask = torch.zeros_like(domains, dtype=torch.bool)
        for d in meta_test_domains:
            meta_test_mask |= (domains == d)
        meta_train_mask = ~meta_test_mask

        if int(meta_train_mask.sum().item()) == 0 or int(meta_test_mask.sum().item()) == 0:
            ce = criterion(logits, y)
            out.update({
                'loss': ce,
                'meta_train_loss': ce.detach(),
                'meta_test_loss': logits.new_tensor(0.0),
                'mldg_beta': logits.new_tensor(0.0),
            })
            return out

        batch_train = self._subset_batch(batch, meta_train_mask)
        batch_test = self._subset_batch(batch, meta_test_mask)

        meta_train_out = self.forward(batch_train)
        meta_train_loss = criterion(meta_train_out['logits'], batch_train['y'])

        named_params = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        if not named_params:
            loss = meta_train_loss
            out.update({
                'loss': loss,
                'meta_train_loss': meta_train_loss.detach(),
                'meta_test_loss': logits.new_tensor(0.0),
                'mldg_beta': logits.new_tensor(float(self.mldg_beta)),
            })
            return out

        param_names = [n for n, _ in named_params]
        param_tensors = [p for _, p in named_params]
        grads = torch.autograd.grad(
            meta_train_loss,
            param_tensors,
            create_graph=not self.mldg_first_order,
            allow_unused=True,
        )

        fast_params: Dict[str, torch.Tensor] = {}
        for name, param, grad in zip(param_names, param_tensors, grads):
            if grad is None:
                fast_params[name] = param
            else:
                if self.mldg_first_order:
                    grad = grad.detach()
                fast_params[name] = param - self.mldg_inner_lr * grad

        meta_test_out = self._forward_functional(batch_test, fast_params)
        meta_test_loss = criterion(meta_test_out['logits'], batch_test['y'])

        loss = meta_train_loss + self.mldg_beta * meta_test_loss
        out.update({
            'loss': loss,
            'meta_train_loss': meta_train_loss.detach(),
            'meta_test_loss': meta_test_loss.detach(),
            'mldg_beta': logits.new_tensor(float(self.mldg_beta)),
            'num_meta_train_domains': logits.new_tensor(int(meta_train_domains.numel()), dtype=torch.float32),
            'num_meta_test_domains': logits.new_tensor(int(meta_test_domains.numel()), dtype=torch.float32),
        })
        return out
