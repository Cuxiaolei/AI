# -*- coding: utf-8 -*-
"""Reusable classification losses."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class FocalLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean') -> None:
        super().__init__()
        self.register_buffer('weight', weight if weight is not None else None)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


@torch.no_grad()
def compute_class_weights_from_loader(loader: DataLoader, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float64)
    dataset = loader.dataset
    if hasattr(dataset, 'get_all_labels'):
        labels = torch.as_tensor(dataset.get_all_labels(), dtype=torch.long)
        counts += torch.bincount(labels, minlength=num_classes).to(torch.float64)
    else:
        for batch in loader:
            y = batch['y'].view(-1).long()
            counts += torch.bincount(y, minlength=num_classes).to(torch.float64)
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / counts
    return (weights / weights.mean()).to(torch.float32)


def build_classification_loss(cfg: dict, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    name = str(cfg.get('name', 'cross_entropy')).lower()
    if name in {'cross_entropy', 'ce'}:
        return nn.CrossEntropyLoss(weight=class_weights)
    if name in {'focal', 'focalloss'}:
        gamma = float(cfg.get('gamma', 2.0))
        return FocalLoss(weight=class_weights, gamma=gamma)
    raise ValueError(f'Unsupported loss: {name}')


__all__ = ['FocalLoss', 'compute_class_weights_from_loader', 'build_classification_loss']
