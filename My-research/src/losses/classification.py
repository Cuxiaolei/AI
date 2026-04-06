# -*- coding: utf-8 -*-
"""Reusable classification losses."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
    raise ValueError(f'Unsupported loss: {name}')


__all__ = ['compute_class_weights_from_loader', 'build_classification_loss']
