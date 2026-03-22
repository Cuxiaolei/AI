# -*- coding: utf-8 -*-
"""Optimizer and scheduler builders."""
from __future__ import annotations

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, cfg: dict):
    name = str(cfg.get('name', 'adam')).lower()
    lr = float(cfg.get('lr', 1e-3))
    weight_decay = float(cfg.get('weight_decay', 1e-4))
    momentum = float(cfg.get('momentum', 0.9))
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f'Unsupported optimizer: {name}')


def build_scheduler(optimizer, cfg: dict):
    name = str(cfg.get('name', 'none')).lower()
    if name == 'none':
        return None
    if name == 'cosine':
        epochs = int(cfg.get('epochs', 50))
        min_lr = float(cfg.get('min_lr', 1e-6))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=min_lr)
    if name == 'step':
        step_size = int(cfg.get('step_size', 30))
        gamma = float(cfg.get('gamma', 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    raise ValueError(f'Unsupported scheduler: {name}')


__all__ = ['build_optimizer', 'build_scheduler']
