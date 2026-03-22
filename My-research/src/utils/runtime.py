# -*- coding: utf-8 -*-
"""Runtime helpers such as seed, device, and batch transfer."""
from __future__ import annotations

import random
from typing import Dict

import numpy as np
import torch


try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = 'auto') -> torch.device:
    name = str(name).lower()
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


__all__ = ['set_seed', 'get_device', 'move_batch_to_device', 'tqdm']
