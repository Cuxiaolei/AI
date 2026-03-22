# -*- coding: utf-8 -*-
"""ERM baseline model for unified H5 inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import BaseDGClassifier, BaseDGConfig


@dataclass
class ERMConfig(BaseDGConfig):
    pass


class ERMClassifier(BaseDGClassifier):
    def __init__(self, cfg: ERMConfig) -> None:
        super().__init__(cfg)

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        criterion: nn.Module,
        epoch: Optional[int] = None,
        global_step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.forward(batch)
        ce = criterion(out['logits'], batch['y'])
        out.update({'loss': ce, 'ce_loss': ce.detach()})
        return out
