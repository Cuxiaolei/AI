# -*- coding: utf-8 -*-
"""Simple reusable classification heads."""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    """A lightweight linear classifier for backbone features."""

    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.in_dim, self.num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(x))
