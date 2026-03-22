# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    """
    Encode physical condition vector into condition embedding.
    Input dim is fixed to 3:
        [speed_rpm_norm, torque_nm_norm, radial_force_n_norm]
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, cond_vec: torch.Tensor) -> torch.Tensor:
        return self.net(cond_vec)