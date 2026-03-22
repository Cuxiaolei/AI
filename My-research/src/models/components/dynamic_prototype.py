# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicPrototypeGenerator(nn.Module):
    """
    Residual dynamic prototype:
        p_{k,d} = e_k + alpha * Delta(e_k, c_d)
    """
    def __init__(self, feat_dim: int = 128, hidden_dim: int = 256, alpha: float = 0.2) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim),
        )

    def forward(self, class_anchor: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        """
        class_anchor: [K, C]
        cond_emb:     [D, C]
        return:       [D, K, C]
        """
        d = cond_emb.size(0)
        k = class_anchor.size(0)
        c = class_anchor.size(1)

        anchor_expand = class_anchor.unsqueeze(0).expand(d, k, c)
        cond_expand = cond_emb.unsqueeze(1).expand(d, k, c)

        delta = self.mlp(torch.cat([anchor_expand, cond_expand], dim=-1))
        proto = anchor_expand + self.alpha * delta
        return F.normalize(proto, dim=-1)