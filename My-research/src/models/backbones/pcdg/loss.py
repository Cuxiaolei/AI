# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeMemory:
    def __init__(self, num_classes: int, emb_dim: int, momentum: float = 0.9, device: str = "cuda"):
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.momentum = momentum
        self.device = device
        self.proto = torch.zeros(num_classes, emb_dim, device=device)
        self.seen = torch.zeros(num_classes, dtype=torch.bool, device=device)

    def train_mode(self):
        pass

    def update(self, class_ids: torch.Tensor, protos: torch.Tensor):
        # class_ids: (K,) global ids; protos: (K,D)
        for cid, p in zip(class_ids.tolist(), protos):
            if cid < 0 or cid >= self.num_classes:
                continue
            if not self.seen[cid]:
                self.proto[cid] = p
                self.seen[cid] = True
            else:
                self.proto[cid] = self.momentum * self.proto[cid] + (1 - self.momentum) * p

    def get(self, class_ids: torch.Tensor) -> torch.Tensor:
        # returns (K,D)
        return self.proto[class_ids]


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.t = float(temperature)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # z: (B,D) normalized; y: (B,)
        B = z.size(0)
        sim = torch.mm(z, z.t()) / self.t  # (B,B)
        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()

        mask = torch.eq(y.view(-1, 1), y.view(1, -1)).float().to(z.device)
        logits_mask = torch.ones_like(mask) - torch.eye(B, device=z.device)
        mask = mask * logits_mask

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        return loss


class PCDGLoss(nn.Module):
    def __init__(self, model_cfg: Dict[str, Any]):
        super().__init__()
        loss_cfg = model_cfg.get("loss", {})
        self.w_proto = float(loss_cfg.get("w_proto", 1.0))
        self.w_supcon = float(loss_cfg.get("w_supcon", 0.2))
        self.w_drift = float(loss_cfg.get("w_drift", 0.1))
        self.supcon = SupConLoss(temperature=float(loss_cfg.get("temperature", 0.07)))

    def forward(self, out: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        logits = out["logits"]
        y = out["y"]
        zq = out["zq"]
        protos = out["protos"]
        class_ids = out["class_ids"]
        memory = out.get("memory", None)

        ce = F.cross_entropy(logits, y)

        # supcon on query embedding using within-episode label y
        zq_n = F.normalize(zq, dim=1)
        supcon = self.supcon(zq_n, y)

        drift = torch.tensor(0.0, device=logits.device)
        if (memory is not None) and getattr(memory, "seen", None) is not None:
            # only drift for classes already seen
            seen_mask = memory.seen[class_ids].float().view(-1, 1)
            mem_p = memory.get(class_ids).detach()
            drift = ((protos - mem_p) ** 2).mean(dim=1, keepdim=True)
            drift = (drift * seen_mask).mean()

        total = self.w_proto * ce + self.w_supcon * supcon + self.w_drift * drift
        return {"total": total, "ce": ce, "supcon": supcon, "drift": drift}
