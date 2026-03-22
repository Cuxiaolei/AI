# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn.functional as F


def inverse_frequency_weights(labels: torch.Tensor, num_classes: int, power: float = 0.5) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = 1.0 / (counts ** power)
    weights = weights / weights.mean()
    return weights


def masked_proto_align_loss(
    proto_pred: torch.Tensor,
    proto_emp: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Align generated prototypes with empirical prototypes.
    """
    if not valid_mask.any():
        return proto_pred.new_tensor(0.0)
    diff = (proto_pred - proto_emp) ** 2
    diff = diff.sum(dim=-1)  # [D, K]
    return diff[valid_mask].mean()


def sample_prototype_contrastive_loss(
    feat: torch.Tensor,
    labels: torch.Tensor,
    proto_bank: torch.Tensor,
    temperature: float = 0.1,
    imbalance_power: float = 0.5,
) -> torch.Tensor:
    """
    Positive prototypes:
        all same-class prototypes across all source domains
    Negative prototypes:
        all different-class prototypes across all source domains
    """
    b, c = feat.size()
    d, k, _ = proto_bank.size()

    feat = F.normalize(feat, dim=-1)
    proto_bank = F.normalize(proto_bank, dim=-1)

    sim = torch.einsum("bc,dkc->bdk", feat, proto_bank) / temperature  # [B, D, K]
    exp_sim = torch.exp(sim)

    class_ids = torch.arange(k, device=labels.device).view(1, 1, k)
    pos_mask = (class_ids == labels.view(b, 1, 1))
    neg_mask = ~pos_mask

    pos_sum = (exp_sim * pos_mask).sum(dim=(1, 2)).clamp_min(1e-8)
    neg_sum = (exp_sim * neg_mask).sum(dim=(1, 2)).clamp_min(1e-8)
    loss = -torch.log(pos_sum / (pos_sum + neg_sum))

    class_w = inverse_frequency_weights(labels, num_classes=k, power=imbalance_power)
    sample_w = class_w[labels]
    return (loss * sample_w).mean()