# -*- coding: utf-8 -*-
from __future__ import annotations

import torch


def negative_sq_logits(
    feat: torch.Tensor,
    proto_bank: torch.Tensor,
    inverse_domain_index: torch.Tensor,
) -> torch.Tensor:
    # feat:         [B, C]
    # proto_bank:   [D, K, C]
    # inverse_domain_index: [B], mapping each sample to local domain index in [0, D-1]
    # return logits: [B, K]
    sample_proto = proto_bank[inverse_domain_index]              # [B, K, C]
    logits = -((feat.unsqueeze(1) - sample_proto) ** 2).sum(dim=-1)
    return logits


def empirical_prototypes(
    feat: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    unique_domains: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # proto_emp: [D, K, C]
    # valid_mask: [D, K] bool
    d, c = unique_domains.numel(), feat.size(-1)
    proto_emp = feat.new_zeros(d, num_classes, c)
    valid_mask = torch.zeros(d, num_classes, dtype=torch.bool, device=feat.device)

    for i, domain_id in enumerate(unique_domains.tolist()):
        domain_mask = (domains == int(domain_id))
        for k in range(num_classes):
            mask = domain_mask & (labels == k)
            if mask.any():
                proto_emp[i, k] = feat[mask].mean(dim=0)
                valid_mask[i, k] = True
    return proto_emp, valid_mask






# 元学习与原型学习融合
def global_empirical_prototypes(
    feat: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    c = feat.size(-1)
    proto_emp = feat.new_zeros(num_classes, c)
    valid_mask = torch.zeros(num_classes, dtype=torch.bool, device=feat.device)

    for k in range(num_classes):
        mask = (labels == k)
        if mask.any():
            proto_emp[k] = feat[mask].mean(dim=0)
            proto_emp[k] = torch.nn.functional.normalize(proto_emp[k], dim=-1)
            valid_mask[k] = True

    return proto_emp, valid_mask

def fuse_proto_bank(
    proto_dyn: torch.Tensor,
    proto_emp: torch.Tensor,
    valid_mask: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    emp_expand = proto_emp.unsqueeze(0).expand(proto_dyn.size(0), -1, -1)
    valid_expand = valid_mask.view(1, -1, 1).expand_as(emp_expand)

    mixed = (1.0 - beta) * proto_dyn + beta * emp_expand
    proto_fused = torch.where(valid_expand, mixed, proto_dyn)
    proto_fused = torch.nn.functional.normalize(proto_fused, dim=-1)
    return proto_fused

def negative_sq_logits_by_domain(
    feat: torch.Tensor,
    proto_bank: torch.Tensor,
    sample_domains: torch.Tensor,
    proto_domains: torch.Tensor,
) -> torch.Tensor:
    local_index = torch.full_like(sample_domains, fill_value=-1)

    for i, d in enumerate(proto_domains.tolist()):
        local_index[sample_domains == int(d)] = i

    sample_proto = proto_bank[local_index]
    logits = -((feat.unsqueeze(1) - sample_proto) ** 2).sum(dim=-1)
    return logits