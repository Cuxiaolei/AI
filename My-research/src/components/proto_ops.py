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
