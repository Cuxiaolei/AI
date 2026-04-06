# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class AsymMetaSplitConfig:
    debug: bool = False
    debug_max_steps: int = 20


class AsymMetaSplitter:
    """按当前 batch 中“样本数最少的域”切出 query 域"""

    def __init__(self, cfg: AsymMetaSplitConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def subset_batch(batch: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            if torch.is_tensor(v) and v.ndim > 0 and v.size(0) == mask.size(0):
                out[k] = v[mask]
            else:
                out[k] = v
        return out

    def split(
        self,
        batch: Dict[str, torch.Tensor],
        step: int = 0,
    ) -> Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]:
        domains = batch.get("domain", None)
        if domains is None:
            return None

        unique_domains, counts = torch.unique(domains, return_counts=True)
        if int(unique_domains.numel()) < 2:
            return None

        # query 域就是当前 batch 中样本数最少的那个域
        query_domain = unique_domains[torch.argmin(counts)]
        support_domains = unique_domains[unique_domains != query_domain]

        query_mask = (domains == query_domain)
        support_mask = ~query_mask

        if int(query_mask.sum().item()) == 0 or int(support_mask.sum().item()) == 0:
            return None

        if self.cfg.debug and step < self.cfg.debug_max_steps:
            print(
                f"[AsymMetaSplit][Step {step}] "
                f"support_domains={support_domains.detach().cpu().tolist()} "
                f"query_domain={int(query_domain.item())} "
                f"support_size={int(support_mask.sum().item())} "
                f"query_size={int(query_mask.sum().item())}"
            )

        return (
            self.subset_batch(batch, support_mask),
            self.subset_batch(batch, query_mask),
            support_domains,
            query_domain.view(1),
        )