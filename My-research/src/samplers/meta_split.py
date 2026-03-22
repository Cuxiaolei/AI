# -*- coding: utf-8 -*-
"""Common meta-task splitting utilities for source-only meta-learning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class DomainMetaSplitConfig:
    meta_test_domains: int = 1
    randomize: bool = True
    seed: int = 42


class DomainMetaSplitter:
    """Split a batch into meta-train and meta-test subsets by domain.

    This is a reusable component for source-only meta-learning methods.
    It assumes the batch contains a ``domain`` tensor of shape [B].
    """

    def __init__(self, cfg: DomainMetaSplitConfig) -> None:
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
        domains = batch.get('domain', None)
        if domains is None:
            return None
        unique_domains = torch.unique(domains)
        if int(unique_domains.numel()) < 2:
            return None

        n_meta_test = max(1, min(int(self.cfg.meta_test_domains), int(unique_domains.numel()) - 1))
        if self.cfg.randomize:
            g = torch.Generator(device=unique_domains.device)
            g.manual_seed(int(self.cfg.seed) + int(step))
            perm = torch.randperm(int(unique_domains.numel()), generator=g, device=unique_domains.device)
            unique_domains = unique_domains[perm]
        meta_test_domains = unique_domains[:n_meta_test]
        meta_train_domains = unique_domains[n_meta_test:]

        meta_test_mask = torch.zeros_like(domains, dtype=torch.bool)
        for d in meta_test_domains:
            meta_test_mask |= (domains == d)
        meta_train_mask = ~meta_test_mask

        if int(meta_train_mask.sum().item()) == 0 or int(meta_test_mask.sum().item()) == 0:
            return None

        return (
            self.subset_batch(batch, meta_train_mask),
            self.subset_batch(batch, meta_test_mask),
            meta_train_domains,
            meta_test_domains,
        )
