# -*- coding: utf-8 -*-
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any

import torch
from torch import Tensor


@dataclass
class AsymMetaSplitConfig:
    train_per_class: int = 2
    test_per_class: int = 2
    random_query_domain: bool = True
    seed: int = 42
    debug: bool = False
    debug_max_steps: int = 20


class AsymMetaSplitter:
    def __init__(self, cfg: AsymMetaSplitConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def subset_batch_by_indices(
        batch: Dict[str, torch.Tensor],
        indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            if torch.is_tensor(v) and v.ndim > 0 and v.size(0) == batch["y"].size(0):
                out[k] = v.index_select(0, indices)
            else:
                out[k] = v
        return out

    @staticmethod
    def _sample_class_indices(
        y: torch.Tensor,
        domain: torch.Tensor,
        domain_ids: List[int],
        num_classes: int,
        per_class: int,
    ) -> torch.Tensor:
        selected: List[torch.Tensor] = []

        for d in domain_ids:
            for c in range(num_classes):
                idx = ((domain == int(d)) & (y == c)).nonzero(as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue

                idx = idx[torch.randperm(idx.numel(), device=idx.device)]
                take = min(per_class, idx.numel())
                if take > 0:
                    selected.append(idx[:take])

        if len(selected) == 0:
            return torch.empty(0, dtype=torch.long, device=y.device)

        return torch.cat(selected, dim=0)

    def split(
        self,
        batch: Dict[str, torch.Tensor],
        step: int = 0,
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        domains = batch.get("domain", None)
        y = batch.get("y", None)

        if domains is None or y is None:
            return None

        unique_domains, counts = torch.unique(domains, sorted=True, return_counts=True)
        if int(unique_domains.numel()) < 2:
            return None

        domain_list = unique_domains.detach().cpu().tolist()

        # 1. 选 query 域
        if self.cfg.random_query_domain:
            rng = random.Random(int(self.cfg.seed) + int(step))
            query_domain_id = rng.choice(domain_list)
            query_domain = torch.tensor(query_domain_id, device=domains.device, dtype=domains.dtype)
        else:
            query_domain = unique_domains[torch.argmin(counts)]

        support_domains = unique_domains[unique_domains != query_domain]
        support_domain_list = support_domains.detach().cpu().tolist()

        # 2. 按类抽样形成 meta_train_batch / meta_test_batch
        num_classes = int(torch.max(y).item()) + 1

        meta_train_idx = self._sample_class_indices(
            y=y,
            domain=domains,
            domain_ids=support_domain_list,
            num_classes=num_classes,
            per_class=int(self.cfg.train_per_class),
        )

        meta_test_idx = self._sample_class_indices(
            y=y,
            domain=domains,
            domain_ids=[int(query_domain.item())],
            num_classes=num_classes,
            per_class=int(self.cfg.test_per_class),
        )

        if meta_train_idx.numel() == 0 or meta_test_idx.numel() == 0:
            return None

        if self.cfg.debug and step < self.cfg.debug_max_steps:
            print(
                f"[AsymMetaSplitV2][Step {step}] "
                f"support_domains={support_domains.detach().cpu().tolist()} "
                f"query_domain={int(query_domain.item())} "
                f"meta_train_size={int(meta_train_idx.numel())} "
                f"meta_test_size={int(meta_test_idx.numel())}"
            )

        support_batch = self.subset_batch_by_indices(batch, meta_train_idx)

        # 从 support_batch 里再切出 support / query_train
        support_y = support_batch["y"]
        support_domain = support_batch["domain"]
        num_classes = int(torch.max(support_y).item()) + 1

        support_idx_list = []
        query_train_idx_list = []

        for d in support_domains.detach().cpu().tolist():
            for c in range(num_classes):
                idx = ((support_domain == int(d)) & (support_y == c)).nonzero(as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                idx = idx[torch.randperm(idx.numel(), device=idx.device)]

                n_sup = min(1, idx.numel())
                n_qtr = min(1, max(idx.numel() - n_sup, 0))

                if n_sup > 0:
                    support_idx_list.append(idx[:n_sup])
                if n_qtr > 0:
                    query_train_idx_list.append(idx[n_sup:n_sup + n_qtr])

        if len(support_idx_list) == 0 or len(query_train_idx_list) == 0:
            return None

        support_idx = torch.cat(support_idx_list, dim=0)
        query_train_idx = torch.cat(query_train_idx_list, dim=0)

        query_meta_batch = self.subset_batch_by_indices(batch, meta_test_idx)

        return {
            "support": self.subset_batch_by_indices(support_batch, support_idx),
            "query_train": self.subset_batch_by_indices(support_batch, query_train_idx),
            "query_meta": query_meta_batch,
            "support_domains": support_domains,
            "query_domain": query_domain.view(1),
        }
