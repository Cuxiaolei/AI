# -*- coding: utf-8 -*-
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch


@dataclass
class AsymMetaSplitConfig:
    train_per_class: int = 2
    test_per_class: int = 2
    random_query_domain: bool = True
    seed: int = 42
    debug: bool = False
    debug_max_steps: int = 20


class AsymMetaSplitter:
    """
    兼容当前 compute_loss() 的增强版 splitter

    返回:
        meta_train_batch, meta_test_batch, support_domains, query_domain

    逻辑:
        1. 从当前 batch 的多个域中选 1 个 query 域
        2. 剩余域作为 support 域
        3. support 域内部按类抽样，形成 meta_train_batch
        4. query 域内部按类抽样，形成 meta_test_batch
    """

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
    ) -> Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]:
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

        return (
            self.subset_batch_by_indices(batch, meta_train_idx),
            self.subset_batch_by_indices(batch, meta_test_idx),
            support_domains,
            query_domain.view(1),
        )