# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class AsymMetaSplitConfig:
    debug: bool = False
    debug_max_steps: int = 20

    query_ratio: float = 0.35
    min_query_size: int = 4
    max_query_size: int = 8
    minority_first: bool = True


class AsymMetaSplitter:
    """按当前 batch 中“样本数最少的域”切出 query 域，
    并在 query 域内部执行“少数类优先”的 meta-test 划分。
    """

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

    @staticmethod
    def _minority_first_pick(local_labels: torch.Tensor, pick_size: int) -> torch.Tensor:
        """
        在 query 域局部标签中，优先挑选少数类样本。
        返回的是 query 域内部的局部 bool mask。
        """
        n = int(local_labels.size(0))
        device = local_labels.device
        picked_mask = torch.zeros(n, dtype=torch.bool, device=device)

        if n == 0 or pick_size <= 0:
            return picked_mask

        classes, counts = torch.unique(local_labels, return_counts=True)
        order = torch.argsort(counts, descending=False)  # 少数类优先
        classes = classes[order]

        picked = 0

        # 第一轮：每个出现过的类先拿 1 个
        for cls in classes.tolist():
            cls_idx = torch.nonzero(local_labels == cls, as_tuple=False).flatten()
            if cls_idx.numel() == 0:
                continue
            idx = cls_idx[0]
            if not picked_mask[idx]:
                picked_mask[idx] = True
                picked += 1
            if picked >= pick_size:
                return picked_mask

        # 第二轮：继续按少数类优先补满
        for cls in classes.tolist():
            cls_idx = torch.nonzero(local_labels == cls, as_tuple=False).flatten()
            for idx in cls_idx.tolist():
                if not picked_mask[idx]:
                    picked_mask[idx] = True
                    picked += 1
                    if picked >= pick_size:
                        return picked_mask

        return picked_mask

    def split(
        self,
        batch: Dict[str, torch.Tensor],
        step: int = 0,
    ) -> Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]:
        domains = batch.get("domain", None)
        labels = batch.get("y", None)
        if domains is None or labels is None:
            return None

        unique_domains, counts = torch.unique(domains, return_counts=True)
        if int(unique_domains.numel()) < 2:
            return None

        # 仍然保留：query 域 = 当前 batch 中样本数最少的域
        query_domain = unique_domains[torch.argmin(counts)]
        support_domains = unique_domains[unique_domains != query_domain]

        query_mask = (domains == query_domain)
        support_mask = ~query_mask

        query_size = int(query_mask.sum().item())
        support_size = int(support_mask.sum().item())

        if query_size <= 1 or support_size == 0:
            return None

        # query 域内部样本
        query_labels = labels[query_mask]

        # 计算 meta-test 大小
        raw_pick = int(round(query_size * float(self.cfg.query_ratio)))
        pick_size = max(int(self.cfg.min_query_size), raw_pick)
        pick_size = min(pick_size, int(self.cfg.max_query_size))
        pick_size = min(pick_size, query_size - 1)  # 至少给 meta-train 留 1 个 query 样本

        if pick_size <= 0:
            return None

        # 在 query 域内部执行“少数类优先”挑选
        if self.cfg.minority_first:
            query_meta_test_local_mask = self._minority_first_pick(query_labels, pick_size)
        else:
            query_meta_test_local_mask = torch.zeros_like(query_labels, dtype=torch.bool)
            query_meta_test_local_mask[:pick_size] = True

        if int(query_meta_test_local_mask.sum().item()) == 0:
            return None


        # 把局部 mask 映射回全 batch mask
        meta_test_mask = torch.zeros_like(query_mask, dtype=torch.bool)
        query_global_idx = torch.nonzero(query_mask, as_tuple=False).flatten()
        meta_test_global_idx = query_global_idx[query_meta_test_local_mask]
        meta_test_mask[meta_test_global_idx] = True

        # meta-train = support 域全部 + query 域剩余样本
        meta_train_mask = ~meta_test_mask

        meta_train_size = int(meta_train_mask.sum().item())
        meta_test_size = int(meta_test_mask.sum().item())

        if meta_train_size == 0 or meta_test_size == 0:
            # fallback: 使用旧版整域 query
            return (
                self.subset_batch(batch, support_mask),
                self.subset_batch(batch, query_mask),
                support_domains,
                query_domain.view(1),
            )

        if self.cfg.debug and step < self.cfg.debug_max_steps:
            q_classes, q_counts = torch.unique(query_labels, return_counts=True)
            print(
                f"[AsymMetaSplit][Step {step}] "
                f"support_domains={support_domains.detach().cpu().tolist()} "
                f"query_domain={int(query_domain.item())} "
                f"query_domain_class_counts="
                f"{dict(zip(q_classes.detach().cpu().tolist(), q_counts.detach().cpu().tolist()))} "
                f"meta_train_size={meta_train_size} "
                f"meta_test_size={meta_test_size}"
            )

        return (
            self.subset_batch(batch, meta_train_mask),
            self.subset_batch(batch, meta_test_mask),
            support_domains,
            query_domain.view(1),
        )