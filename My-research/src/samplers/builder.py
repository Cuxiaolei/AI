# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional
from torch.utils.data import BatchSampler, Dataset
from .domain_batch_sampler import MetaDomainBatchSampler


def build_train_batch_sampler(
    dataset: Dataset,
    sampler_cfg: Optional[Dict[str, Any]],
    batch_size: int,
    seed: int = 42
) -> BatchSampler | None:
    sampler_cfg = sampler_cfg or {}
    if not bool(sampler_cfg.get("enabled", False)):
        print("enabled: false")
        return None

    return MetaDomainBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        domains_per_batch=int(sampler_cfg.get("domains_per_batch", 2)),
        samples_per_domain=sampler_cfg.get("samples_per_domain", None),
        shuffle=bool(sampler_cfg.get("shuffle", True)),
        drop_last=bool(sampler_cfg.get("drop_last", False)),
        seed=int(sampler_cfg.get("seed", seed)),
    )