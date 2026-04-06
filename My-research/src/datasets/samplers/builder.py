# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional
from torch.utils.data import BatchSampler, Dataset
from .domain_class_balanced_batch_sampler import DomainClassBalancedBatchSampler


def build_train_batch_sampler(
    dataset,
    sampler_cfg,
    batch_size: int,
    seed: int = 42,
):
    sampler_cfg = sampler_cfg or {}
    if not bool(sampler_cfg.get("enabled", False)):
        print("enabled: false")
        return None

    return DomainClassBalancedBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        domains_per_batch=int(sampler_cfg.get("domains_per_batch", 3)),
        per_class_per_domain=int(sampler_cfg.get("per_class_per_domain", 2)),
        seed=seed,
        drop_last=bool(sampler_cfg.get("drop_last", True)),
        domain_weights=sampler_cfg.get("domain_weights", None),
    )