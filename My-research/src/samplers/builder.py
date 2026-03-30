# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional

from torch.utils.data import BatchSampler, Dataset

from .domain_batch_sampler import MetaDomainBatchSampler
from .asym_episode_sampler import AsymEpisodeBatchSampler


def build_train_batch_sampler(
    dataset: Dataset,
    sampler_cfg: Optional[Dict[str, Any]],
    batch_size: int,
    seed: int = 42
) -> BatchSampler | None:
    sampler_cfg = sampler_cfg or {}
    if not bool(sampler_cfg.get("enabled", False)):
        return None

    name = str(sampler_cfg.get("name", "meta_domain_batch_sampler")).lower()

    if name == "meta_domain_batch_sampler":
        return MetaDomainBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            domains_per_batch=int(sampler_cfg.get("domains_per_batch", 2)),
            samples_per_domain=sampler_cfg.get("samples_per_domain", None),
            shuffle=bool(sampler_cfg.get("shuffle", True)),
            drop_last=bool(sampler_cfg.get("drop_last", False)),
            seed=int(sampler_cfg.get("seed", seed)),
        )

    if name == "asym_episode_sampler":
        return AsymEpisodeBatchSampler(
            dataset=dataset,
            support_domains=int(sampler_cfg.get("support_domains", 2)),
            support_samples_per_domain=int(sampler_cfg.get("support_samples_per_domain", 12)),
            query_samples_per_domain=int(sampler_cfg.get("query_samples_per_domain", 8)),
            support_fault_ratio=float(sampler_cfg.get("support_fault_ratio", 0.33)),
            query_min_fault=int(sampler_cfg.get("query_min_fault", 1)),
            query_temp=float(sampler_cfg.get("query_temp", 0.9)),
            normal_label=int(sampler_cfg.get("normal_label", 0)),
            shuffle=bool(sampler_cfg.get("shuffle", True)),
            drop_last=bool(sampler_cfg.get("drop_last", False)),
            seed=int(sampler_cfg.get("seed", seed)),
            debug=bool(sampler_cfg.get("debug", False)),
            debug_max_batches=int(sampler_cfg.get("debug_max_batches", 10)),
        )

    raise ValueError(f"Unsupported sampler name: {name}")