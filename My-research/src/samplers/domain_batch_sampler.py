# -*- coding: utf-8 -*-
"""Common batch sampler for source-only meta-learning.

This sampler tries to ensure that each training batch contains samples from
multiple source domains, which is important for methods such as MLDG and for
future source-only episodic/meta-learning methods.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Sequence

from torch.utils.data import BatchSampler, Dataset


class MetaDomainBatchSampler(BatchSampler):
    """Build mini-batches with multiple source domains.

    Parameters
    ----------
    dataset:
        Dataset that provides ``get_all_domains()``.
    batch_size:
        Total batch size.
    domains_per_batch:
        Number of distinct domains to include in one batch.
    samples_per_domain:
        Optional number of samples per chosen domain. If ``None``, it is
        computed as ``batch_size // domains_per_batch``.
    shuffle:
        Whether to shuffle domain order and domain-local indices each epoch.
    drop_last:
        Whether to drop incomplete final batches.
    seed:
        Base random seed.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        domains_per_batch: int = 2,
        samples_per_domain: int | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        if not hasattr(dataset, 'get_all_domains'):
            raise ValueError('MetaDomainBatchSampler requires dataset.get_all_domains().')
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.domains_per_batch = int(domains_per_batch)
        self.samples_per_domain = samples_per_domain
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        domains = list(map(int, dataset.get_all_domains().tolist()))
        self.domain_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, d in enumerate(domains):
            self.domain_to_indices[d].append(idx)
        self.domain_ids = sorted(self.domain_to_indices.keys())
        if len(self.domain_ids) < 2:
            raise ValueError('MetaDomainBatchSampler needs at least 2 source domains in train dataset.')
        if self.domains_per_batch < 2:
            raise ValueError('domains_per_batch should be >= 2 for source-only meta-learning.')
        if self.domains_per_batch > len(self.domain_ids):
            self.domains_per_batch = len(self.domain_ids)

        if self.samples_per_domain is None:
            self.samples_per_domain = max(1, self.batch_size // self.domains_per_batch)
        self.effective_batch_size = self.samples_per_domain * self.domains_per_batch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_pools(self, rng: random.Random) -> Dict[int, List[int]]:
        pools: Dict[int, List[int]] = {}
        for d, indices in self.domain_to_indices.items():
            items = list(indices)
            if self.shuffle:
                rng.shuffle(items)
            pools[d] = items
        return pools

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        pools = self._build_pools(rng)
        domain_cycle = list(self.domain_ids)
        if self.shuffle:
            rng.shuffle(domain_cycle)

        remaining = sum(len(v) for v in pools.values())
        while remaining > 0:
            # choose distinct domains for current batch
            if self.shuffle:
                chosen_domains = rng.sample(self.domain_ids, k=self.domains_per_batch)
            else:
                chosen_domains = domain_cycle[:self.domains_per_batch]
                domain_cycle = domain_cycle[self.domains_per_batch:] + chosen_domains

            batch: List[int] = []
            for d in chosen_domains:
                need = self.samples_per_domain
                take: List[int] = []
                while need > 0:
                    if len(pools[d]) == 0:
                        # refill this domain pool to allow oversampling when needed
                        pools[d] = list(self.domain_to_indices[d])
                        if self.shuffle:
                            rng.shuffle(pools[d])
                    n_take = min(need, len(pools[d]))
                    take.extend(pools[d][:n_take])
                    pools[d] = pools[d][n_take:]
                    need -= n_take
                batch.extend(take)

            if self.shuffle:
                rng.shuffle(batch)

            if len(batch) < self.effective_batch_size and self.drop_last:
                break
            if len(batch) > 0:
                yield batch

            # rough progress accounting
            remaining -= min(self.effective_batch_size, remaining)

    def __len__(self) -> int:
        n = len(self.dataset)
        bs = max(1, self.effective_batch_size)
        if self.drop_last:
            return n // bs
        return math.ceil(n / bs)
