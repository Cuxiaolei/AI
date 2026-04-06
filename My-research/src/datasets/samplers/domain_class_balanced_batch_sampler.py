# -*- coding: utf-8 -*-
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from torch.utils.data import BatchSampler, Dataset


def _to_list(x):
    if torch.is_tensor(x):
        return x.cpu().tolist()
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


class DomainClassBalancedBatchSampler(BatchSampler):
    """
    每个 batch 的逻辑
    1. 先选若干个域
    2. 每个域内按类取固定数量样本
    3. 不够 batch_size 时再补齐
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        domains_per_batch: int = 3,
        per_class_per_domain: int = 2,
        seed: int = 42,
        drop_last: bool = True,
        domain_weights: Optional[Dict[int, float]] = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.domains_per_batch = int(domains_per_batch)
        self.per_class_per_domain = int(per_class_per_domain)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        self.labels = _to_list(dataset.get_all_labels())
        self.domains = _to_list(dataset.get_all_domains())

        self.unique_domains = sorted(set(int(d) for d in self.domains))
        self.unique_classes = sorted(set(int(y) for y in self.labels))
        self.domain_weights = domain_weights or {d: 1.0 for d in self.unique_domains}

        self.pool = defaultdict(lambda: defaultdict(list))
        for idx, (y, d) in enumerate(zip(self.labels, self.domains)):
            self.pool[int(d)][int(y)].append(idx)

        self.num_samples = len(self.labels)
        self.num_batches = (
            self.num_samples // self.batch_size
            if self.drop_last
            else (self.num_samples + self.batch_size - 1) // self.batch_size
        )

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed)

        epoch_pool = defaultdict(dict)
        for d in self.unique_domains:
            for c in self.unique_classes:
                arr = list(self.pool[d][c])
                rng.shuffle(arr)
                epoch_pool[d][c] = arr

        for _ in range(self.num_batches):
            batch_indices: List[int] = []

            selected_domains = self._sample_domains(rng)

            for d in selected_domains:
                for c in self.unique_classes:
                    picked = self._pop_from_pool(
                        current_pool=epoch_pool[d][c],
                        take=self.per_class_per_domain,
                        rng=rng,
                        fallback_full_pool=self.pool[d][c],
                    )
                    batch_indices.extend(picked)

            if len(batch_indices) < self.batch_size:
                remain = self.batch_size - len(batch_indices)
                batch_indices.extend(
                    self._fill_from_selected_domains(
                        epoch_pool=epoch_pool,
                        selected_domains=selected_domains,
                        remain=remain,
                        rng=rng,
                    )
                )

            if len(batch_indices) > self.batch_size:
                rng.shuffle(batch_indices)
                batch_indices = batch_indices[:self.batch_size]

            if len(batch_indices) < self.batch_size and self.drop_last:
                continue

            yield batch_indices

    def _sample_domains(self, rng: random.Random) -> List[int]:
        domains = self.unique_domains[:]
        weights = [float(self.domain_weights.get(d, 1.0)) for d in domains]

        selected = []
        candidates = list(zip(domains, weights))
        for _ in range(self.domains_per_batch):
            total = sum(w for _, w in candidates)
            r = rng.random() * total
            acc = 0.0
            chosen_i = 0
            for i, (_, w) in enumerate(candidates):
                acc += w
                if acc >= r:
                    chosen_i = i
                    break
            chosen_d, _ = candidates.pop(chosen_i)
            selected.append(chosen_d)
        return selected

    def _pop_from_pool(
        self,
        current_pool: List[int],
        take: int,
        rng: random.Random,
        fallback_full_pool: List[int],
    ) -> List[int]:
        out = []
        while len(out) < take:
            if len(current_pool) > 0:
                out.append(current_pool.pop())
            else:
                refill = list(fallback_full_pool)
                rng.shuffle(refill)
                current_pool.extend(refill)
        return out

    def _fill_from_selected_domains(
        self,
        epoch_pool,
        selected_domains: List[int],
        remain: int,
        rng: random.Random,
    ) -> List[int]:
        out = []
        pairs = [(d, c) for d in selected_domains for c in self.unique_classes]
        rng.shuffle(pairs)

        ptr = 0
        while len(out) < remain:
            d, c = pairs[ptr % len(pairs)]
            picked = self._pop_from_pool(
                current_pool=epoch_pool[d][c],
                take=1,
                rng=rng,
                fallback_full_pool=self.pool[d][c],
            )
            out.extend(picked)
            ptr += 1

        return out[:remain]