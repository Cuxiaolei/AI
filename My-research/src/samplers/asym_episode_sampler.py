# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Iterator, List

from torch.utils.data import BatchSampler, Dataset


class AsymEpisodeBatchSampler(BatchSampler):
    """
    非对称长尾元学习采样器：
    - 先选 1 个 query 域
    - 再选若干 support 域
    - support 域：轻量增强故障类
    - query 域：接近自然长尾分布，仅保底极少故障样本
    """
    def __init__(
        self,
        dataset: Dataset,
        support_domains: int = 2,
        support_samples_per_domain: int = 12,
        query_samples_per_domain: int = 8,
        support_fault_ratio: float = 0.33,
        query_min_fault: int = 1,
        query_temp: float = 0.9,
        normal_label: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
        debug: bool = False,
        debug_max_batches: int = 10,
    ) -> None:
        if not hasattr(dataset, "get_all_domains"):
            raise ValueError("AsymEpisodeBatchSampler requires dataset.get_all_domains().")
        if not hasattr(dataset, "get_all_labels"):
            raise ValueError("AsymEpisodeBatchSampler requires dataset.get_all_labels().")

        self.dataset = dataset
        self.support_domains = int(support_domains)
        self.support_samples_per_domain = int(support_samples_per_domain)
        self.query_samples_per_domain = int(query_samples_per_domain)
        self.support_fault_ratio = float(support_fault_ratio)
        self.query_min_fault = int(query_min_fault)
        self.query_temp = float(query_temp)
        self.normal_label = int(normal_label)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.debug = bool(debug)
        self.debug_max_batches = int(debug_max_batches)
        self.epoch = 0

        # 为了后续按“样本数最少的域”切出 query，要求 query 样本数更少
        if self.query_samples_per_domain >= self.support_samples_per_domain:
            raise ValueError("query_samples_per_domain must be < support_samples_per_domain.")

        domains = list(map(int, dataset.get_all_domains().tolist()))
        labels = list(map(int, dataset.get_all_labels().tolist()))

        self.domain_to_indices: Dict[int, List[int]] = defaultdict(list)
        self.domain_to_class_to_indices: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))

        for idx, (d, y) in enumerate(zip(domains, labels)):
            self.domain_to_indices[d].append(idx)
            self.domain_to_class_to_indices[d][y].append(idx)

        self.domain_ids = sorted(self.domain_to_indices.keys())
        if len(self.domain_ids) < 2:
            raise ValueError("Need at least 2 source domains.")
        self.support_domains = min(self.support_domains, len(self.domain_ids) - 1)

        self.episode_size = self.support_domains * self.support_samples_per_domain + self.query_samples_per_domain

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _pick(self, pool: List[int], k: int, rng: random.Random) -> List[int]:
        """允许重复采样，避免少数类样本过少直接失败"""
        if k <= 0 or len(pool) == 0:
            return []
        if len(pool) >= k:
            return rng.sample(pool, k) if self.shuffle else pool[:k]
        out = []
        while len(out) < k:
            need = k - len(out)
            out.extend(rng.choices(pool, k=need))
        return out[:k]

    def _sample_support(self, d: int, rng: random.Random) -> List[int]:
        """
        support 域：
        先抽一部分故障类，再从全体补齐。
        这样故障类能稳定进入元训练任务。
        """
        class_to_idx = self.domain_to_class_to_indices[d]
        fault_pool = []
        for c, idxs in class_to_idx.items():
            if c != self.normal_label:
                fault_pool.extend(idxs)

        n_fault = min(int(round(self.support_samples_per_domain * self.support_fault_ratio)),
                      self.support_samples_per_domain)
        n_rest = self.support_samples_per_domain - n_fault

        out = []
        out.extend(self._pick(fault_pool, n_fault, rng))
        out.extend(self._pick(self.domain_to_indices[d], n_rest, rng))
        rng.shuffle(out)
        return out[:self.support_samples_per_domain]

    def _sample_query(self, d: int, rng: random.Random) -> List[int]:
        """
        query 域：
        保留接近自然长尾分布，只做极轻故障保底。
        temp 越接近 1，越接近自然分布；越小越抬高少数类概率。
        """
        class_to_idx = self.domain_to_class_to_indices[d]
        classes = sorted(class_to_idx.keys())

        # 先保底少量故障样本
        fault_pool = []
        for c, idxs in class_to_idx.items():
            if c != self.normal_label:
                fault_pool.extend(idxs)

        out = []
        out.extend(self._pick(fault_pool, self.query_min_fault, rng))
        remain = self.query_samples_per_domain - len(out)
        if remain <= 0:
            rng.shuffle(out)
            return out[:self.query_samples_per_domain]

        # 按 n_c ^ temp 的类概率抽样，接近自然长尾
        counts = {c: max(len(class_to_idx[c]), 1) for c in classes}
        weights = [counts[c] ** self.query_temp for c in classes]
        s = sum(weights)
        probs = [w / s for w in weights]

        for _ in range(remain):
            c = rng.choices(classes, weights=probs, k=1)[0]
            out.extend(self._pick(class_to_idx[c], 1, rng))

        rng.shuffle(out)
        return out[:self.query_samples_per_domain]

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        n_batches = len(self)

        for batch_id in range(n_batches):
            query_domain = rng.choice(self.domain_ids)
            support_candidates = [d for d in self.domain_ids if d != query_domain]
            support_domains = rng.sample(support_candidates, k=self.support_domains)

            batch = []
            for d in support_domains:
                batch.extend(self._sample_support(d, rng))
            batch.extend(self._sample_query(query_domain, rng))

            if self.shuffle:
                rng.shuffle(batch)

            if self.debug and batch_id < self.debug_max_batches:
                print(
                    f"[AsymEpisode][Epoch {self.epoch}][Batch {batch_id}] "
                    f"support_domains={support_domains} query_domain={query_domain} "
                    f"support_n={self.support_samples_per_domain} query_n={self.query_samples_per_domain}"
                )

            yield batch

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // max(1, self.episode_size)
        return math.ceil(n / max(1, self.episode_size))