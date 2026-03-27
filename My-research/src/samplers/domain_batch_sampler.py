# -*- coding: utf-8 -*-
"""Common batch sampler for source-only meta-learning."""
from __future__ import annotations

import math
import random
from collections import defaultdict, Counter
from typing import Dict, Iterator, List

from torch.utils.data import BatchSampler, Dataset


class MetaDomainBatchSampler(BatchSampler):
    """Build mini-batches with multiple source domains."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        domains_per_batch: int = 2,
        samples_per_domain: int | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
        debug: bool = False,
        debug_max_batches: int = 10,
        debug_print_indices: bool = False,
        class_aware: bool = False,
        normal_label: int = 0,
        min_per_fault_class: int = 1,
        oversample_minority: bool = True,

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

        # debug options
        self.debug = bool(debug)
        self.debug_max_batches = int(debug_max_batches)
        self.debug_print_indices = bool(debug_print_indices)



        # 获取有几个域、每个域多少样本、每个类被多少个
        domains = list(map(int, dataset.get_all_domains().tolist()))
        self.sample_domains: List[int] = domains
        if not hasattr(dataset, 'get_all_labels'):
            raise ValueError('MetaDomainBatchSampler requires dataset.get_all_labels() for class-level debug.')
        labels = list(map(int, dataset.get_all_labels().tolist()))
        self.sample_labels: List[int] = labels

        self.domain_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, d in enumerate(domains):
            self.domain_to_indices[d].append(idx)

        self.domain_ids = sorted(self.domain_to_indices.keys())

        self.class_aware = bool(class_aware)
        self.normal_label = int(normal_label)
        self.min_per_fault_class = int(min_per_fault_class)
        self.oversample_minority = bool(oversample_minority)
        if not hasattr(dataset, 'get_all_labels'):
            raise ValueError('MetaDomainBatchSampler requires dataset.get_all_labels().')
        labels = list(map(int, dataset.get_all_labels().tolist()))
        self.sample_labels = labels
        self.domain_to_class_to_indices = defaultdict(lambda: defaultdict(list))
        for idx, (d, y) in enumerate(zip(self.sample_domains, self.sample_labels)):
            self.domain_to_class_to_indices[d][y].append(idx)

        if len(self.domain_ids) < 2:
            raise ValueError('MetaDomainBatchSampler needs at least 2 source domains in train dataset.')
        if self.domains_per_batch < 2:
            raise ValueError('domains_per_batch should be >= 2 for source-only meta-learning.')
        if self.domains_per_batch > len(self.domain_ids):
            self.domains_per_batch = len(self.domain_ids)

        if self.samples_per_domain is None:
            self.samples_per_domain = max(1, self.batch_size // self.domains_per_batch)

        self.effective_batch_size = self.samples_per_domain * self.domains_per_batch

        if self.debug:
            print(
                f"[Sampler Init] total_samples={len(self.dataset)} "
                f"domain_ids={self.domain_ids} "
                f"domains_per_batch={self.domains_per_batch} "
                f"samples_per_domain={self.samples_per_domain} "
                f"effective_batch_size={self.effective_batch_size}"
            )
            for d in self.domain_ids:
                print(f"[Sampler Init] domain={d} num_samples={len(self.domain_to_indices[d])}")

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

    def _debug_print_batch(
            self,
            batch_id: int,
            chosen_domains: List[int],
            batch: List[int],
            taken_per_domain: Dict[int, List[int]],
    ) -> None:
        if not self.debug:
            return
        if batch_id >= self.debug_max_batches:
            return

        batch_domain_list = [self.sample_domains[idx] for idx in batch]
        batch_domain_counter = Counter(batch_domain_list)

        print(f"\n[Sampler][Epoch {self.epoch}][Batch {batch_id}]")
        print(f"  chosen_domains={chosen_domains}")
        print(f"  batch_size={len(batch)}")
        print(f"  domain_count_in_batch={dict(sorted(batch_domain_counter.items()))}")

        for d in chosen_domains:
            indices_d = taken_per_domain[d]
            labels_d = [self.sample_labels[idx] for idx in indices_d]
            class_counter = Counter(labels_d)

            print(f"  domain={d} -> picked {len(indices_d)} samples")
            print(f"    class_count={dict(sorted(class_counter.items()))}")

            if self.debug_print_indices:
                class_to_indices = defaultdict(list)
                for idx in indices_d:
                    y = self.sample_labels[idx]
                    class_to_indices[y].append(idx)

                for cls_id in sorted(class_to_indices.keys()):
                    print(
                        f"    class={cls_id} -> "
                        f"num={len(class_to_indices[cls_id])}, "
                        f"indices={class_to_indices[cls_id]}"
                    )

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        pools = self._build_pools(rng)
        domain_cycle = list(self.domain_ids)

        if self.shuffle:
            rng.shuffle(domain_cycle)

        remaining = sum(len(v) for v in pools.values())
        batch_id = 0

        while remaining > 0:
            # choose distinct domains for current batch
            if self.shuffle:
                chosen_domains = rng.sample(self.domain_ids, k=self.domains_per_batch)
            else:
                chosen_domains = domain_cycle[:self.domains_per_batch]
                domain_cycle = domain_cycle[self.domains_per_batch:] + chosen_domains

            batch: List[int] = []
            taken_per_domain: Dict[int, List[int]] = {}

            for d in chosen_domains:
                take = self._sample_indices_for_domain(d, rng)
                taken_per_domain[d] = list(take)
                batch.extend(take)

            if self.shuffle:
                rng.shuffle(batch)

            self._debug_print_batch(
                batch_id=batch_id,
                chosen_domains=chosen_domains,
                batch=batch,
                taken_per_domain=taken_per_domain,
            )

            if len(batch) < self.effective_batch_size and self.drop_last:
                break

            if len(batch) > 0:
                yield batch

            remaining -= min(self.effective_batch_size, remaining)
            batch_id += 1

    def __len__(self) -> int:
        n = len(self.dataset)
        bs = max(1, self.effective_batch_size)
        if self.drop_last:
            return n // bs
        return math.ceil(n / bs)

    def _sample_with_replacement(self, pool, k, rng):
        if len(pool) == 0:
            return []
        if len(pool) >= k:
            if self.shuffle:
                return rng.sample(pool, k)
            return list(pool[:k])

        out = []
        while len(out) < k:
            need = k - len(out)
            if self.shuffle:
                out.extend(rng.choices(pool, k=need))
            else:
                out.extend(pool[:need])
        return out[:k]

    def _sample_indices_for_domain(self, domain_id, rng):
        # 默认：原始随机采样
        if not self.class_aware:
            need = self.samples_per_domain
            take = []
            pool = list(self.domain_to_indices[domain_id])

            while need > 0:
                if len(pool) == 0:
                    pool = list(self.domain_to_indices[domain_id])
                    if self.shuffle:
                        rng.shuffle(pool)

                n_take = min(need, len(pool))
                take.extend(pool[:n_take])
                pool = pool[n_take:]
                need -= n_take

            return take

        # 类别感知采样
        class_to_indices = self.domain_to_class_to_indices[domain_id]
        fault_classes = sorted([c for c in class_to_indices.keys() if c != self.normal_label])

        take = []

        # 先保证每个故障类至少来 min_per_fault_class 个
        for cls in fault_classes:
            k = self.min_per_fault_class
            if k <= 0:
                continue
            take.extend(self._sample_with_replacement(class_to_indices[cls], k, rng))

        # 如果已经超过配额，直接截断
        if len(take) >= self.samples_per_domain:
            if self.shuffle:
                rng.shuffle(take)
            return take[:self.samples_per_domain]

        # 剩余位置优先补正常类
        remain = self.samples_per_domain - len(take)
        normal_pool = class_to_indices.get(self.normal_label, [])

        if len(normal_pool) > 0:
            take.extend(self._sample_with_replacement(normal_pool, remain, rng))
        else:
            # 没有正常类，就从所有类补
            all_pool = []
            for idxs in class_to_indices.values():
                all_pool.extend(idxs)
            take.extend(self._sample_with_replacement(all_pool, remain, rng))

        if self.shuffle:
            rng.shuffle(take)

        return take[:self.samples_per_domain]