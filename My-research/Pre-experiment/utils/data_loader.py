#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch版数据集加载器（域泛化版）
"""

import os
import scipy.io as sio
import torch
import numpy as np
from torch.utils.data import Dataset


class EpisodeDataset:
    """适用于小样本域泛化的Episode数据集"""

    def __init__(self, data_path, config):
        self.data_path = data_path
        self.window_size = config['DATA']['window_size']
        self.overlap = config['DATA']['overlap']
        self.domain_map = self._create_domain_map()

        # 类别到域的映射：每个健康状态对应4个速度条件
        self.health_domains = {
            0: [0, 1, 2, 3],  # H-A, H-B, H-C, H-D
            1: [4, 5, 6, 7],  # I-A, I-B, I-C, I-D
            2: [8, 9, 10, 11]  # O-A, O-B, O-C, O-D
        }

    def _create_domain_map(self):
        """创建域映射"""
        health_states = ['H', 'I', 'O']
        speed_conditions = ['A', 'B', 'C', 'D']
        domain_map = {}

        idx = 0
        for health in health_states:
            for speed in speed_conditions:
                domain_map[idx] = {
                    'health': health,
                    'speed': speed,
                    'files': [f"{health}-{speed}-{trial}.mat" for trial in [1, 2, 3]]
                }
                idx += 1
        return domain_map

    def load_domain(self, domain_idx):
        """加载单个域的所有数据
        Returns:
            samples: 样本数据 [N, window_size]
            labels: 标签 [N]
        """
        domain_info = self.domain_map[domain_idx]
        all_signals = []

        for file_name in domain_info['files']:
            file_path = os.path.join(self.data_path, file_name)
            if os.path.exists(file_path):
                try:
                    data = sio.loadmat(file_path)
                    vibration = data['Channel_1'].flatten()
                    all_signals.append(vibration)
                except Exception as e:
                    print(f"警告: 加载{file_path}失败: {e}")

        if not all_signals:
            return torch.empty(0, self.window_size), torch.empty(0, dtype=torch.long)

        # 拼接所有trial的信号
        full_signal = np.concatenate(all_signals)

        # 滑窗分割
        samples = self._segment_signal(full_signal)

        # 确定标签：H->0, I->1, O->2
        health = domain_info['health']
        label = 0 if health == 'H' else 1 if health == 'I' else 2
        labels = torch.full((len(samples),), label, dtype=torch.long)

        return torch.FloatTensor(samples), labels

    def _segment_signal(self, signal):
        """滑窗分割信号
        Returns:
            samples: 样本列表
        """
        step = int(self.window_size * (1 - self.overlap))
        n_samples = max(1, (len(signal) - self.window_size) // step)
        return [signal[i * step:i * step + self.window_size] for i in range(n_samples)]

    def sample_from_domains(self, domain_indices, n_samples_per_class, per_class=True):
        """从指定域采样
        Args:
            domain_indices: 域索引列表
            n_samples_per_class: 每类采样数
            per_class: 是否按类别均衡采样
        Returns:
            samples: 采样的数据
            labels: 采样标签
        """
        all_samples, all_labels = [], []

        if per_class:
            # 每类从对应域采样
            for class_idx in range(3):
                class_domains = self.health_domains[class_idx]
                available_domains = [d for d in class_domains if d in domain_indices]

                if not available_domains:
                    continue

                for _ in range(n_samples_per_class):
                    domain = np.random.choice(available_domains)
                    domain_samples, domain_labels = self.load_domain(domain)

                    if len(domain_samples) > 0:
                        idx = np.random.randint(len(domain_samples))
                        all_samples.append(domain_samples[idx])
                        all_labels.append(class_idx)
        else:
            # 随机采样
            for _ in range(n_samples_per_class):
                domain = np.random.choice(domain_indices)
                domain_samples, domain_labels = self.load_domain(domain)

                if len(domain_samples) > 0:
                    idx = np.random.randint(len(domain_samples))
                    all_samples.append(domain_samples[idx])
                    all_labels.append(domain_labels[idx].item())

        if not all_samples:
            return torch.empty(0, self.window_size), torch.empty(0, dtype=torch.long)

        return torch.stack(all_samples), torch.LongTensor(all_labels)

    def generate_fsdg_episode(self, source_domains, target_domain, k_shot, n_query):
        """生成小样本域泛化Episode
        支持集：从源域按类采样k_shot
        查询集：从目标域采样n_query
        """
        # 支持集：从源域按类别采样
        support_x, support_y = self.sample_from_domains(
            source_domains, k_shot, per_class=True
        )

        # 查询集：从目标域采样（这才是域泛化的关键）
        target_x, target_y = self.load_domain(target_domain)

        if len(target_x) < n_query * 3:  # 3个类别
            # 如果目标域样本不足，从源域补充（用于冷启动）
            query_x, query_y = self.sample_from_domains(
                source_domains, n_query * 3, per_class=True
            )
        else:
            # 标准域泛化：查询集必须来自目标域
            query_indices = np.random.choice(
                len(target_x), min(n_query * 3, len(target_x)), replace=False
            )
            query_x = target_x[query_indices]
            query_y = target_y[query_indices]

        # 目标域无标签数据（用于域对齐）
        unlabeled_indices = np.random.choice(
            len(target_x), min(100, len(target_x)), replace=False
        )
        target_unlabeled = target_x[unlabeled_indices]

        support_set = {'x': support_x, 'y': support_y}
        query_set = {'x': query_x, 'y': query_y}

        return support_set, query_set, target_unlabeled