#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch版数据集加载器
"""

import os
import scipy.io as sio
import torch
import numpy as np


class EpisodeDataset:
    """Episode数据集"""

    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        self.window_size = config['DATA']['window_size']
        self.overlap = config['DATA']['overlap']
        self.domain_map = self._create_domain_map()
        self.health_domains = {
            0: [0, 1, 2, 3],
            1: [4, 5, 6, 7],
            2: [8, 9, 10, 11]
        }

    def _create_domain_map(self):
        health = ['H', 'I', 'O']
        speed = ['A', 'B', 'C', 'D']
        domain_map = {}
        idx = 0
        for h in health:
            for s in speed:
                domain_map[idx] = {
                    'health': h,
                    'speed': s,
                    'files': [f"{h}-{s}-{t}.mat" for t in [1, 2, 3]]
                }
                idx += 1
        return domain_map

    def load_domain(self, domain_idx):
        domain_info = self.domain_map[domain_idx]
        all_vibration = []

        for file_name in domain_info['files']:
            file_path = os.path.join(self.data_path, file_name)
            if os.path.exists(file_path):
                try:
                    data = sio.loadmat(file_path)
                    vibration = data['Channel_1'].flatten()
                    all_vibration.append(vibration)
                except:
                    pass

        if not all_vibration:
            return torch.empty(0, self.window_size)

        full_signal = np.concatenate(all_vibration)
        samples = self._segment_signal(full_signal)

        label = 0 if domain_info['health'] == 'H' else \
            1 if domain_info['health'] == 'I' else 2

        return torch.FloatTensor(samples), torch.full((len(samples),), label, dtype=torch.long)

    def _segment_signal(self, signal):
        step = int(self.window_size * (1 - self.overlap))
        n_samples = (len(signal) - self.window_size) // step
        return [signal[i * step:i * step + self.window_size] for i in range(n_samples)]

    def generate_episode(self, source_domains, target_domain, k_shot, n_query):
        """生成episode"""
        source_data = {idx: self.load_domain(idx) for idx in source_domains}
        target_data, _ = self.load_domain(target_domain)

        support_x, support_y = [], []
        query_x, query_y = [], []

        for class_idx in range(3):
            class_domains = self.health_domains[class_idx]
            available = [d for d in class_domains if d in source_domains]

            # 支持集采样
            for _ in range(k_shot):
                domain = np.random.choice(available)
                x, y = source_data[domain]
                idx = np.random.randint(len(x))
                support_x.append(x[idx])
                support_y.append(class_idx)

            # 查询集采样
            for _ in range(n_query):
                domain = np.random.choice(available)
                x, y = source_data[domain]
                idx = np.random.randint(len(x))
                query_x.append(x[idx])
                query_y.append(class_idx)

        support_set = {
            'x': torch.stack(support_x),
            'y': torch.LongTensor(support_y)
        }
        query_set = {
            'x': torch.stack(query_x),
            'y': torch.LongTensor(query_y)
        }

        return support_set, query_set, target_data