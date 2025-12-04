#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
深度学习评估指标（含持续学习）
"""

import torch
import torch.nn.functional as F
import numpy as np


class DeepMetrics:
    @staticmethod
    def accuracy(logits, labels):
        """计算准确率"""
        preds = logits.argmax(dim=1)
        return (preds == labels).float().mean().item()

    @staticmethod
    def prototype_loss(query_features, query_labels, prototypes):
        """原型对比损失"""
        features = F.normalize(query_features, dim=1)
        prototypes = F.normalize(prototypes, dim=1)
        logits = torch.mm(features, prototypes.t())
        return F.cross_entropy(logits, query_labels)

    @staticmethod
    def coral_distance(cov_s, cov_t):
        """CORAL距离"""
        return torch.norm(cov_s - cov_t, p='fro').item()


class ContinualMetrics:
    """持续学习评估指标"""

    @staticmethod
    def backward_transfer(results):
        """后向迁移能力：学习新域后，在旧域上的性能变化"""
        if len(results) < 2:
            return 0.0

        bwt = 0.0
        for i in range(len(results) - 1):
            bwt += results[-1][f'domain_{i}'] - results[i][f'domain_{i}']

        return bwt / (len(results) - 1)

    @staticmethod
    def forward_transfer(results):
        """前向迁移能力：在未见域上的初始化性能"""
        if len(results) < 2:
            return 0.0

        # 计算平均性能提升
        ft = 0.0
        for i in range(1, len(results)):
            ft += results[i][f'domain_{i}'] - results[0][f'domain_{i}']

        return ft / (len(results) - 1)

    @staticmethod
    def average_accuracy(results):
        """平均准确率"""
        accs = []
        for r in results:
            domain_accs = [v for k, v in r.items() if k.startswith('domain_')]
            accs.append(np.mean(domain_accs))
        return np.mean(accs)

    @staticmethod
    def forgetting_measure(results):
        """遗忘度量：学习新域后，旧域性能下降程度"""
        if len(results) < 2:
            return 0.0

        forget = 0.0
        n_old_domains = 0

        for i in range(len(results) - 1):
            for j in range(i):
                diff = results[i][f'domain_{j}'] - results[-1][f'domain_{j}']
                if diff > 0:
                    forget += diff
                n_old_domains += 1

        return forget / n_old_domains if n_old_domains > 0 else 0.0