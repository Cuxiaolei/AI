import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class PrototypicalNetwork(nn.Module):
    """原型网络 - 基于原型对比学习"""

    def __init__(self, backbone, feature_dim: int = 512, num_classes: int = 3):
        super(PrototypicalNetwork, self).__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # 类别原型（可作为可学习参数）
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, feature_dim)
        )

        # 初始化原型
        nn.init.kaiming_normal_(self.prototypes, mode='fan_in')

    def forward(self, x, return_features=False):
        """前向传播"""
        logits, features = self.backbone(x, return_features=True)

        if return_features:
            return logits, features
        return logits

    def compute_prototypes(self, support_data: torch.Tensor,
                           support_labels: torch.Tensor) -> torch.Tensor:
        """
        从支持集计算类别原型

        Args:
            support_data: 支持集数据 [N, C, L]
            support_labels: 支持集标签 [N]

        Returns:
            prototypes: 类别原型 [num_classes, feature_dim]
        """
        batch_size = support_data.shape[0]

        # 提取特征
        _, support_features = self.backbone(support_data, return_features=True)
        # support_features: [N, feature_dim]

        # 重置原型
        new_prototypes = torch.zeros_like(self.prototypes)

        # 计算每个类的均值
        for class_id in range(self.num_classes):
            mask = (support_labels == class_id)
            if mask.sum() > 0:
                class_features = support_features[mask]
                new_prototypes[class_id] = class_features.mean(dim=0)
            else:
                # 如果没有该类样本，保留原原型
                new_prototypes[class_id] = self.prototypes[class_id]

        # 更新原型
        self.prototypes.data = new_prototypes

        return new_prototypes

    def prototypical_loss(self, query_data: torch.Tensor,
                          query_labels: torch.Tensor,
                          temperature: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        原型对比损失

        Args:
            query_data: 查询集数据 [N, C, L]
            query_labels: 查询集标签 [N]
            temperature: 温度参数

        Returns:
            loss: 原型损失
            accuracy: 准确率
        """
        # 提取特征
        _, query_features = self.backbone(query_data, return_features=True)
        # query_features: [N, feature_dim]

        # 计算余弦相似度
        query_features = F.normalize(query_features, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)

        # 相似度矩阵
        logits = torch.mm(query_features, prototypes.t()) / temperature
        # logits: [N, num_classes]

        # 交叉熵损失
        loss = F.cross_entropy(logits, query_labels)

        # 准确率
        pred = torch.argmax(logits, dim=1)
        accuracy = (pred == query_labels).float().mean()

        return loss, accuracy

    def domain_contrastive_loss(self, features: torch.Tensor,
                                domain_labels: torch.Tensor,
                                temperature: float = 0.1) -> torch.Tensor:
        """
        域对比损失：对齐不同域的同类特征

        Args:
            features: 特征 [N, feature_dim]
            domain_labels: 域标签 [N]
            temperature: 温度参数

        Returns:
            loss: 域对比损失
        """
        features = F.normalize(features, dim=1)

        # 计算相似度矩阵
        sim_matrix = torch.mm(features, features.t()) / temperature
        # sim_matrix: [N, N]

        # 构建正样本掩码（同类不同域）
        batch_size = features.shape[0]
        mask = torch.zeros(batch_size, batch_size, device=features.device)

        for i in range(batch_size):
            for j in range(batch_size):
                # 同类但不同域
                if i != j and domain_labels[i] == domain_labels[j]:
                    mask[i, j] = 1

        # InfoNCE损失
        exp_sim = torch.exp(sim_matrix) * (1 - torch.eye(batch_size, device=features.device))

        loss = 0
        for i in range(batch_size):
            pos_sum = (exp_sim[i] * mask[i]).sum()
            all_sum = exp_sim[i].sum()
            loss -= torch.log(pos_sum / all_sum + 1e-8)

        return loss / batch_size

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测"""
        _, features = self.backbone(x, return_features=True)

        features = F.normalize(features, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)

        logits = torch.mm(features, prototypes.t())
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)

        return pred, probs