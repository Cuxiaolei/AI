import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PrototypicalLoss(nn.Module):
    """原型损失封装"""

    def __init__(self, temperature: float = 0.1,
                 proto_weight: float = 1.0,
                 domain_weight: float = 0.5):
        super(PrototypicalLoss, self).__init__()
        self.temperature = temperature
        self.proto_weight = proto_weight
        self.domain_weight = domain_weight

    def forward(self, model, support_data, support_labels,
                query_data, query_labels, domain_labels=None) -> torch.Tensor:
        """
        前向传播

        Args:
            model: 原型网络模型
            support_data: 支持集数据
            support_labels: 支持集标签
            query_data: 查询集数据
            query_labels: 查询集标签
            domain_labels: 域标签（用于域对比损失）

        Returns:
            total_loss: 总损失
            metrics: 指标字典
        """
        # 计算原型
        model.compute_prototypes(support_data, support_labels)

        # 原型损失
        proto_loss, accuracy = model.prototypical_loss(
            query_data, query_labels, self.temperature
        )

        total_loss = self.proto_weight * proto_loss

        metrics = {
            'proto_loss': proto_loss.item(),
            'accuracy': accuracy.item(),
            'total_loss': total_loss.item()
        }

        # 域对比损失
        if domain_labels is not None and self.domain_weight > 0:
            # 提取查询集特征
            _, query_features = model.backbone(query_data, return_features=True)

            domain_loss = model.domain_contrastive_loss(
                query_features, domain_labels, self.temperature
            )

            total_loss += self.domain_weight * domain_loss
            metrics['domain_loss'] = domain_loss.item()
            metrics['total_loss'] = total_loss.item()

        return total_loss, metrics


class SupConLoss(nn.Module):
    """监督对比损失（备选方案）"""

    def __init__(self, temperature: float = 0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 归一化的特征 [N, feature_dim]
            labels: 标签 [N]

        Returns:
            loss: 对比损失
        """
        batch_size = features.shape[0]

        # 计算相似度矩阵
        sim_matrix = torch.mm(features, features.t()) / self.temperature
        # sim_matrix: [N, N]

        # 构建掩码
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()
        mask = mask * (1 - torch.eye(batch_size, device=features.device))

        # 计算损失
        exp_sim = torch.exp(sim_matrix) * (1 - torch.eye(batch_size, device=features.device))

        loss = 0
        for i in range(batch_size):
            pos_sum = (exp_sim[i] * mask[i]).sum()
            all_sum = exp_sim[i].sum()
            loss -= torch.log(pos_sum / all_sum + 1e-8)

        return loss / batch_size