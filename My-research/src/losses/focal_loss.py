# src/losses/focal_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    带类别权重的 Focal Loss
    完全兼容你现有的代码结构
    """
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        weight: torch.Tensor = None,  # 类别权重，由 build_classification_loss 传入
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight  # 类别权重
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. 计算带类别权重的交叉熵损失
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction='none'
        )

        # 2. 计算 Focal Loss 的调制因子
        pt = torch.exp(-ce_loss)  # 模型对正确类别的预测概率
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        # 3. 计算最终的 Focal Loss
        focal_loss = focal_weight * ce_loss

        # 4. 按 reduction 聚合
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss