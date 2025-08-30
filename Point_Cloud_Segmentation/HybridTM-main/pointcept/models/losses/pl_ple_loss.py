import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES


@LOSSES.register_module()
class PLPLELoss(nn.Module):
    def __init__(self,
                 ignore_index=-1,
                 pseudo_threshold=0.6,
                 curvature_threshold=0.1,
                 pseudo_weight=0.3,
                 physical_weight=0.2):
        super().__init__()
        self.ignore_index = ignore_index
        self.pseudo_threshold = pseudo_threshold  # 伪标签置信度阈值
        self.curvature_threshold = curvature_threshold  # 曲率阈值（低曲率判定为电力线）
        self.pseudo_weight = pseudo_weight  # 伪标签损失权重
        self.physical_weight = physical_weight  # 物理先验损失权重

    def forward(self, preds, targets, curvatures=None, **kwargs):
        # 如果没有曲率信息，返回0损失（不影响基础损失）
        if curvatures is None:
            return torch.tensor(0.0, device=preds.device)

        # 计算预测概率
        prob = F.softmax(preds, dim=1)
        power_line_prob = prob[:, 2]  # 电力线是类别2

        # 筛选有效区域（排除忽略索引）
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=preds.device)

        # 1. 物理先验损失（鼓励低曲率区域预测为电力线）
        # 仅在有效区域计算
        valid_curvatures = curvatures[valid_mask]
        valid_pl_prob = power_line_prob[valid_mask]

        # 物理先验标签：低曲率区域应为电力线
        physical_target = (valid_curvatures < self.curvature_threshold).float()
        physical_loss = F.binary_cross_entropy(
            valid_pl_prob,
            physical_target,
            reduction='mean'
        )

        # 2. 伪标签损失
        # 条件：中等置信度 + 低曲率 + 有效区域
        cond1 = (power_line_prob > self.pseudo_threshold) & (power_line_prob < 1 - self.pseudo_threshold)
        cond2 = curvatures < self.curvature_threshold
        pseudo_mask = cond1 & cond2 & valid_mask

        if pseudo_mask.any():
            # 创建伪标签（将符合条件的区域标记为电力线）
            pseudo_targets = targets.clone()
            pseudo_targets[pseudo_mask] = 2  # 电力线类别
            pseudo_loss = F.cross_entropy(
                preds,
                pseudo_targets,
                ignore_index=self.ignore_index,
                reduction='mean'
            )
        else:
            pseudo_loss = torch.tensor(0.0, device=preds.device)

        # 总补充损失（带权重）
        total_loss = self.pseudo_weight * pseudo_loss + self.physical_weight * physical_loss
        return total_loss
