import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES


@LOSSES.register_module()
class PLCCLoss(nn.Module):
    """Power Line Continuity Constrained Contrastive Loss"""

    def __init__(self, temperature=0.1, gamma=0.5, ignore_index=-1, loss_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight  # 新增：损失权重参数

    def forward(self, features, labels, coords):
        # 电力线在类别顺序中是第2类（索引为2）
        line_mask = (labels != self.ignore_index) & (labels == 2)
        if line_mask.sum() < 2:
            return torch.tensor(0.0, device=features.device)

        line_feats = features[line_mask]
        line_coords = coords[line_mask]

        # 计算特征相似度
        feat_sim = F.cosine_similarity(line_feats.unsqueeze(1), line_feats.unsqueeze(0), dim=2)

        # 计算3D空间距离
        coord_dist = torch.cdist(line_coords, line_coords)

        # 构建正样本对（空间距离近的点）
        pos_mask = coord_dist < 1.0  # 距离小于1m的视为连续点
        pos_mask = pos_mask & (coord_dist > 1e-6)  # 排除自身

        # 构建负样本对
        neg_mask = ~pos_mask

        # 计算InfoNCE损失
        exp_sim = torch.exp(feat_sim / self.temperature)
        sum_exp = exp_sim * neg_mask.float()
        sum_exp = sum_exp.sum(dim=1, keepdim=True)

        pos_exp = exp_sim * pos_mask.float()
        pos_sum = pos_exp.sum(dim=1)

        info_nce_loss = -torch.log(pos_sum / (sum_exp.squeeze() + pos_sum + 1e-6)).mean()

        # 计算连续性约束损失
        feat_dist = 1 - feat_sim
        continuity_loss = torch.mean(torch.abs(feat_dist - coord_dist) * pos_mask.float())

        # 总损失，并应用权重
        total_loss = (info_nce_loss + self.gamma * continuity_loss) * self.loss_weight

        return total_loss
