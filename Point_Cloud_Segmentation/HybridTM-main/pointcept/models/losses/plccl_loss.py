import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES

import logging
import time

@LOSSES.register_module()
# 策略3: 电力线连续性约束对比学习（PLCCL）
class PLCCLoss(nn.Module):
    def __init__(self, temperature=0.1, gamma=0.5, loss_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma
        self.loss_weight = loss_weight
        self.requires_coords = True


    def forward(self, features, labels, coords):
        start_time = time.time()
        logger.debug(
            f"PLCCLoss forward start - features shape: {features.shape}, labels shape: {labels.shape}, coords shape: {coords.shape}")

        # 筛选电力线点
        line_mask = labels == 2
        line_count = line_mask.sum().item()
        logger.debug(f"PLCCLoss found {line_count} power line points")

        if line_count < 2:
            logger.debug(f"Not enough power line points ({line_count}), returning 0 loss")
            return torch.tensor(0.0, device=features.device) * self.loss_weight

        # 提取电力线特征和坐标
        extract_start = time.time()
        line_feats = features[line_mask]
        line_coords = coords[line_mask]
        logger.debug(
            f"PLCCLoss feature extraction took {time.time() - extract_start:.4f}s - line_feats shape: {line_feats.shape}")

        # 计算特征相似度
        sim_start = time.time()
        feat_sim = F.cosine_similarity(line_feats.unsqueeze(1), line_feats.unsqueeze(0), dim=2)
        logger.debug(f"Feature similarity calculation took {time.time() - sim_start:.4f}s - shape: {feat_sim.shape}")

        # 计算空间距离和掩码
        dist_start = time.time()
        coord_dist = torch.cdist(line_coords, line_coords)
        pos_mask = (coord_dist < 1.0) & (coord_dist > 1e-6)
        neg_mask = ~pos_mask
        logger.debug(
            f"Spatial distance calculation took {time.time() - dist_start:.4f}s - pos_mask count: {pos_mask.sum().item()}")

        # 计算InfoNCE损失
        nce_start = time.time()
        exp_sim = torch.exp(feat_sim / self.temperature)
        sum_exp = exp_sim * neg_mask.float()
        sum_exp = sum_exp.sum(dim=1, keepdim=True)
        pos_exp = exp_sim * pos_mask.float()
        pos_sum = pos_exp.sum(dim=1)
        info_nce_loss = -torch.log(pos_sum / (sum_exp.squeeze() + pos_sum + 1e-6)).mean()
        logger.debug(f"InfoNCE loss calculation took {time.time() - nce_start:.4f}s - value: {info_nce_loss.item()}")

        # 计算连续性约束损失
        cont_start = time.time()
        feat_dist = 1 - feat_sim
        continuity_loss = torch.mean(torch.abs(feat_dist - coord_dist) * pos_mask.float())
        logger.debug(
            f"Continuity loss calculation took {time.time() - cont_start:.4f}s - value: {continuity_loss.item()}")

        # 总损失
        total_loss = (info_nce_loss + self.gamma * continuity_loss) * self.loss_weight
        logger.debug(
            f"PLCCLoss forward complete - total time: {time.time() - start_time:.4f}s - total loss: {total_loss.item()}")

        return total_loss