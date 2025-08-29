import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES

import logging
import time

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@LOSSES.register_module()
class PLCCLoss(nn.Module):
    need_coord = True  # 新增：标记该损失需要 coord
    def __init__(self, temperature=0.1, gamma=0.5, loss_weight=1.0, ignore_index=-1):
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma
        self.loss_weight = loss_weight
        self.requires_coords = True
        self.ignore_index = ignore_index

    # 修改参数接收方式，适配模型传递的参数顺序
    def forward(self, features, labels, coords=None):  # 直接接收三个参数
        start_time = time.time()

        # 校验必要数据是否存在
        if labels is None or (self.requires_coords and coords is None):
            logger.warning("Missing labels or coords, returning 0 loss")
            return torch.tensor(0.0, device=features.device) * self.loss_weight

        logger.debug(
            f"PLCCLoss forward start - features shape: {features.shape}, "
            f"labels shape: {labels.shape}, coords shape: {coords.shape}")

        # 过滤忽略标签
        if self.ignore_index != -1:
            mask = labels != self.ignore_index
            features = features[mask]
            labels = labels[mask]
            coords = coords[mask]
            logger.debug(f"Applied ignore_index filter - remaining points: {features.shape[0]}")

        # 筛选电力线点（假设标签2代表电力线）
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
            f"Feature extraction took {time.time() - extract_start:.4f}s - "
            f"line_feats shape: {line_feats.shape}")

        # 计算特征相似度（余弦相似度）
        sim_start = time.time()
        feat_sim = F.cosine_similarity(line_feats.unsqueeze(1), line_feats.unsqueeze(0), dim=2)
        logger.debug(f"Feature similarity took {time.time() - sim_start:.4f}s - shape: {feat_sim.shape}")

        # 计算空间距离和掩码（正样本：距离<1.0且非自身）
        dist_start = time.time()
        coord_dist = torch.cdist(line_coords, line_coords)
        pos_mask = (coord_dist < 1.0) & (coord_dist > 1e-6)  # 排除自身点
        neg_mask = ~pos_mask
        logger.debug(
            f"Spatial distance took {time.time() - dist_start:.4f}s - "
            f"positive pairs: {pos_mask.sum().item()}")

        # 计算InfoNCE损失
        nce_start = time.time()
        exp_sim = torch.exp(feat_sim / self.temperature)
        sum_exp = (exp_sim * neg_mask.float()).sum(dim=1, keepdim=True).squeeze()
        pos_sum = (exp_sim * pos_mask.float()).sum(dim=1)
        info_nce_loss = -torch.log(pos_sum / (sum_exp + pos_sum + 1e-6)).mean()
        logger.debug(f"InfoNCE loss: {info_nce_loss.item()} (took {time.time() - nce_start:.4f}s)")

        # 计算连续性约束损失（特征距离与空间距离的一致性）
        cont_start = time.time()
        feat_dist = 1 - feat_sim  # 余弦距离（1-相似度）
        continuity_loss = torch.mean(torch.abs(feat_dist - coord_dist) * pos_mask.float())
        logger.debug(f"Continuity loss: {continuity_loss.item()} (took {time.time() - cont_start:.4f}s)")

        # 总损失（加权求和）
        total_loss = (info_nce_loss + self.gamma * continuity_loss) * self.loss_weight
        logger.debug(
            f"PLCCLoss complete - total time: {time.time() - start_time:.4f}s - "
            f"total loss: {total_loss.item()}")

        return total_loss
