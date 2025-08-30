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
    need_coord = True  # 标记该损失需要 coord

    def __init__(self,
                 temperature=0.1,
                 gamma=0.5,
                 loss_weight=1.0,
                 ignore_index=-1,
                 max_batch_size=5000,  # 新增：每批次处理的最大点数
                 pos_dist_thresh=1.0,  # 正样本距离阈值
                 neg_sample_ratio=2.0  # 负样本采样比例
                 ):
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma
        self.loss_weight = loss_weight
        self.requires_coords = True
        self.ignore_index = ignore_index
        self.max_batch_size = max_batch_size  # 控制单批次内存占用
        self.pos_dist_thresh = pos_dist_thresh
        self.neg_sample_ratio = neg_sample_ratio

    def forward(self, features, labels, coords=None):
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
        M = line_feats.shape[0]  # 电力线点总数
        logger.debug(
            f"Feature extraction took {time.time() - extract_start:.4f}s - "
            f"line_feats shape: {line_feats.shape}")

        # 计算总批次数量
        num_batches = (M + self.max_batch_size - 1) // self.max_batch_size
        logger.debug(f"Split into {num_batches} batches (max size: {self.max_batch_size})")

        total_info_nce = 0.0
        total_continuity = 0.0
        total_pairs = 0

        # 分批处理以避免内存溢出
        for batch_idx in range(num_batches):
            # 计算当前批次的索引范围
            start = batch_idx * self.max_batch_size
            end = min((batch_idx + 1) * self.max_batch_size, M)
            batch_size = end - start

            # 当前批次的特征和坐标
            batch_feats = line_feats[start:end]  # [B, C]
            batch_coords = line_coords[start:end]  # [B, 3]

            # 计算当前批次与所有电力线点的特征相似度 [B, M]
            sim_start = time.time()
            feat_sim = F.cosine_similarity(
                batch_feats.unsqueeze(1),  # [B, 1, C]
                line_feats.unsqueeze(0),  # [1, M, C]
                dim=2  # 结果形状 [B, M]
            )
            logger.debug(
                f"Batch {batch_idx} similarity took {time.time() - sim_start:.4f}s - "
                f"shape: {feat_sim.shape}")

            # 计算当前批次与所有点的空间距离 [B, M]
            dist_start = time.time()
            coord_dist = torch.cdist(batch_coords, line_coords)
            # 正样本掩码：距离小于阈值且非自身点
            pos_mask = (coord_dist < self.pos_dist_thresh) & (coord_dist > 1e-6)
            pos_count = pos_mask.sum().item()

            if pos_count == 0:
                logger.debug(f"Batch {batch_idx} has no positive pairs, skipping")
                continue

            total_pairs += pos_count
            logger.debug(
                f"Batch {batch_idx} distance took {time.time() - dist_start:.4f}s - "
                f"positive pairs: {pos_count}")

            # 负样本采样（避免全量计算）
            neg_mask = ~pos_mask
            # 每个点最多采样 neg_ratio 倍于正样本数量的负样本
            max_neg_per_point = int(self.neg_sample_ratio * pos_mask.sum(dim=1).max().item())

            # 计算InfoNCE损失（当前批次）
            nce_start = time.time()
            exp_sim = torch.exp(feat_sim / self.temperature)

            # 负样本求和（带采样）
            if max_neg_per_point > 0:
                # 对负样本进行排序并取topK（最相似的负样本）
                neg_values, neg_indices = torch.topk(
                    exp_sim * neg_mask.float(),
                    k=min(max_neg_per_point, M),
                    dim=1
                )
                sum_exp = neg_values.sum(dim=1)
            else:
                sum_exp = (exp_sim * neg_mask.float()).sum(dim=1)

            # 正样本求和
            pos_sum = (exp_sim * pos_mask.float()).sum(dim=1)

            # 计算当前批次的InfoNCE损失
            batch_info_nce = -torch.log(pos_sum / (sum_exp + pos_sum + 1e-6)).mean()
            logger.debug(
                f"Batch {batch_idx} InfoNCE: {batch_info_nce.item()} "
                f"(took {time.time() - nce_start:.4f}s)")

            # 计算连续性约束损失（当前批次）
            cont_start = time.time()
            feat_dist = 1 - feat_sim  # 余弦距离（1-相似度）
            batch_continuity = torch.mean(torch.abs(feat_dist - coord_dist) * pos_mask.float())
            logger.debug(
                f"Batch {batch_idx} continuity: {batch_continuity.item()} "
                f"(took {time.time() - cont_start:.4f}s)")

            # 累加损失（按批次大小加权）
            total_info_nce += batch_info_nce * batch_size
            total_continuity += batch_continuity * batch_size

        # 计算平均损失
        if total_pairs == 0:
            logger.debug("No valid positive pairs found, returning 0 loss")
            return torch.tensor(0.0, device=features.device) * self.loss_weight

        # 总损失（加权求和）
        avg_info_nce = total_info_nce / M
        avg_continuity = total_continuity / M
        total_loss = (avg_info_nce + self.gamma * avg_continuity) * self.loss_weight

        logger.debug(
            f"PLCCLoss complete - total time: {time.time() - start_time:.4f}s - "
            f"total loss: {total_loss.item()} - total positive pairs: {total_pairs}")

        return total_loss
