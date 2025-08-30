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
                 max_batch_size=3000,  # 减小批次大小，更保守的内存使用
                 pos_dist_thresh=1.0,
                 neg_sample_ratio=1.5,  # 降低负样本比例
                 max_neg_samples=2000,  # 新增：负样本绝对数量上限
                 memory_safe_mode=False  # 新增：内存安全模式开关
                 ):
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma
        self.loss_weight = loss_weight
        self.requires_coords = True
        self.ignore_index = ignore_index
        self.max_batch_size = max_batch_size
        self.pos_dist_thresh = pos_dist_thresh
        self.neg_sample_ratio = neg_sample_ratio
        self.max_neg_samples = max_neg_samples
        self.memory_safe_mode = memory_safe_mode  # 新增参数

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

        # 内存安全模式下进一步减小批次大小
        batch_size = self.max_batch_size
        if self.memory_safe_mode and M > 10000:
            batch_size = max(1000, batch_size // 2)
            logger.debug(f"Memory safe mode activated, reduced batch size to {batch_size}")

        # 计算总批次数量
        num_batches = (M + batch_size - 1) // batch_size
        logger.debug(f"Split into {num_batches} batches (max size: {batch_size})")

        total_info_nce = 0.0
        total_continuity = 0.0
        total_pairs = 0

        # 分批处理以避免内存溢出
        for batch_idx in range(num_batches):
            # 计算当前批次的索引范围
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, M)
            current_batch_size = end - start

            # 当前批次的特征和坐标
            batch_feats = line_feats[start:end]  # [B, C]
            batch_coords = line_coords[start:end]  # [B, 3]

            # 计算当前批次与所有电力线点的特征相似度 [B, M]
            sim_start = time.time()
            # 内存优化：使用mm代替unsqueeze+cosine_similarity，减少中间变量
            feat_norm = F.normalize(batch_feats, dim=1)
            all_feat_norm = F.normalize(line_feats, dim=1)
            feat_sim = torch.mm(feat_norm, all_feat_norm.t())  # 余弦相似度等价计算
            logger.debug(
                f"Batch {batch_idx} similarity took {time.time() - sim_start:.4f}s - "
                f"shape: {feat_sim.shape}")

            # 计算当前批次与所有点的空间距离 [B, M]
            dist_start = time.time()
            # 内存优化：只计算必要的距离
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

            # 负样本采样（优化版本）
            neg_mask = ~pos_mask
            # 每个点的正样本数量
            pos_per_point = pos_mask.sum(dim=1, keepdim=True)
            # 计算每个点的最大负样本数量（取比例和绝对上限的最小值）
            max_neg_per_point = torch.min(
                (pos_per_point * self.neg_sample_ratio).int(),
                torch.tensor(self.max_neg_samples, device=pos_per_point.device)
            ).squeeze(1)

            # 确保至少有一个负样本
            max_neg_per_point = torch.clamp(max_neg_per_point, min=1)

            # 计算InfoNCE损失（当前批次）
            nce_start = time.time()
            exp_sim = torch.exp(feat_sim / self.temperature)

            # 负样本求和（带采样）
            sum_exp = torch.zeros(current_batch_size, device=exp_sim.device)
            for i in range(current_batch_size):
                # 获取当前点的负样本
                neg_indices = torch.nonzero(neg_mask[i]).squeeze(1)
                if neg_indices.numel() == 0:
                    sum_exp[i] = 1e-6  # 避免除零
                    continue

                # 采样负样本
                k = min(max_neg_per_point[i], neg_indices.numel())
                if k <= 0:
                    sum_exp[i] = 1e-6
                    continue

                # 对负样本按相似度排序并取top k
                neg_values = exp_sim[i, neg_indices]
                top_neg_values, _ = torch.topk(neg_values, k=k)
                sum_exp[i] = top_neg_values.sum()

            # 正样本求和
            pos_sum = (exp_sim * pos_mask.float()).sum(dim=1)

            # 计算当前批次的InfoNCE损失
            batch_info_nce = -torch.log(pos_sum / (sum_exp + pos_sum + 1e-6)).mean()
            logger.debug(
                f"Batch {batch_idx} InfoNCE: {batch_info_nce.item()} "
                f"(took {time.time() - nce_start:.4f}s)")

            # 计算连续性约束损失（当前批次）- 内存优化版本
            cont_start = time.time()
            # 只计算正样本对的连续性损失，减少计算量
            feat_dist = 1 - feat_sim  # 余弦距离（1-相似度）
            pos_feat_dist = feat_dist * pos_mask.float()
            pos_coord_dist = coord_dist * pos_mask.float()

            # 只对有正样本的点计算损失
            valid_mask = pos_mask.sum(dim=1) > 0
            if valid_mask.sum() == 0:
                batch_continuity = torch.tensor(0.0, device=feat_dist.device)
            else:
                batch_continuity = torch.mean(
                    torch.abs(pos_feat_dist[valid_mask] - pos_coord_dist[valid_mask])
                )

            logger.debug(
                f"Batch {batch_idx} continuity: {batch_continuity.item()} "
                f"(took {time.time() - cont_start:.4f}s)")

            # 累加损失（按批次大小加权）
            total_info_nce += batch_info_nce * current_batch_size
            total_continuity += batch_continuity * current_batch_size

            # 显式释放当前批次的中间变量内存
            del feat_sim, coord_dist, pos_mask, exp_sim, sum_exp, pos_sum
            torch.cuda.empty_cache()

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