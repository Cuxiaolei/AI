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
        print("PLCCLoss (PL-PLE Loss) initialized with parameters:")
        print(f"  ignore_index: {ignore_index}, pseudo_threshold: {pseudo_threshold}")
        print(f"  curvature_threshold: {curvature_threshold}, pseudo_weight: {pseudo_weight}")
        print(f"  physical_weight: {physical_weight}")

    def forward(self, preds, targets, original_curvature=None):
        print("\n===== PLCCLoss forward started =====")
        # 打印输入基本信息
        print(f"  preds shape: {preds.shape}, device: {preds.device}")
        print(f"  targets shape: {targets.shape}, device: {targets.device}")
        print(f"  curvatures exists: {original_curvature is not None}")
        if original_curvature is not None:
            print(f"  curvatures shape: {original_curvature.shape}, device: {original_curvature.device}")

        # 如果没有曲率信息，返回0损失（不影响基础损失）
        if original_curvature is None:
            print("  curvatures is None, return 0.0 loss")
            return torch.tensor(0.0, device=preds.device)

        # 计算预测概率
        prob = F.softmax(preds, dim=1)
        power_line_prob = prob[:, 2]  # 电力线是类别2
        print(f"  prob shape: {prob.shape}, power_line_prob shape: {power_line_prob.shape}")
        print(
            f"  power_line_prob stats: mean={power_line_prob.mean().item():.4f}, max={power_line_prob.max().item():.4f}, min={power_line_prob.min().item():.4f}")

        # 筛选有效区域（排除忽略索引）
        valid_mask = targets != self.ignore_index
        valid_count = valid_mask.sum().item()
        print(f"  valid_mask shape: {valid_mask.shape}, valid elements count: {valid_count}")
        if not valid_mask.any():
            print("  No valid elements, return 0.0 loss")
            return torch.tensor(0.0, device=preds.device)

        # 1. 物理先验损失（鼓励低曲率区域预测为电力线）
        print("\n  Calculating physical prior loss...")
        # 仅在有效区域计算
        valid_curvatures = original_curvature[valid_mask]
        valid_pl_prob = power_line_prob[valid_mask]
        print(
            f"  valid_curvatures shape: {valid_curvatures.shape}, stats: mean={valid_curvatures.mean().item():.4f}, max={valid_curvatures.max().item():.4f}")
        print(f"  valid_pl_prob shape: {valid_pl_prob.shape}, mean={valid_pl_prob.mean().item():.4f}")

        # 物理先验标签：低曲率区域应为电力线
        physical_target = (valid_curvatures < self.curvature_threshold).float()
        low_curv_count = physical_target.sum().item()
        print(
            f"  physical_target: low curvature (power line) count={low_curv_count}/{valid_count} ({low_curv_count / valid_count * 100:.2f}%)")

        physical_loss = F.binary_cross_entropy(
            valid_pl_prob,
            physical_target,
            reduction='mean'
        )
        print(f"  physical_loss value: {physical_loss.item():.6f}")

        # 2. 伪标签损失
        print("\n  Calculating pseudo label loss...")
        # 条件：中等置信度 + 低曲率 + 有效区域
        cond1 = (power_line_prob > self.pseudo_threshold) & (power_line_prob < 1 - self.pseudo_threshold)
        cond2 = original_curvature < self.curvature_threshold
        pseudo_mask = cond1 & cond2 & valid_mask
        pseudo_count = pseudo_mask.sum().item()
        print(f"  cond1 (medium confidence) count: {cond1.sum().item()}")
        print(f"  cond2 (low curvature) count: {cond2.sum().item()}")
        print(f"  pseudo_mask valid count: {pseudo_count}/{valid_count} ({pseudo_count / valid_count * 100:.2f}%)")

        if pseudo_mask.any():
            # 创建伪标签（将符合条件的区域标记为电力线）
            pseudo_targets = targets.clone()
            pseudo_targets[pseudo_mask] = 2  # 电力线类别
            print(f"  pseudo_targets updated {pseudo_count} elements to class 2")

            pseudo_loss = F.cross_entropy(
                preds,
                pseudo_targets,
                ignore_index=self.ignore_index,
                reduction='mean'
            )
            print(f"  pseudo_loss value: {pseudo_loss.item():.6f}")
        else:
            pseudo_loss = torch.tensor(0.0, device=preds.device)
            print("  No elements meet pseudo label conditions, pseudo_loss=0.0")

        # 总补充损失（带权重）
        total_loss = self.pseudo_weight * pseudo_loss + self.physical_weight * physical_loss
        print("\n  Loss breakdown:")
        print(
            f"  weighted pseudo_loss: {self.pseudo_weight} * {pseudo_loss.item():.6f} = {self.pseudo_weight * pseudo_loss.item():.6f}")
        print(
            f"  weighted physical_loss: {self.physical_weight} * {physical_loss.item():.6f} = {self.physical_weight * physical_loss.item():.6f}")
        print(f"  Total PL-PLE loss: {total_loss.item():.6f}")
        print("===== PLCCLoss forward ended =====\n")
        return total_loss
