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
                 physical_weight=0.2,
                 debug=False):
        super().__init__()
        self.ignore_index = ignore_index
        self.pseudo_threshold = pseudo_threshold
        self.curvature_threshold = curvature_threshold
        self.pseudo_weight = pseudo_weight
        self.physical_weight = physical_weight
        self.debug = debug

        if self.debug:
            print("PLPLELoss initialized with parameters:")
            print(f"  ignore_index: {ignore_index}, pseudo_threshold: {pseudo_threshold}")
            print(f"  curvature_threshold: {curvature_threshold}, pseudo_weight: {pseudo_weight}")
            print(f"  physical_weight: {physical_weight}")

    def forward(self, preds, targets, curvatures=None, **kwargs):
        device = preds.device
        if self.debug:
            print("\n===== PLPLELoss forward started =====")
            print(f"  preds shape: {preds.shape}, device: {device}")
            print(f"  targets shape: {targets.shape}, device: {device}")
            print(f"  curvatures exists: {curvatures is not None}")
            if curvatures is not None:
                print(f"  curvatures shape: {curvatures.shape}, device: {curvatures.device}")

        # 无曲率信息时返回0损失
        if curvatures is None:
            if self.debug:
                print("  curvatures is None, return 0.0 loss")
            return torch.tensor(0.0, device=device)

        # 关键修改1：获取电力线类别的logits（不经过softmax），用于后续损失计算
        power_line_logits = preds[:, 2]  # 类别2的logits
        # 计算概率用于阈值判断（仅用于掩码计算，不参与损失）
        power_line_prob = F.softmax(preds, dim=1)[:, 2]

        if self.debug:
            print(f"  power_line_prob stats: mean={power_line_prob.mean().item():.4f}, "
                  f"max={power_line_prob.max().item():.4f}, min={power_line_prob.min().item():.4f}")

        # 有效区域掩码
        valid_mask = targets != self.ignore_index
        valid_count = valid_mask.sum().item()
        if self.debug:
            print(f"  valid elements count: {valid_count}")

        # 无有效元素时返回0损失
        if valid_count == 0:
            if self.debug:
                print("  No valid elements, return 0.0 loss")
            return torch.tensor(0.0, device=device)

        # 1. 物理先验损失（关键修改2：使用带logits的二分类交叉熵，兼容AMP）
        valid_curvatures = curvatures[valid_mask]
        # 使用logits计算损失（无需提前softmax）
        valid_pl_logits = power_line_logits[valid_mask]
        physical_target = (valid_curvatures < self.curvature_threshold).float()

        # 关键修改3：替换为binary_cross_entropy_with_logits
        physical_loss = F.binary_cross_entropy_with_logits(
            valid_pl_logits,  # 输入logits而非概率
            physical_target,
            reduction='mean'
        )

        if self.debug:
            low_curv_count = physical_target.sum().item()
            print(f"\n  Physical prior loss: {physical_loss.item():.6f}")
            print(f"  Low curvature (power line) ratio: {low_curv_count / valid_count * 100:.2f}%")

        # 2. 伪标签损失（保持不变）
        cond1 = (power_line_prob > self.pseudo_threshold) & (power_line_prob < 1 - self.pseudo_threshold)
        cond2 = curvatures < self.curvature_threshold
        pseudo_mask = cond1 & cond2 & valid_mask
        pseudo_count = pseudo_mask.sum().item()

        if pseudo_count > 0:
            pseudo_targets = targets.clone()
            pseudo_targets[pseudo_mask] = 2  # 标记为电力线类别
            pseudo_loss = F.cross_entropy(
                preds,
                pseudo_targets,
                ignore_index=self.ignore_index,
                reduction='mean'
            )
            if self.debug:
                print(f"\n  Pseudo label loss: {pseudo_loss.item():.6f}")
                print(f"  Pseudo label ratio: {pseudo_count / valid_count * 100:.2f}%")
        else:
            pseudo_loss = torch.tensor(0.0, device=device)
            if self.debug:
                print("\n  No valid pseudo labels, pseudo_loss=0.0")

        # 总损失计算
        total_loss = self.pseudo_weight * pseudo_loss + self.physical_weight * physical_loss

        if self.debug:
            print("\n  Loss breakdown:")
            print(f"  weighted pseudo_loss: {self.pseudo_weight * pseudo_loss.item():.6f}")
            print(f"  weighted physical_loss: {self.physical_weight * physical_loss.item():.6f}")
            print(f"  Total PLPLE loss: {total_loss.item():.6f}")
            print("===== PLPLELoss forward ended =====\n")

        return total_loss
