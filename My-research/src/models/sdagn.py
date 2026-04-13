from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDGClassifier, BaseDGConfig


@dataclass
class SDAGNConfig(BaseDGConfig):
    """Core-idea reproduction of SDAGN for unified strict-DG training.

    Main components retained:
    1) same-class input-level mixup for minority classes
    2) semantic regularization via class-conditional MMD between original and augmented features
    3) discriminative representation learning via triplet loss
    """

    sdagn_mixup_alpha: float = 0.4
    sdagn_mixup_mode: str = "beta"  # beta | uniform
    sdagn_cls_weight: float = 1.0
    sdagn_aug_cls_weight: float = 1.0
    sdagn_semantic_weight: float = 1.0
    sdagn_triplet_weight: float = 0.1
    sdagn_triplet_margin: float = 1.0
    sdagn_mmd_gamma: float = 1.0
    sdagn_mmd_num_kernels: int = 1
    sdagn_normalize_triplet_feat: bool = True
    sdagn_max_aug_per_class: int = 64
    sdagn_balance_to_max: bool = True
    sdagn_min_samples_per_class_to_mix: int = 2


class SDAGNClassifier(BaseDGClassifier):
    def __init__(self, cfg: SDAGNConfig) -> None:
        super().__init__(cfg)
        self.sdagn_mixup_alpha = float(cfg.sdagn_mixup_alpha)
        self.sdagn_mixup_mode = str(cfg.sdagn_mixup_mode).lower()
        self.sdagn_cls_weight = float(cfg.sdagn_cls_weight)
        self.sdagn_aug_cls_weight = float(cfg.sdagn_aug_cls_weight)
        self.sdagn_semantic_weight = float(cfg.sdagn_semantic_weight)
        self.sdagn_triplet_weight = float(cfg.sdagn_triplet_weight)
        self.sdagn_triplet_margin = float(cfg.sdagn_triplet_margin)
        self.sdagn_mmd_gamma = float(cfg.sdagn_mmd_gamma)
        self.sdagn_mmd_num_kernels = int(cfg.sdagn_mmd_num_kernels)
        self.sdagn_normalize_triplet_feat = bool(cfg.sdagn_normalize_triplet_feat)
        self.sdagn_max_aug_per_class = int(cfg.sdagn_max_aug_per_class)
        self.sdagn_balance_to_max = bool(cfg.sdagn_balance_to_max)
        self.sdagn_min_samples_per_class_to_mix = int(cfg.sdagn_min_samples_per_class_to_mix)

        self.triplet_loss_fn = nn.TripletMarginLoss(margin=self.sdagn_triplet_margin, p=2)

    # ------------------------------------------------------------------
    # Core utilities
    # ------------------------------------------------------------------
    def _sample_lambda(self, num: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if num <= 0:
            return torch.empty(0, device=device, dtype=dtype)
        if self.sdagn_mixup_mode == "uniform":
            lam = torch.rand(num, device=device, dtype=dtype)
        else:
            alpha = max(self.sdagn_mixup_alpha, 1e-6)
            beta_dist = torch.distributions.Beta(alpha, alpha)
            lam = beta_dist.sample((num,)).to(device=device, dtype=dtype)
        # symmetric mixup ratio, same as standard mixup stabilization
        lam = torch.maximum(lam, 1.0 - lam)
        return lam

    def _build_augmented_minority_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[int]]:
        """Generate same-class input-level mixed samples for minority classes.

        Returns:
            aug_x, aug_y, class_ids_with_aug
        """
        device = x.device
        dtype = x.dtype
        unique_classes, counts = torch.unique(y, return_counts=True)
        if unique_classes.numel() <= 1:
            return None, None, []

        max_count = int(counts.max().item())
        aug_x_list: List[torch.Tensor] = []
        aug_y_list: List[torch.Tensor] = []
        aug_classes: List[int] = []

        for cls, cnt in zip(unique_classes.tolist(), counts.tolist()):
            cls_mask = (y == cls)
            cls_idx = torch.nonzero(cls_mask, as_tuple=False).flatten()
            cls_count = int(cnt)
            if cls_count < self.sdagn_min_samples_per_class_to_mix:
                continue

            if self.sdagn_balance_to_max:
                target_aug = max(0, max_count - cls_count)
            else:
                target_aug = cls_count

            if target_aug <= 0:
                continue

            target_aug = min(target_aug, self.sdagn_max_aug_per_class)
            if target_aug <= 0:
                continue

            # sample two indices from the same class for each synthesized sample
            idx1 = cls_idx[torch.randint(0, cls_idx.numel(), (target_aug,), device=device)]
            idx2 = cls_idx[torch.randint(0, cls_idx.numel(), (target_aug,), device=device)]
            # avoid degenerate same-instance pairs when possible
            if cls_idx.numel() > 1:
                same_mask = idx1 == idx2
                retries = 0
                while bool(same_mask.any()) and retries < 3:
                    idx2[same_mask] = cls_idx[torch.randint(0, cls_idx.numel(), (int(same_mask.sum().item()),), device=device)]
                    same_mask = idx1 == idx2
                    retries += 1

            lam = self._sample_lambda(target_aug, device=device, dtype=dtype).view(-1, 1, 1)
            x1 = x[idx1]
            x2 = x[idx2]
            x_mix = lam * x1 + (1.0 - lam) * x2
            y_mix = torch.full((target_aug,), int(cls), device=device, dtype=torch.long)

            aug_x_list.append(x_mix)
            aug_y_list.append(y_mix)
            aug_classes.append(int(cls))

        if not aug_x_list:
            return None, None, []

        aug_x = torch.cat(aug_x_list, dim=0)
        aug_y = torch.cat(aug_y_list, dim=0)
        return aug_x, aug_y, aug_classes

    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xx = torch.cdist(x, x, p=2).pow(2)
        yy = torch.cdist(y, y, p=2).pow(2)
        xy = torch.cdist(x, y, p=2).pow(2)

        if self.sdagn_mmd_num_kernels <= 1:
            gamma = max(self.sdagn_mmd_gamma, 1e-8)
            k_xx = torch.exp(-gamma * xx)
            k_yy = torch.exp(-gamma * yy)
            k_xy = torch.exp(-gamma * xy)
            return k_xx, k_yy, k_xy

        base_gamma = max(self.sdagn_mmd_gamma, 1e-8)
        k_xx = 0.0
        k_yy = 0.0
        k_xy = 0.0
        for i in range(self.sdagn_mmd_num_kernels):
            gamma = base_gamma * (2.0 ** i)
            k_xx = k_xx + torch.exp(-gamma * xx)
            k_yy = k_yy + torch.exp(-gamma * yy)
            k_xy = k_xy + torch.exp(-gamma * xy)
        return k_xx, k_yy, k_xy

    def _mmd_rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.size(0) == 0 or y.size(0) == 0:
            return x.new_tensor(0.0)
        k_xx, k_yy, k_xy = self._rbf_kernel(x, y)
        return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()

    def _conditional_mmd(
        self,
        src_feat: torch.Tensor,
        src_y: torch.Tensor,
        aug_feat: torch.Tensor,
        aug_y: torch.Tensor,
        aug_classes: List[int],
    ) -> torch.Tensor:
        losses = []
        for cls in aug_classes:
            src_cls = src_feat[src_y == cls]
            aug_cls = aug_feat[aug_y == cls]
            if src_cls.size(0) == 0 or aug_cls.size(0) == 0:
                continue
            losses.append(self._mmd_rbf(src_cls, aug_cls))
        if not losses:
            return src_feat.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _batch_hard_triplets(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        n = feats.size(0)
        if n < 3:
            return None, None, None

        dist = torch.cdist(feats, feats, p=2)
        anchors = []
        positives = []
        negatives = []

        for i in range(n):
            same = labels == labels[i]
            same[i] = False
            diff = labels != labels[i]
            if not bool(same.any()) or not bool(diff.any()):
                continue

            pos_idx = torch.argmax(dist[i][same])
            neg_idx = torch.argmin(dist[i][diff])
            pos_pool = torch.nonzero(same, as_tuple=False).flatten()
            neg_pool = torch.nonzero(diff, as_tuple=False).flatten()

            anchors.append(feats[i])
            positives.append(feats[pos_pool[pos_idx]])
            negatives.append(feats[neg_pool[neg_idx]])

        if not anchors:
            return None, None, None

        return torch.stack(anchors, dim=0), torch.stack(positives, dim=0), torch.stack(negatives, dim=0)

    def _triplet_loss(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.sdagn_normalize_triplet_feat:
            feats = F.normalize(feats, p=2, dim=1)
        anc, pos, neg = self._batch_hard_triplets(feats, labels)
        if anc is None:
            return feats.new_tensor(0.0)
        return self.triplet_loss_fn(anc, pos, neg)

    # ------------------------------------------------------------------
    # Main forward/loss
    # ------------------------------------------------------------------
    def _forward_from_x(self, x_freq: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.extract_freq_feature(x_freq)
        logits = self.forward_logits(feat)
        return {
            "feature": feat,
            "logits": logits,
        }

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        criterion: nn.Module,
        epoch: Optional[int] = None,
        global_step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        x = batch["x_freq"]
        y = batch["y"]

        out_src = self._forward_from_x(x)
        src_logits = out_src["logits"]
        src_feat = out_src["feature"]
        src_cls_loss = criterion(src_logits, y)

        aug_x, aug_y, aug_classes = self._build_augmented_minority_batch(x, y)

        if aug_x is None:
            total_loss = self.sdagn_cls_weight * src_cls_loss
            return {
                **out_src,
                "loss": total_loss,
                "ce_loss": src_cls_loss.detach(),
                "sdagn_src_cls_loss": src_cls_loss.detach(),
                "sdagn_aug_cls_loss": src_cls_loss.new_tensor(0.0),
                "sdagn_semantic_loss": src_cls_loss.new_tensor(0.0),
                "sdagn_triplet_loss": src_cls_loss.new_tensor(0.0),
                "sdagn_num_aug": src_cls_loss.new_tensor(0.0),
                "sdagn_num_aug_classes": src_cls_loss.new_tensor(0.0),
            }

        out_aug = self._forward_from_x(aug_x)
        aug_logits = out_aug["logits"]
        aug_feat = out_aug["feature"]

        aug_cls_loss = criterion(aug_logits, aug_y)
        semantic_loss = self._conditional_mmd(src_feat, y, aug_feat, aug_y, aug_classes)

        joint_feat = torch.cat([src_feat, aug_feat], dim=0)
        joint_y = torch.cat([y, aug_y], dim=0)
        triplet_loss = self._triplet_loss(joint_feat, joint_y)

        total_loss = (
            self.sdagn_cls_weight * src_cls_loss
            + self.sdagn_aug_cls_weight * aug_cls_loss
            + self.sdagn_semantic_weight * semantic_loss
            + self.sdagn_triplet_weight * triplet_loss
        )

        return {
            **out_src,
            "loss": total_loss,
            "ce_loss": src_cls_loss.detach(),
            "sdagn_src_cls_loss": src_cls_loss.detach(),
            "sdagn_aug_cls_loss": aug_cls_loss.detach(),
            "sdagn_semantic_loss": semantic_loss.detach(),
            "sdagn_triplet_loss": triplet_loss.detach(),
            "sdagn_num_aug": src_cls_loss.new_tensor(float(aug_x.size(0))),
            "sdagn_num_aug_classes": src_cls_loss.new_tensor(float(len(aug_classes))),
        }
