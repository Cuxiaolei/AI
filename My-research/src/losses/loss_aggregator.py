# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from src.losses.prototype_losses import masked_proto_align_loss, sample_prototype_contrastive_loss
from src.components.proto_ops import empirical_prototypes


def _compute_class_rarity(
    labels: torch.Tensor,
    num_classes: int,
    imbalance_power: float,
) -> torch.Tensor:
    # rarity in [0, 1], majority -> 0, minority -> larger value
    counts = torch.bincount(labels, minlength=num_classes).float()
    valid = counts > 0

    rarity = counts.new_zeros(num_classes)
    if not valid.any():
        return rarity

    max_count = counts[valid].max().clamp(min=1.0)
    rarity[valid] = (max_count / counts[valid].clamp(min=1.0)) - 1.0

    if valid.sum() > 1:
        denom = rarity[valid].max().clamp(min=1e-6)
        rarity[valid] = (rarity[valid] / denom).pow(imbalance_power)
    else:
        rarity.zero_()

    return rarity


def _minority_prototype_calibration_loss(
    proto_bank: torch.Tensor,
    proto_invariant_bank: torch.Tensor,
    proto_emp: torch.Tensor,
    valid_mask: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    imbalance_power: float,
) -> torch.Tensor:
    # proto_bank:            [D, K, C]
    # proto_invariant_bank:  [D, K, C]
    # proto_emp:             [D, K, C]
    # valid_mask:            [D, K]
    zero = proto_bank.new_tensor(0.0)

    if proto_invariant_bank is None:
        return zero

    rarity = _compute_class_rarity(
        labels=labels,
        num_classes=num_classes,
        imbalance_power=imbalance_power,
    ).to(proto_bank.device)

    proto_emp = F.normalize(proto_emp, dim=-1)

    rarity_3d = rarity.view(1, num_classes, 1)
    rarity_2d = rarity.view(1, num_classes)

    # majority -> more rely on dynamic prototype
    # minority -> more rely on invariant prototype
    calibrated_proto = F.normalize(
        (1.0 - rarity_3d) * proto_bank + rarity_3d * proto_invariant_bank,
        dim=-1,
    )

    dist = ((calibrated_proto - proto_emp) ** 2).sum(dim=-1)
    weight = valid_mask.float() * (1.0 + rarity_2d)

    if weight.sum() <= 0:
        return zero

    return (dist * weight).sum() / weight.sum().clamp(min=1.0)


def compute_branch_loss(
    out: Dict[str, torch.Tensor],
    criterion,
    use_linear_head: bool,
    use_proto_cls: bool,
    proto_cls_weight: float,
    use_align_loss: bool,
    align_weight: float,
    use_pcl_loss: bool,
    pcl_weight: float,
    pcl_temperature: float,
    use_minority_calib_loss: bool,
    minority_calib_weight: float,
    imbalance_power: float,
    num_classes: int,
) -> Dict[str, torch.Tensor]:

    feature = out["feature"]
    y = out["y"]
    domains = out["domain"]
    unique_domains = out["unique_domains"]
    proto_bank = out["proto_bank"]
    proto_invariant_bank = out.get("proto_invariant_bank", None)
    logits_linear = out["logits_linear"]
    logits_proto = out["logits_proto"]

    zero = feature.new_tensor(0.0)

    # ----------------------------
    # 分类损失
    # ----------------------------
    loss_cls_linear = criterion(logits_linear, y) if (use_linear_head and logits_linear is not None) else zero
    loss_cls_proto = criterion(logits_proto, y) if (use_proto_cls and logits_proto is not None) else zero

    if use_linear_head and use_proto_cls:
        loss_cls = loss_cls_linear + proto_cls_weight * loss_cls_proto
    elif use_linear_head:
        loss_cls = loss_cls_linear
    else:
        loss_cls = loss_cls_proto

    # ----------------------------
    # 经验原型
    # ----------------------------
    proto_emp = None
    valid_mask = None
    if (use_align_loss or use_minority_calib_loss) and use_proto_cls:
        proto_emp, valid_mask = empirical_prototypes(
            feat=feature,
            labels=y,
            domains=domains,
            unique_domains=unique_domains,
            num_classes=num_classes,
        )

    # ----------------------------
    # 对齐损失
    # ----------------------------
    loss_align = zero
    if use_align_loss and use_proto_cls and proto_emp is not None:
        loss_align = masked_proto_align_loss(proto_bank, proto_emp, valid_mask)

    # ----------------------------
    # 原型对比损失
    # ----------------------------
    loss_pcl = zero
    if use_pcl_loss and use_proto_cls:
        loss_pcl = sample_prototype_contrastive_loss(
            feat=feature,
            labels=y,
            proto_bank=proto_bank,
            temperature=pcl_temperature,
            imbalance_power=imbalance_power,
        )

    # ----------------------------
    # 少数类原型校准损失
    # ----------------------------
    loss_minority_calib = zero
    if use_minority_calib_loss and use_proto_cls and proto_emp is not None:
        loss_minority_calib = _minority_prototype_calibration_loss(
            proto_bank=proto_bank,
            proto_invariant_bank=proto_invariant_bank,
            proto_emp=proto_emp,
            valid_mask=valid_mask,
            labels=y,
            num_classes=num_classes,
            imbalance_power=imbalance_power,
        )

    # ----------------------------
    # 总损失
    # ----------------------------
    total_loss = (
        loss_cls
        + align_weight * loss_align
        + pcl_weight * loss_pcl
        + minority_calib_weight * loss_minority_calib
    )

    return {
        "loss": total_loss,
        "loss_cls": loss_cls.detach(),
        "loss_cls_linear": loss_cls_linear.detach(),
        "loss_cls_proto": loss_cls_proto.detach(),
        "loss_align": loss_align.detach(),
        "loss_pcl": loss_pcl.detach(),
        "loss_minority_calib": loss_minority_calib.detach(),
    }