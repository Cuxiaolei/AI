import torch
from typing import Dict
from src.losses.prototype_losses import masked_proto_align_loss, sample_prototype_contrastive_loss
from src.prototype.proto_ops import empirical_prototypes  # 你原来在哪里引就放哪里


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
    imbalance_power: float,
    num_classes: int,
) -> Dict[str, torch.Tensor]:

    feature = out["feature"]
    y = out["y"]
    domains = out["domain"]
    unique_domains = out["unique_domains"]
    proto_bank = out["proto_bank"]
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
    # 对齐损失
    # ----------------------------
    loss_align = zero
    if use_align_loss and use_proto_cls:
        proto_emp, valid_mask = empirical_prototypes(
            feat=feature,
            labels=y,
            domains=domains,
            unique_domains=unique_domains,
            num_classes=num_classes,
        )
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
    # 总损失（加权求和）
    # ----------------------------
    total_loss = loss_cls + align_weight * loss_align + pcl_weight * loss_pcl

    return {
        "loss": total_loss,
        "loss_cls": loss_cls.detach(),
        "loss_cls_linear": loss_cls_linear.detach(),
        "loss_cls_proto": loss_cls_proto.detach(),
        "loss_align": loss_align.detach(),
        "loss_pcl": loss_pcl.detach(),
    }