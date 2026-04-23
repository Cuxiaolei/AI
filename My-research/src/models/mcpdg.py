# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDGClassifier, BaseDGConfig
from src.losses.loss_aggregator import compute_branch_loss
from src.components.condition_encoder import ConditionEncoder
from src.components.dynamic_prototype import DynamicPrototypeGenerator
from src.components.proto_ops import negative_sq_logits
from src.datasets import AsymMetaSplitConfig, AsymMetaSplitter


@dataclass
class MCPDGConfig(BaseDGConfig):
    cond_dim: int = 3
    cond_hidden_dim: int = 64
    proto_hidden_dim: int = 256

    # switches
    use_linear_head: bool = True
    use_dynamic_proto: bool = True
    use_proto_cls: bool = True
    use_align_loss: bool = True
    use_pcl_loss: bool = False
    use_meta_loss: bool = True

    # new switches
    use_disentangled_proto: bool = True
    use_minority_calib_loss: bool = True

    # weights
    proto_residual_alpha: float = 0.2
    proto_cls_weight: float = 0.5
    eval_proto_weight: float = 0.5
    align_weight: float = 1.0
    pcl_weight: float = 0.1
    pcl_temperature: float = 0.1
    meta_test_weight: float = 1.0
    imbalance_power: float = 0.5
    minority_calib_weight: float = 0.1

    meta_split_seed: int = 42
    meta_debug: bool = False
    meta_debug_max_steps: int = 20


class MCPDGClassifier(BaseDGClassifier):
    def __init__(self, cfg: MCPDGConfig) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.num_classes = int(cfg.num_classes)

        self.use_linear_head = bool(cfg.use_linear_head)
        self.use_dynamic_proto = bool(cfg.use_dynamic_proto)
        self.use_proto_cls = bool(cfg.use_proto_cls)
        self.use_align_loss = bool(cfg.use_align_loss)
        self.use_pcl_loss = bool(cfg.use_pcl_loss)
        self.use_meta_loss = bool(cfg.use_meta_loss)

        self.use_disentangled_proto = bool(cfg.use_disentangled_proto)
        self.use_minority_calib_loss = bool(cfg.use_minority_calib_loss)

        self.proto_cls_weight = float(cfg.proto_cls_weight)
        self.eval_proto_weight = float(cfg.eval_proto_weight)
        self.align_weight = float(cfg.align_weight)
        self.pcl_weight = float(cfg.pcl_weight)
        self.pcl_temperature = float(cfg.pcl_temperature)
        self.meta_test_weight = float(cfg.meta_test_weight)
        self.imbalance_power = float(cfg.imbalance_power)
        self.minority_calib_weight = float(cfg.minority_calib_weight)

        self.register_buffer(
            "condition_table",
            torch.zeros(0, int(cfg.cond_dim)),
            persistent=False
        )

        self.class_embed = nn.Embedding(
            self.num_classes,
            self.feat_dim
        )

        self.cond_encoder = ConditionEncoder(
            input_dim=int(cfg.cond_dim),
            hidden_dim=int(cfg.cond_hidden_dim),
            out_dim=self.feat_dim,
        )

        self.proto_generator = DynamicPrototypeGenerator(
            feat_dim=self.feat_dim,
            hidden_dim=int(cfg.proto_hidden_dim),
            alpha=float(cfg.proto_residual_alpha),
        )

        self.meta_splitter = AsymMetaSplitter(
            AsymMetaSplitConfig(
                debug=bool(cfg.meta_debug),
            )
        )

    def set_condition_lookup(self, condition_table) -> None:
        if condition_table.dim() != 2:
            raise ValueError("condition_table tensor must be [num_domains, cond_dim]")
        if condition_table.size(1) != int(self.cfg.cond_dim):
            raise ValueError(
                f"condition_table second dim must equal cond_dim={self.cfg.cond_dim}, "
                f"got {condition_table.size(1)}"
            )
        self.condition_table = condition_table.float()
        return

    def _lookup_condition(self, domains: torch.Tensor) -> torch.Tensor:
        if self.condition_table.size(0) == 0:
            raise RuntimeError("Condition lookup table is empty. Please call set_condition_lookup().")

        cond_dim = self.condition_table.size(1)
        cond_vec = self.condition_table.new_zeros((domains.numel(), cond_dim))

        known_mask = domains < self.condition_table.size(0)
        if known_mask.any():
            cond_vec[known_mask] = self.condition_table[domains[known_mask]]

        # strict DG fallback:
        # unseen target-domain ids use mean source condition instead of test-domain metadata
        if (~known_mask).any():
            fallback = self.condition_table.mean(dim=0, keepdim=True)
            cond_vec[~known_mask] = fallback.expand(int((~known_mask).sum().item()), -1)

        return cond_vec

    def _build_proto_bank(self, unique_domains: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
        class_anchor = F.normalize(self.class_embed.weight, dim=-1)
        d = unique_domains.numel()

        proto_invariant_bank = class_anchor.unsqueeze(0).expand(d, -1, -1).contiguous()
        proto_residual_bank = torch.zeros_like(proto_invariant_bank)
        proto_bank = F.normalize(proto_invariant_bank, dim=-1)

        cond_emb = None
        if self.use_dynamic_proto:
            cond_vec = self._lookup_condition(unique_domains).to(device)   # [D, cond_dim]
            cond_emb = self.cond_encoder(cond_vec)                         # [D, C]

            if self.use_disentangled_proto:
                proto_parts = self.proto_generator(
                    class_anchor=class_anchor,
                    cond_emb=cond_emb,
                    return_parts=True,
                )
                proto_bank = proto_parts["proto"]
                proto_invariant_bank = proto_parts["proto_base"]
                proto_residual_bank = proto_parts["proto_residual"]
            else:
                proto_bank = self.proto_generator(class_anchor, cond_emb)
                proto_residual_bank = proto_bank - proto_invariant_bank

        return {
            "proto_bank": proto_bank,
            "proto_invariant_bank": proto_invariant_bank,
            "proto_residual_bank": proto_residual_bank,
            "cond_emb": cond_emb,
            "class_anchor": class_anchor,
        }

    def _combine_logits(
        self,
        logits_linear: Optional[torch.Tensor],
        logits_proto: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.use_linear_head and logits_linear is not None and self.use_proto_cls and logits_proto is not None:
            return logits_linear + self.eval_proto_weight * logits_proto
        if self.use_linear_head and logits_linear is not None:
            return logits_linear
        if self.use_proto_cls and logits_proto is not None:
            return logits_proto
        raise RuntimeError("No valid logits branch is enabled.")

    def _forward_branch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feat_out = self.extract_features(batch)
        feature = F.normalize(feat_out["feature"], dim=-1)

        y = batch["y"]
        domains = batch["domain"]

        unique_domains, inverse_domain_index = torch.unique(
            domains, sorted=True, return_inverse=True
        )

        proto_pack = self._build_proto_bank(unique_domains, device=feature.device)
        proto_bank = proto_pack["proto_bank"]

        logits_linear = self.forward_logits(feature) if self.use_linear_head else None
        logits_proto = negative_sq_logits(feature, proto_bank, inverse_domain_index) if self.use_proto_cls else None
        logits = self._combine_logits(logits_linear, logits_proto)

        out = {
            **feat_out,
            "feature": feature,
            "logits": logits,
            "logits_linear": logits_linear,
            "logits_proto": logits_proto,

            "proto_bank": proto_pack["proto_bank"],
            "proto_invariant_bank": proto_pack["proto_invariant_bank"],
            "proto_residual_bank": proto_pack["proto_residual_bank"],
            "cond_emb": proto_pack["cond_emb"],
            "class_anchor": proto_pack["class_anchor"],

            "unique_domains": unique_domains,
            "inverse_domain_index": inverse_domain_index,
            "y": y,
            "domain": domains,
        }
        return out

    def _compute_branch_objective(self, out: Dict[str, torch.Tensor], criterion) -> Dict[str, torch.Tensor]:
        return compute_branch_loss(
            out=out,
            criterion=criterion,
            use_linear_head=self.use_linear_head,
            use_proto_cls=self.use_proto_cls,
            proto_cls_weight=self.proto_cls_weight,
            use_align_loss=self.use_align_loss,
            align_weight=self.align_weight,
            use_pcl_loss=self.use_pcl_loss,
            pcl_weight=self.pcl_weight,
            pcl_temperature=self.pcl_temperature,
            use_minority_calib_loss=self.use_minority_calib_loss,
            minority_calib_weight=self.minority_calib_weight,
            imbalance_power=self.imbalance_power,
            num_classes=self.num_classes,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = self._forward_branch(batch)
        return {
            "logits": out["logits"],
            "feature": out["feature"],
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor], criterion, epoch: int = 0, global_step: int = 0) -> Dict[str, torch.Tensor]:
        full_out = self._forward_branch(batch)

        def _pack_full_batch_result(stat: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            zero = stat["loss"].detach().new_tensor(0.0)
            return {
                "logits": full_out["logits"],
                "feature": full_out["feature"],
                "loss": stat["loss"],

                "loss_cls": stat["loss_cls"],
                "loss_cls_linear": stat["loss_cls_linear"],
                "loss_cls_proto": stat["loss_cls_proto"],
                "loss_align": stat["loss_align"],
                "loss_pcl": stat["loss_pcl"],
                "loss_minority_calib": stat["loss_minority_calib"],

                # 为了和 meta 模式下的日志字段保持一致，这里补 0
                "loss_cls_meta": zero,
                "loss_cls_linear_meta": zero,
                "loss_cls_proto_meta": zero,
                "loss_align_meta": zero,
                "loss_pcl_meta": zero,
                "loss_minority_calib_meta": zero,
            }

        # 不启用 meta loss，直接普通训练
        if not self.use_meta_loss:
            stat = self._compute_branch_objective(full_out, criterion)
            return _pack_full_batch_result(stat)


        split = self.meta_splitter.split(batch, step=global_step)

        # 关键修复：切分失败时回退到普通整 batch 损失
        if split is None:
            if self.cfg.meta_debug and global_step < self.cfg.meta_debug_max_steps:
                domains = batch.get("domain", None)
                labels = batch.get("y", None)

                msg = f"[AsymMetaSplit][Step {global_step}] fallback_to_full_batch"
                if domains is not None:
                    unique_domains, counts = torch.unique(domains, return_counts=True)
                    msg += (
                        f" unique_domains={unique_domains.detach().cpu().tolist()}"
                        f" domain_counts={counts.detach().cpu().tolist()}"
                    )
                if labels is not None:
                    cls_ids, cls_counts = torch.unique(labels, return_counts=True)
                    msg += (
                        f" class_counts="
                        f"{dict(zip(cls_ids.detach().cpu().tolist(), cls_counts.detach().cpu().tolist()))}"
                    )
                print(msg)

            stat = self._compute_branch_objective(full_out, criterion)
            return _pack_full_batch_result(stat)

        meta_train_batch, meta_test_batch, _, _ = split

        train_out = self._forward_branch(meta_train_batch)
        test_out = self._forward_branch(meta_test_batch)

        train_stat = self._compute_branch_objective(train_out, criterion)
        test_stat = self._compute_branch_objective(test_out, criterion)

        total_loss = train_stat["loss"] + self.meta_test_weight * test_stat["loss"]

        return {
            "logits": full_out["logits"],
            "feature": full_out["feature"],
            "loss": total_loss,

            "loss_cls": train_stat["loss_cls"],
            "loss_cls_linear": train_stat["loss_cls_linear"],
            "loss_cls_proto": train_stat["loss_cls_proto"],
            "loss_align": train_stat["loss_align"],
            "loss_pcl": train_stat["loss_pcl"],
            "loss_minority_calib": train_stat["loss_minority_calib"],

            "loss_cls_meta": test_stat["loss_cls"],
            "loss_cls_linear_meta": test_stat["loss_cls_linear"],
            "loss_cls_proto_meta": test_stat["loss_cls_proto"],
            "loss_align_meta": test_stat["loss_align"],
            "loss_pcl_meta": test_stat["loss_pcl"],
            "loss_minority_calib_meta": test_stat["loss_minority_calib"],
        }