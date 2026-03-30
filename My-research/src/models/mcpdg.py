# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDGClassifier, BaseDGConfig
from src.losses.prototype_losses import (
    masked_proto_align_loss,
    sample_prototype_contrastive_loss,
)
from src.models.components.condition_encoder import ConditionEncoder
from src.models.components.dynamic_prototype import DynamicPrototypeGenerator
from src.models.prototype.proto_ops import (
    negative_sq_logits,
    empirical_prototypes,
)
from src.samplers.asym_meta_split import AsymMetaSplitConfig, AsymMetaSplitter




def _inverse_frequency_weights(
    labels: torch.Tensor,
    num_classes: int,
    power: float = 0.5,
) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = 1.0 / (counts ** power)
    weights = weights / weights.mean()
    return weights


# =========================================================
# config
# =========================================================
@dataclass
class MCPDGConfig(BaseDGConfig):
    # condition / prototype dims
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

    # weights
    proto_residual_alpha: float = 0.2
    proto_cls_weight: float = 0.5
    eval_proto_weight: float = 0.5
    align_weight: float = 1.0
    pcl_weight: float = 0.1
    pcl_temperature: float = 0.1
    meta_test_weight: float = 1.0
    imbalance_power: float = 0.5

    # meta split
    meta_test_domains: int = 1
    meta_randomize: bool = True
    meta_split_seed: int = 42

    asym_meta: bool = True
    meta_debug: bool = False



# =========================================================
# model
# =========================================================
class MCPDGClassifier(BaseDGClassifier):
    """
    Debug / ablation version of MCPDG.

    Goals:
    - keep current project unchanged as much as possible
    - switch behaviors by config only
    - support freq / tf / both through BaseDGClassifier
    """

    def __init__(self, cfg: MCPDGConfig) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.num_classes = int(cfg.num_classes)

        if (not cfg.use_linear_head) and (not cfg.use_proto_cls):
            raise ValueError("At least one of use_linear_head or use_proto_cls must be True.")

        self.class_embed = nn.Embedding(self.num_classes, self.feat_dim)
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

        self.use_linear_head = bool(cfg.use_linear_head)
        self.use_dynamic_proto = bool(cfg.use_dynamic_proto)
        self.use_proto_cls = bool(cfg.use_proto_cls)
        self.use_align_loss = bool(cfg.use_align_loss)
        self.use_pcl_loss = bool(cfg.use_pcl_loss)
        self.use_meta_loss = bool(cfg.use_meta_loss)

        self.proto_cls_weight = float(cfg.proto_cls_weight)
        self.eval_proto_weight = float(cfg.eval_proto_weight)
        self.align_weight = float(cfg.align_weight)
        self.pcl_weight = float(cfg.pcl_weight)
        self.pcl_temperature = float(cfg.pcl_temperature)
        self.meta_test_weight = float(cfg.meta_test_weight)
        self.imbalance_power = float(cfg.imbalance_power)

        self.meta_splitter = AsymMetaSplitter(
            AsymMetaSplitConfig(
                debug=bool(cfg.meta_debug),
            )
        )

        # filled by main.py through set_condition_lookup()
        self.register_buffer("condition_table", torch.zeros(1, int(cfg.cond_dim)), persistent=False)

    # -----------------------------------------------------
    # external hook for main.py
    # -----------------------------------------------------
    def set_condition_lookup(self, condition_table) -> None:
        """
        Accept:
            1) torch.Tensor [num_domains, cond_dim]
            2) dict[int, list/tuple/tensor]
        """
        if torch.is_tensor(condition_table):
            if condition_table.dim() != 2:
                raise ValueError("condition_table tensor must be [num_domains, cond_dim]")
            self.condition_table = condition_table.float()
            return

        if isinstance(condition_table, dict):
            max_domain_id = max(int(k) for k in condition_table.keys())
            table = torch.zeros(max_domain_id + 1, int(self.cfg.cond_dim), dtype=torch.float32)
            for domain_id, cond_vec in condition_table.items():
                table[int(domain_id)] = torch.as_tensor(cond_vec, dtype=torch.float32)
            self.condition_table = table
            return

        raise TypeError("condition_table must be torch.Tensor or dict")

    def _lookup_condition(self, domains: torch.Tensor) -> torch.Tensor:
        max_id = int(domains.max().item())
        if max_id >= self.condition_table.size(0):
            raise RuntimeError("Condition lookup table is missing or incomplete.")
        return self.condition_table[domains]

    # -----------------------------------------------------
    # prototype branch
    # -----------------------------------------------------
    def _build_proto_bank(self, unique_domains: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        return: [D, K, C]
        """
        class_anchor = F.normalize(self.class_embed.weight, dim=-1)

        if self.use_dynamic_proto:
            cond_vec = self._lookup_condition(unique_domains).to(device)   # [D, cond_dim]
            cond_emb = self.cond_encoder(cond_vec)                         # [D, C]
            proto_bank = self.proto_generator(class_anchor, cond_emb)      # [D, K, C]
        else:
            d = unique_domains.numel()
            proto_bank = class_anchor.unsqueeze(0).expand(d, -1, -1).contiguous()
            proto_bank = F.normalize(proto_bank, dim=-1)

        return proto_bank

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

    # -----------------------------------------------------
    # branch forward
    # -----------------------------------------------------
    def _forward_branch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feat_out = self.extract_features(batch)
        feature = F.normalize(feat_out["feature"], dim=-1)

        y = batch["y"]
        domains = batch["domain"]

        unique_domains, inverse_domain_index = torch.unique(
            domains, sorted=True, return_inverse=True
        )

        proto_bank = self._build_proto_bank(unique_domains, device=feature.device)

        logits_linear = self.forward_logits(feature) if self.use_linear_head else None
        logits_proto = negative_sq_logits(feature, proto_bank, inverse_domain_index) if self.use_proto_cls else None
        logits = self._combine_logits(logits_linear, logits_proto)

        out = {
            **feat_out,
            "feature": feature,
            "logits": logits,
            "logits_linear": logits_linear,
            "logits_proto": logits_proto,
            "proto_bank": proto_bank,
            "unique_domains": unique_domains,
            "inverse_domain_index": inverse_domain_index,
            "y": y,
            "domain": domains,
        }
        return out

    # -----------------------------------------------------
    # losses
    # -----------------------------------------------------
    def _compute_branch_objective(
        self,
        out: Dict[str, torch.Tensor],
        criterion,
    ) -> Dict[str, torch.Tensor]:
        feature = out["feature"]
        y = out["y"]
        domains = out["domain"]
        unique_domains = out["unique_domains"]
        proto_bank = out["proto_bank"]
        logits_linear = out["logits_linear"]
        logits_proto = out["logits_proto"]

        zero = feature.new_tensor(0.0)

        # classification losses
        loss_cls_linear = criterion(logits_linear, y) if (self.use_linear_head and logits_linear is not None) else zero
        loss_cls_proto = criterion(logits_proto, y) if (self.use_proto_cls and logits_proto is not None) else zero

        if self.use_linear_head and self.use_proto_cls:
            loss_cls = loss_cls_linear + self.proto_cls_weight * loss_cls_proto
        elif self.use_linear_head:
            loss_cls = loss_cls_linear
        else:
            loss_cls = loss_cls_proto

        # align loss
        if self.use_align_loss and self.use_proto_cls:
            proto_emp, valid_mask = empirical_prototypes(
                feat=feature,
                labels=y,
                domains=domains,
                unique_domains=unique_domains,
                num_classes=self.num_classes,
            )
            loss_align = masked_proto_align_loss(proto_bank, proto_emp, valid_mask)
        else:
            loss_align = zero

        # prototype contrastive loss
        if self.use_pcl_loss and self.use_proto_cls:
            loss_pcl = sample_prototype_contrastive_loss(
                feat=feature,
                labels=y,
                proto_bank=proto_bank,
                temperature=self.pcl_temperature,
                imbalance_power=self.imbalance_power,
            )
        else:
            loss_pcl = zero

        total_loss = loss_cls + self.align_weight * loss_align + self.pcl_weight * loss_pcl

        return {
            "loss": total_loss,
            "loss_cls": loss_cls.detach(),
            "loss_cls_linear": loss_cls_linear.detach(),
            "loss_cls_proto": loss_cls_proto.detach(),
            "loss_align": loss_align.detach(),
            "loss_pcl": loss_pcl.detach(),
        }

    # -----------------------------------------------------
    # standard inference for test
    # -----------------------------------------------------
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = self._forward_branch(batch)
        return {
            "logits": out["logits"],
            "feature": out["feature"],
        }

    # -----------------------------------------------------
    # training entry for current Trainer
    # -----------------------------------------------------
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        criterion,
        epoch: int = 0,
        global_step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        full_out = self._forward_branch(batch)

        # no meta split
        if not self.use_meta_loss:
            stat = self._compute_branch_objective(full_out, criterion)
            return {
                "logits": full_out["logits"],
                "feature": full_out["feature"],
                "loss": stat["loss"],
                "loss_cls": stat["loss_cls"],
                "loss_cls_linear": stat["loss_cls_linear"],
                "loss_cls_proto": stat["loss_cls_proto"],
                "loss_align": stat["loss_align"],
                "loss_pcl": stat["loss_pcl"],
            }

        split = self.meta_splitter.split(batch, step=global_step)
        if split is None:
            stat = self._compute_branch_objective(full_out, criterion)
            return {
                "logits": full_out["logits"],
                "feature": full_out["feature"],
                "loss": stat["loss"],
                "loss_cls": stat["loss_cls"],
                "loss_cls_linear": stat["loss_cls_linear"],
                "loss_cls_proto": stat["loss_cls_proto"],
                "loss_align": stat["loss_align"],
                "loss_pcl": stat["loss_pcl"],
            }

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

            "loss_cls_train": train_stat["loss_cls"],
            "loss_cls_linear_train": train_stat["loss_cls_linear"],
            "loss_cls_proto_train": train_stat["loss_cls_proto"],
            "loss_align_train": train_stat["loss_align"],
            "loss_pcl_train": train_stat["loss_pcl"],

            "loss_cls_meta": test_stat["loss_cls"],
            "loss_cls_linear_meta": test_stat["loss_cls_linear"],
            "loss_cls_proto_meta": test_stat["loss_cls_proto"],
            "loss_align_meta": test_stat["loss_align"],
            "loss_pcl_meta": test_stat["loss_pcl"],
        }