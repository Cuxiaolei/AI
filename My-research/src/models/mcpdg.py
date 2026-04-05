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
from src.prototype.proto_ops import negative_sq_logits


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

    # weights
    proto_residual_alpha: float = 0.2
    proto_cls_weight: float = 0.5
    eval_proto_weight: float = 0.5
    align_weight: float = 1.0
    pcl_weight: float = 0.1
    pcl_temperature: float = 0.1
    imbalance_power: float = 0.5

class MCPDGClassifier(BaseDGClassifier):
    def __init__(self, cfg: MCPDGConfig) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.num_classes = int(cfg.num_classes)

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

        self.proto_cls_weight = float(cfg.proto_cls_weight)
        self.eval_proto_weight = float(cfg.eval_proto_weight)
        self.align_weight = float(cfg.align_weight)
        self.pcl_weight = float(cfg.pcl_weight)
        self.pcl_temperature = float(cfg.pcl_temperature)
        self.imbalance_power = float(cfg.imbalance_power)

        self.register_buffer("condition_table", torch.zeros(1, int(cfg.cond_dim)), persistent=False)

    # external hook for main.py
    def set_condition_lookup(self, condition_table) -> None:
        # Accept:
        #     1) torch.Tensor [num_domains, cond_dim]
        #     2) dict[int, list/tuple/tensor]
        if torch.is_tensor(condition_table):
            if condition_table.dim() != 2: raise ValueError("condition_table tensor must be [num_domains, cond_dim]")
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

    def _build_proto_bank(self, unique_domains: torch.Tensor, device: torch.device) -> torch.Tensor:
        # return: [D, K, C]
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

    def _compute_branch_objective(
            self,
            out: Dict[str, torch.Tensor],
            criterion,
    ) -> Dict[str, torch.Tensor]:
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
            imbalance_power=self.imbalance_power,
            num_classes=self.num_classes,
        )


    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = self._forward_branch(batch)
        return {
            "logits": out["logits"],
            "feature": out["feature"],
        }

    def compute_loss(
            self,
            batch: Dict[str, torch.Tensor],
            criterion,
            epoch: int = 0,
            global_step: int = 0,
    ) -> Dict[str, torch.Tensor]:

        full_out = self._forward_branch(batch)
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