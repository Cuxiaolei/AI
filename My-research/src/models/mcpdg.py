# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

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
from src.samplers.meta_split import DomainMetaSplitConfig, DomainMetaSplitter


@dataclass
class MCPDGConfig(BaseDGConfig):
    cond_dim: int = 3
    cond_hidden_dim: int = 64
    proto_hidden_dim: int = 256

    align_weight: float = 1.0
    pcl_weight: float = 0.2
    pcl_temperature: float = 0.1
    meta_test_weight: float = 1.0

    meta_test_domains: int = 1
    meta_randomize: bool = True
    meta_split_seed: int = 42


class MCPDGClassifier(BaseDGClassifier):
    """
    MCPDG:
    Meta Condition-aware Prototype Domain Generalization

    核心思想：
    1) 复用 BaseDGClassifier 的多模态 backbone 框架
    2) 用类别锚点 + 工况条件嵌入生成动态原型
    3) 用 MLDG 风格 source-only meta split 做训练
    4) 结合分类损失 + 原型对齐损失 + 原型对比损失
    """

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
        )

        self.align_weight = float(cfg.align_weight)
        self.pcl_weight = float(cfg.pcl_weight)
        self.pcl_temperature = float(cfg.pcl_temperature)
        self.meta_test_weight = float(cfg.meta_test_weight)

        self.meta_splitter = DomainMetaSplitter(
            DomainMetaSplitConfig(
                meta_test_domains=int(cfg.meta_test_domains),
                randomize=bool(cfg.meta_randomize),
                seed=int(cfg.meta_split_seed),
            )
        )

        # main.py 会调用 set_condition_lookup(condition_table)
        self.register_buffer("condition_table", torch.zeros(1, int(cfg.cond_dim)), persistent=False)

    # --------------------------------------------------
    # condition hook
    # --------------------------------------------------
    def set_condition_lookup(self, condition_table: torch.Tensor) -> None:
        """
        condition_table: [max_domain_id + 1, cond_dim]
        """
        if not torch.is_tensor(condition_table):
            raise TypeError("condition_table must be a torch.Tensor")
        if condition_table.dim() != 2:
            raise ValueError("condition_table must have shape [num_domains, cond_dim]")
        self.condition_table = condition_table.float()

    def _lookup_condition(self, domains: torch.Tensor) -> torch.Tensor:
        max_id = int(domains.max().item())
        if max_id >= self.condition_table.size(0):
            raise RuntimeError("Condition lookup table is missing or incomplete.")
        return self.condition_table[domains]

    # --------------------------------------------------
    # core branch
    # --------------------------------------------------
    def _forward_branch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feat_out = self.extract_features(batch)
        feature = F.normalize(feat_out["feature"], dim=-1)

        y = batch["y"]
        domains = batch["domain"]

        unique_domains, inverse_domain_index = torch.unique(
            domains, sorted=True, return_inverse=True
        )

        cond_vec = self._lookup_condition(unique_domains).to(feature.device)     # [D, cond_dim]
        cond_emb = self.cond_encoder(cond_vec)                                   # [D, C]
        proto_bank = self.proto_generator(self.class_embed.weight, cond_emb)     # [D, K, C]
        logits = negative_sq_logits(feature, proto_bank, inverse_domain_index)   # [B, K]

        out = {
            **feat_out,
            "feature": feature,
            "logits": logits,
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
        feature = out["feature"]
        logits = out["logits"]
        proto_bank = out["proto_bank"]
        y = out["y"]
        domains = out["domain"]
        unique_domains = out["unique_domains"]

        loss_cls = criterion(logits, y)

        proto_emp, valid_mask = empirical_prototypes(
            feat=feature,
            labels=y,
            domains=domains,
            unique_domains=unique_domains,
            num_classes=self.num_classes,
        )
        loss_align = masked_proto_align_loss(proto_bank, proto_emp, valid_mask)

        loss_pcl = sample_prototype_contrastive_loss(
            feat=feature,
            labels=y,
            proto_bank=proto_bank,
            temperature=self.pcl_temperature,
        )

        loss = loss_cls + self.align_weight * loss_align + self.pcl_weight * loss_pcl
        return {
            "loss": loss,
            "loss_cls": loss_cls.detach(),
            "loss_align": loss_align.detach(),
            "loss_pcl": loss_pcl.detach(),
        }

    # --------------------------------------------------
    # inference
    # --------------------------------------------------
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = self._forward_branch(batch)
        return {
            "logits": out["logits"],
            "feature": out["feature"],
            **({"feat_freq": out["feat_freq"]} if "feat_freq" in out else {}),
            **({"feat_tf": out["feat_tf"]} if "feat_tf" in out else {}),
        }

    # --------------------------------------------------
    # training entry for current Trainer
    # --------------------------------------------------
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        criterion,
        epoch: int = 0,
        global_step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        # 用完整 batch 产出 logits，保证 Trainer 能直接算 acc
        full_out = self._forward_branch(batch)

        split = self.meta_splitter.split(batch, step=global_step)
        if split is None:
            stat = self._compute_branch_objective(full_out, criterion)
            return {
                "logits": full_out["logits"],
                "feature": full_out["feature"],
                **({"feat_freq": full_out["feat_freq"]} if "feat_freq" in full_out else {}),
                **({"feat_tf": full_out["feat_tf"]} if "feat_tf" in full_out else {}),
                "loss": stat["loss"],
                "loss_cls": stat["loss_cls"],
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
            "logits": full_out["logits"],       # 给 Trainer 算整 batch acc
            "feature": full_out["feature"],
            **({"feat_freq": full_out["feat_freq"]} if "feat_freq" in full_out else {}),
            **({"feat_tf": full_out["feat_tf"]} if "feat_tf" in full_out else {}),
            "loss": total_loss,
            "loss_cls_train": train_stat["loss_cls"],
            "loss_align_train": train_stat["loss_align"],
            "loss_pcl_train": train_stat["loss_pcl"],
            "loss_cls_meta": test_stat["loss_cls"],
            "loss_align_meta": test_stat["loss_align"],
            "loss_pcl_meta": test_stat["loss_pcl"],
        }