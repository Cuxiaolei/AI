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
from src.prototype.proto_ops import negative_sq_logits, global_empirical_prototypes, fuse_proto_bank, negative_sq_logits_by_domain
from src.datasets.samplers import AsymMetaSplitConfig, AsymMetaSplitter
from src.losses.prototype_losses import masked_proto_align_loss, sample_prototype_contrastive_loss

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

    # weights
    proto_residual_alpha: float = 0.2
    proto_cls_weight: float = 0.5
    eval_proto_weight: float = 0.5
    align_weight: float = 1.0
    pcl_weight: float = 0.1
    pcl_temperature: float = 0.1
    meta_test_weight: float = 1.0
    imbalance_power: float = 0.5

    meta_split_seed: int = 42
    meta_debug: bool = False
    meta_debug_max_steps: int = 20

    meta_train_per_class: int = 2
    meta_test_per_class: int = 2
    meta_random_query_domain: bool = True

    episode_support_beta: float = 0.5
    support_per_class: int = 1
    query_train_per_class: int = 1
    meta_query_per_class: int = 2

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

        self.proto_cls_weight = float(cfg.proto_cls_weight)
        self.eval_proto_weight = float(cfg.eval_proto_weight)
        self.align_weight = float(cfg.align_weight)
        self.pcl_weight = float(cfg.pcl_weight)
        self.pcl_temperature = float(cfg.pcl_temperature)
        self.meta_test_weight = float(cfg.meta_test_weight)
        self.imbalance_power = float(cfg.imbalance_power)

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
                train_per_class=int(cfg.meta_train_per_class),
                test_per_class=int(cfg.meta_test_per_class),
                random_query_domain=bool(cfg.meta_random_query_domain),
                seed=int(cfg.meta_split_seed),
                debug=bool(cfg.meta_debug),
                debug_max_steps=int(cfg.meta_debug_max_steps),
            )
        )
        self.episode_support_beta = float(cfg.episode_support_beta)
        # self.meta_splitter = AsymMetaSplitter(
        #     AsymMetaSplitConfig(
        #         debug=bool(cfg.meta_debug),
        #     )
        # )

        self.register_buffer(
            "condition_table",
            torch.zeros(1, int(cfg.cond_dim)),
            persistent=False
        )
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
            imbalance_power=self.imbalance_power,
            num_classes=self.num_classes,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = self._forward_branch(batch)
        return {
            "logits": out["logits"],
            "feature": out["feature"],
        }

    # 正常对一个batch进行元训练域和元测试域进行分割
    def compute_loss(self, batch: Dict[str, torch.Tensor], criterion, epoch: int = 0, global_step: int = 0) -> Dict[str, torch.Tensor]:
        full_out = self._forward_branch(batch)

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
            raise ValueError(f"训练终止：meta_splitter.split() 返回空值 None，step={global_step}")

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

            "loss_cls_meta": test_stat["loss_cls"],
            "loss_cls_linear_meta": test_stat["loss_cls_linear"],
            "loss_cls_proto_meta": test_stat["loss_cls_proto"],
            "loss_align_meta": test_stat["loss_align"],
            "loss_pcl_meta": test_stat["loss_pcl"],
        }

    # 给 query_train / query_meta 统一算 logits。
    def _compute_episode_logits(
            self,
            feat: torch.Tensor,
            sample_domains: torch.Tensor,
            proto_bank: torch.Tensor,
            proto_domains: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        logits_linear = self.forward_logits(feat) if self.use_linear_head else None
        logits_proto = negative_sq_logits_by_domain(
            feat=feat,
            proto_bank=proto_bank,
            sample_domains=sample_domains,
            proto_domains=proto_domains,
        ) if self.use_proto_cls else None

        logits = self._combine_logits(logits_linear, logits_proto)
        return logits, logits_linear, logits_proto

    # 从一个 batch 只提归一化特征，不走完整 _forward_branch()。
    def _extract_normalized_feature(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        feat_out = self.extract_features(batch)
        feature = F.normalize(feat_out["feature"], dim=-1)
        return feature

    # 元学习与原型结合
    def compute_episode_loss(
            self,
            episode: Dict[str, Dict[str, torch.Tensor]],
            criterion,
            epoch: int = 0,
            global_step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        support = episode["support"]
        query_train = episode["query_train"]
        query_meta = episode["query_meta"]

        support_feat = self._extract_normalized_feature(support)
        qtr_feat = self._extract_normalized_feature(query_train)
        qmeta_feat = self._extract_normalized_feature(query_meta)

        # 1. support 经验原型
        proto_emp, valid_mask = global_empirical_prototypes(
            feat=support_feat,
            labels=support["y"],
            num_classes=self.num_classes,
        )

        # 2. 训练域 / held-out 域动态原型
        train_domains = torch.unique(query_train["domain"], sorted=True)
        meta_domains = torch.unique(query_meta["domain"], sorted=True)

        proto_dyn_train = self._build_proto_bank(train_domains, device=qtr_feat.device)
        proto_dyn_meta = self._build_proto_bank(meta_domains, device=qmeta_feat.device)

        # 3. 动态原型 + support 原型融合
        proto_fused_train = fuse_proto_bank(
            proto_dyn=proto_dyn_train,
            proto_emp=proto_emp,
            valid_mask=valid_mask,
            beta=self.episode_support_beta,
        )
        proto_fused_meta = fuse_proto_bank(
            proto_dyn=proto_dyn_meta,
            proto_emp=proto_emp,
            valid_mask=valid_mask,
            beta=self.episode_support_beta,
        )

        # 4. query_train / query_meta 分类
        logits_train, logits_linear_train, logits_proto_train = self._compute_episode_logits(
            feat=qtr_feat,
            sample_domains=query_train["domain"],
            proto_bank=proto_fused_train,
            proto_domains=train_domains,
        )
        logits_meta, logits_linear_meta, logits_proto_meta = self._compute_episode_logits(
            feat=qmeta_feat,
            sample_domains=query_meta["domain"],
            proto_bank=proto_fused_meta,
            proto_domains=meta_domains,
        )

        loss_cls_train = criterion(logits_train, query_train["y"])
        loss_cls_meta = criterion(logits_meta, query_meta["y"])

        loss_align = qtr_feat.new_tensor(0.0)
        if self.use_align_loss:
            proto_emp_expand = proto_emp.unsqueeze(0).expand(proto_fused_train.size(0), -1, -1)
            valid_expand = valid_mask.unsqueeze(0).expand(proto_fused_train.size(0), -1)
            loss_align = masked_proto_align_loss(
                proto_fused_train,
                proto_emp_expand,
                valid_expand,
            )

        loss_pcl = qtr_feat.new_tensor(0.0)
        if self.use_pcl_loss:
            feat_all = torch.cat([qtr_feat, qmeta_feat], dim=0)
            y_all = torch.cat([query_train["y"], query_meta["y"]], dim=0)
            proto_all = torch.cat([proto_fused_train, proto_fused_meta], dim=0)
            loss_pcl = sample_prototype_contrastive_loss(
                feat=feat_all,
                labels=y_all,
                proto_bank=proto_all,
                temperature=self.pcl_temperature,
                imbalance_power=self.imbalance_power,
            )

        loss = (
                loss_cls_train
                + self.meta_test_weight * loss_cls_meta
                + self.align_weight * loss_align
                + self.pcl_weight * loss_pcl
        )

        return {
            "loss": loss,
            "loss_cls": loss_cls_train.detach(),
            "loss_cls_meta": loss_cls_meta.detach(),
            "loss_align": loss_align.detach(),
            "loss_pcl": loss_pcl.detach(),
        }

