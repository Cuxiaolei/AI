# -*- coding: utf-8 -*-
"""Core-idea reproduction of DFDN for strict domain generalization.

Paper:
    Tianyu Gao, Jingli Yang, Wenmin Wang, Xiaopeng Fan,
    "A Domain Feature Decoupling Network for Rotating Machinery Fault Diagnosis
    under Unseen Operating Conditions," Reliability Engineering & System Safety, 2024.

This implementation is designed for fair comparative experiments inside the current
unified framework rather than a line-by-line reproduction of the original paper.
It preserves the main ideas:
    1) feature decoupling into fault-related and domain-related branches
    2) flexible domain discriminator
    3) feature integrator for auxiliary diagnosis supervision
    4) joint optimization with decoupling regularization
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDGClassifier, BaseDGConfig, LinearClassifier


# ---------------------------
# Helpers
# ---------------------------

class _GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradReverseFn.apply(x, lambd)


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlexibleDomainDiscriminator(nn.Module):
    """A lightweight domain discriminator suitable for current pooled features."""
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_domains: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------
# Config
# ---------------------------

@dataclass
class DFDNConfig(BaseDGConfig):
    num_domains: int = 3

    decouple_hidden_dim: int = 512
    fault_feat_dim: int = 256
    domain_feat_dim: int = 256
    integrator_hidden_dim: int = 256
    domain_disc_hidden_dim: int = 256
    disc_dropout: float = 0.1

    lambda_fault_cls: float = 1.0
    lambda_aux_cls: float = 0.5
    lambda_domain_cls: float = 1.0
    lambda_adv_domain: float = 0.1
    lambda_orth: float = 0.05
    lambda_fused_align: float = 0.0

    grl_lambda: float = 1.0
    use_grl_schedule: bool = True
    grl_warmup_steps: int = 1000


# ---------------------------
# Model
# ---------------------------

class DFDNClassifier(BaseDGClassifier):
    def __init__(self, cfg: DFDNConfig) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.num_domains = int(cfg.num_domains)

        # Feature decoupler
        self.fault_projector = MLPBlock(
            in_dim=self.feat_dim,
            hidden_dim=int(cfg.decouple_hidden_dim),
            out_dim=int(cfg.fault_feat_dim),
            dropout=float(cfg.classifier_dropout),
        )
        self.domain_projector = MLPBlock(
            in_dim=self.feat_dim,
            hidden_dim=int(cfg.decouple_hidden_dim),
            out_dim=int(cfg.domain_feat_dim),
            dropout=float(cfg.classifier_dropout),
        )

        # Replace the inherited classifier so the main head operates on fault-related features.
        self.classifier = LinearClassifier(
            in_dim=int(cfg.fault_feat_dim),
            num_classes=self.num_classes,
            dropout=float(cfg.classifier_dropout),
        )

        # Flexible domain discriminators
        self.domain_discriminator = FlexibleDomainDiscriminator(
            in_dim=int(cfg.domain_feat_dim),
            hidden_dim=int(cfg.domain_disc_hidden_dim),
            num_domains=self.num_domains,
            dropout=float(cfg.disc_dropout),
        )
        self.adv_domain_discriminator = FlexibleDomainDiscriminator(
            in_dim=int(cfg.fault_feat_dim),
            hidden_dim=int(cfg.domain_disc_hidden_dim),
            num_domains=self.num_domains,
            dropout=float(cfg.disc_dropout),
        )

        # Feature integrator + auxiliary classifier
        fused_dim = int(cfg.fault_feat_dim) + int(cfg.domain_feat_dim)
        self.feature_integrator = nn.Sequential(
            nn.Linear(fused_dim, int(cfg.integrator_hidden_dim)),
            nn.BatchNorm1d(int(cfg.integrator_hidden_dim)),
            nn.ELU(inplace=True),
            nn.Dropout(float(cfg.classifier_dropout)),
        )
        self.aux_classifier = nn.Linear(int(cfg.integrator_hidden_dim), self.num_classes)

    # ---------- forward ----------
    def _grl_lambda(self, global_step: Optional[int]) -> float:
        base = float(self.cfg.grl_lambda)
        if not bool(self.cfg.use_grl_schedule):
            return base
        step = 0 if global_step is None else max(int(global_step), 0)
        warmup = max(int(self.cfg.grl_warmup_steps), 1)
        p = min(step / float(warmup), 1.0)
        schedule = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
        return base * schedule

    def extract_features(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x_freq = batch["x_freq"]
        shared_feat = self.extract_freq_feature(x_freq)

        fault_feat = self.fault_projector(shared_feat)
        domain_feat = self.domain_projector(shared_feat)
        fused_feat = self.feature_integrator(torch.cat([fault_feat, domain_feat], dim=1))

        return {
            "feature": shared_feat,
            "fault_feature": fault_feat,
            "domain_feature": domain_feat,
            "fused_feature": fused_feat,
        }

    def forward_logits(self, feature: torch.Tensor) -> torch.Tensor:
        return self.classifier(feature)

    def forward(self, batch: Dict[str, torch.Tensor], global_step: Optional[int] = None) -> Dict[str, torch.Tensor]:
        feats = self.extract_features(batch)
        fault_feat = feats["fault_feature"]
        domain_feat = feats["domain_feature"]
        fused_feat = feats["fused_feature"]

        logits = self.forward_logits(fault_feat)
        aux_logits = self.aux_classifier(fused_feat)
        domain_logits = self.domain_discriminator(domain_feat)

        grl_lambda = self._grl_lambda(global_step)
        adv_fault_feat = grad_reverse(fault_feat, grl_lambda)
        adv_domain_logits = self.adv_domain_discriminator(adv_fault_feat)

        return {
            **feats,
            "logits": logits,
            "aux_logits": aux_logits,
            "domain_logits": domain_logits,
            "adv_domain_logits": adv_domain_logits,
            "grl_lambda": fault_feat.new_tensor(grl_lambda),
        }

    # ---------- losses ----------
    @staticmethod
    def _orthogonality_loss(fault_feat: torch.Tensor, domain_feat: torch.Tensor) -> torch.Tensor:
        fault_feat = F.normalize(fault_feat, dim=1)
        domain_feat = F.normalize(domain_feat, dim=1)
        cosine_sq = (fault_feat * domain_feat).sum(dim=1).pow(2)
        return cosine_sq.mean()

    @staticmethod
    def _feature_align_loss(fault_feat: torch.Tensor, fused_feat: torch.Tensor) -> torch.Tensor:
        fault_feat = F.normalize(fault_feat, dim=1)
        fused_feat = F.normalize(fused_feat, dim=1)
        return 1.0 - (fault_feat * fused_feat).sum(dim=1).mean()

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        criterion: nn.Module,
        epoch: Optional[int] = None,
        global_step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.forward(batch, global_step=global_step)
        y = batch["y"]
        domains = batch.get("domain", None)

        # Main fault diagnosis supervision
        loss_fault_cls = criterion(out["logits"], y)
        loss_aux_cls = criterion(out["aux_logits"], y)

        # Decoupling regularization
        loss_orth = self._orthogonality_loss(out["fault_feature"], out["domain_feature"])
        loss_fused_align = self._feature_align_loss(out["fault_feature"], out["fused_feature"])

        # Domain supervision
        if domains is not None:
            loss_domain_cls = F.cross_entropy(out["domain_logits"], domains)
            loss_adv_domain = F.cross_entropy(out["adv_domain_logits"], domains)
        else:
            zero = out["logits"].new_tensor(0.0)
            loss_domain_cls = zero
            loss_adv_domain = zero

        total_loss = (
            float(self.cfg.lambda_fault_cls) * loss_fault_cls
            + float(self.cfg.lambda_aux_cls) * loss_aux_cls
            + float(self.cfg.lambda_domain_cls) * loss_domain_cls
            + float(self.cfg.lambda_adv_domain) * loss_adv_domain
            + float(self.cfg.lambda_orth) * loss_orth
            + float(self.cfg.lambda_fused_align) * loss_fused_align
        )

        out.update({
            "loss": total_loss,
            "ce_loss": loss_fault_cls.detach(),
            "dfdn_fault_cls_loss": loss_fault_cls.detach(),
            "dfdn_aux_cls_loss": loss_aux_cls.detach(),
            "dfdn_domain_cls_loss": loss_domain_cls.detach(),
            "dfdn_adv_domain_loss": loss_adv_domain.detach(),
            "dfdn_orth_loss": loss_orth.detach(),
            "dfdn_fused_align_loss": loss_fused_align.detach(),
        })
        return out
