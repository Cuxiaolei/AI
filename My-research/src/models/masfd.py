from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDGClassifier, BaseDGConfig, LinearClassifier


@dataclass
class MASFDConfig(BaseDGConfig):
    """Core-idea MASFD reproduction for unified freq-only inputs.

    This version is designed for fair baseline comparison under the user's
    unified strict-DG framework. Because the current project uses frequency-only
    H5 inputs, the original multimodal signal-processing pipeline is approximated
    by an adaptive multi-band mode decomposition over x_freq.
    """

    num_domains: int = 0
    num_modes: int = 3
    mode_channels: int = 32
    mode_feat_dim: int = 128
    fusion_hidden_dim: int = 128

    cls_weight: float = 1.0
    aux_cls_weight: float = 0.5
    domain_spec_weight: float = 0.5
    domain_inv_weight: float = 0.2
    ortho_weight: float = 0.05
    meta_weight: float = 0.5

    grl_lambda: float = 1.0
    eval_aux_weight: float = 0.25
    meta_min_samples: int = 2


class GradientReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReverseFn.apply(x, lambd)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
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


class AdaptiveModeDecomposer(nn.Module):
    """Learnable soft frequency-band decomposition over x_freq.

    Input: x_freq [B, 1, F]
    Output: modes [B, M, F]
    """

    def __init__(self, num_modes: int = 3, min_width: float = 0.06) -> None:
        super().__init__()
        self.num_modes = int(num_modes)
        centers = torch.linspace(0.15, 0.85, steps=self.num_modes)
        self.centers = nn.Parameter(centers)
        self.log_widths = nn.Parameter(torch.full((self.num_modes,), -2.0))
        self.min_width = float(min_width)

    def _build_masks(self, freq_bins: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        grid = torch.linspace(0.0, 1.0, steps=freq_bins, device=device, dtype=dtype)
        centers = torch.sigmoid(self.centers).unsqueeze(1)  # [M,1]
        widths = F.softplus(self.log_widths).unsqueeze(1) + self.min_width
        masks = torch.exp(-0.5 * ((grid.unsqueeze(0) - centers) / widths).pow(2))
        masks = masks / masks.sum(dim=0, keepdim=True).clamp_min(1e-6)
        return masks  # [M,F]

    def forward(self, x_freq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x_freq.dim() != 3 or x_freq.size(1) != 1:
            raise ValueError(f"Expected x_freq with shape [B,1,F], got {tuple(x_freq.shape)}")
        masks = self._build_masks(x_freq.size(-1), x_freq.device, x_freq.dtype)
        modes = x_freq * masks.unsqueeze(0)  # [B,M,F]
        return modes, masks


class LightweightModeEnhancer(nn.Module):
    """Shared lightweight enhancer for each decomposed mode."""

    def __init__(self, mode_channels: int, mode_feat_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, mode_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(mode_channels),
            nn.ELU(inplace=True),
            nn.Conv1d(mode_channels, mode_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mode_channels),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(mode_channels, mode_feat_dim)

    def forward(self, modes: torch.Tensor) -> torch.Tensor:
        # modes: [B,M,F]
        b, m, f = modes.shape
        x = modes.reshape(b * m, 1, f)
        h = self.encoder(x).flatten(1)
        h = self.proj(h)
        return h.reshape(b, m, -1)


class AdaptiveFusion(nn.Module):
    def __init__(self, feat_dim: int, mode_feat_dim: int, fusion_hidden_dim: int) -> None:
        super().__init__()
        self.mode_proj = nn.Linear(mode_feat_dim, feat_dim)
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, fusion_hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(fusion_hidden_dim, 1),
        )
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, base_feat: torch.Tensor, mode_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # base_feat: [B,D], mode_feats: [B,M,Dm]
        mode_proj = self.mode_proj(mode_feats)  # [B,M,D]
        b, m, d = mode_proj.shape
        base_ctx = base_feat.unsqueeze(1).expand(b, m, d)
        gate_in = torch.cat([base_ctx, mode_proj], dim=-1)
        scores = self.gate(gate_in).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        fused_mode = torch.sum(weights.unsqueeze(-1) * mode_proj, dim=1)
        fused = self.norm(base_feat + fused_mode)
        return fused, fused_mode, weights


class DynamicDomainHead(nn.Module):
    """Linear classifier that can expand output dimension as new domain ids appear."""

    def __init__(self, in_dim: int, out_dim: int = 1) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.fc = nn.Linear(self.in_dim, max(1, int(out_dim)))

    @property
    def out_dim(self) -> int:
        return int(self.fc.out_features)

    def ensure_out_dim(self, needed: int, device: torch.device) -> None:
        needed = max(1, int(needed))
        if self.fc.out_features >= needed:
            if self.fc.weight.device != device:
                self.fc = self.fc.to(device)
            return
        old = self.fc
        new_fc = nn.Linear(self.in_dim, needed).to(device)
        with torch.no_grad():
            new_fc.weight[: old.out_features].copy_(old.weight.data.to(device))
            new_fc.bias[: old.out_features].copy_(old.bias.data.to(device))
            if needed > old.out_features:
                nn.init.kaiming_uniform_(new_fc.weight[old.out_features:], a=5 ** 0.5)
                fan_in = self.in_dim
                bound = 1 / (fan_in ** 0.5)
                nn.init.uniform_(new_fc.bias[old.out_features:], -bound, bound)
        self.fc = new_fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MASFDClassifier(BaseDGClassifier):
    def __init__(self, cfg: MASFDConfig) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.num_domains = int(cfg.num_domains)
        self.num_modes = int(cfg.num_modes)
        self.meta_min_samples = int(cfg.meta_min_samples)
        self.grl_lambda = float(cfg.grl_lambda)
        self.eval_aux_weight = float(cfg.eval_aux_weight)

        self.decomposer = AdaptiveModeDecomposer(num_modes=cfg.num_modes)
        self.mode_enhancer = LightweightModeEnhancer(
            mode_channels=int(cfg.mode_channels),
            mode_feat_dim=int(cfg.mode_feat_dim),
        )
        self.fusion = AdaptiveFusion(
            feat_dim=self.feat_dim,
            mode_feat_dim=int(cfg.mode_feat_dim),
            fusion_hidden_dim=int(cfg.fusion_hidden_dim),
        )

        self.invariant_proj = MLP(self.feat_dim, self.feat_dim, self.feat_dim, dropout=cfg.classifier_dropout)
        self.specific_proj = MLP(self.feat_dim, self.feat_dim, self.feat_dim, dropout=cfg.classifier_dropout)
        self.invariant_bn = nn.LayerNorm(self.feat_dim)
        self.specific_bn = nn.LayerNorm(self.feat_dim)

        self.classifier = LinearClassifier(self.feat_dim, self.num_classes, dropout=cfg.classifier_dropout)
        self.aux_classifier = LinearClassifier(self.feat_dim * 2, self.num_classes, dropout=cfg.classifier_dropout)

        self.domain_classifier_spec = DynamicDomainHead(self.feat_dim, out_dim=max(1, self.num_domains))
        self.domain_classifier_inv = DynamicDomainHead(self.feat_dim, out_dim=max(1, self.num_domains))

        self.cls_weight = float(cfg.cls_weight)
        self.aux_cls_weight = float(cfg.aux_cls_weight)
        self.domain_spec_weight = float(cfg.domain_spec_weight)
        self.domain_inv_weight = float(cfg.domain_inv_weight)
        self.ortho_weight = float(cfg.ortho_weight)
        self.meta_weight = float(cfg.meta_weight)

    def _ensure_domain_heads(self, domains: Optional[torch.Tensor], device: torch.device) -> None:
        if domains is None or domains.numel() == 0:
            return
        needed = int(domains.max().item()) + 1
        self.domain_classifier_spec.ensure_out_dim(needed, device)
        self.domain_classifier_inv.ensure_out_dim(needed, device)

    def _domain_ce(self, logits: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
        if logits.numel() == 0 or domains.numel() == 0:
            return logits.new_tensor(0.0)
        return F.cross_entropy(logits, domains)

    def _orthogonality_loss(self, inv_feat: torch.Tensor, spec_feat: torch.Tensor) -> torch.Tensor:
        inv = F.normalize(inv_feat, dim=1)
        spec = F.normalize(spec_feat, dim=1)
        return (inv * spec).sum(dim=1).abs().mean()

    def _split_meta_domains(self, domains: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        unique = torch.unique(domains)
        if unique.numel() < 2:
            return None, None
        counts: List[Tuple[int, int]] = []
        for d in unique.tolist():
            counts.append((int((domains == d).sum().item()), int(d)))
        counts.sort(key=lambda t: (t[0], t[1]))
        query_domain = counts[0][1]
        support_mask = domains != query_domain
        query_mask = domains == query_domain
        if int(support_mask.sum().item()) < self.meta_min_samples or int(query_mask.sum().item()) < self.meta_min_samples:
            return None, None
        return support_mask, query_mask

    def extract_features(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x_freq = batch["x_freq"]
        base_feat = self.extract_freq_feature(x_freq)
        modes, mode_masks = self.decomposer(x_freq)
        mode_feats = self.mode_enhancer(modes)
        fused_feat, fused_mode, fusion_weights = self.fusion(base_feat, mode_feats)

        inv_feat = self.invariant_bn(self.invariant_proj(fused_feat))
        spec_feat = self.specific_bn(self.specific_proj(fused_feat))
        aux_feat = torch.cat([inv_feat, spec_feat], dim=1)

        return {
            "feature": fused_feat,
            "base_feature": base_feat,
            "mode_features": mode_feats,
            "mode_masks": mode_masks,
            "fusion_weights": fusion_weights,
            "fused_mode": fused_mode,
            "invariant_feature": inv_feat,
            "specific_feature": spec_feat,
            "aux_feature": aux_feat,
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feats = self.extract_features(batch)
        inv_logits = self.classifier(feats["invariant_feature"])
        aux_logits = self.aux_classifier(feats["aux_feature"])
        logits = inv_logits + self.eval_aux_weight * aux_logits
        feats.update({
            "logits": logits,
            "logits_inv": inv_logits,
            "logits_aux": aux_logits,
        })
        return feats

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        criterion: nn.Module,
        epoch: Optional[int] = None,
        global_step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.forward(batch)
        y = batch["y"]
        domains = batch.get("domain", None)

        cls_loss = criterion(out["logits_inv"], y)
        aux_cls_loss = criterion(out["logits_aux"], y)

        device = out["logits"].device
        domain_spec_loss = out["logits"].new_tensor(0.0)
        domain_inv_loss = out["logits"].new_tensor(0.0)
        meta_loss = out["logits"].new_tensor(0.0)

        if domains is not None:
            self._ensure_domain_heads(domains, device)
            spec_dom_logits = self.domain_classifier_spec(out["specific_feature"])
            inv_dom_logits = self.domain_classifier_inv(grad_reverse(out["invariant_feature"], self.grl_lambda))
            domain_spec_loss = self._domain_ce(spec_dom_logits, domains)
            domain_inv_loss = self._domain_ce(inv_dom_logits, domains)
            out["domain_logits_spec"] = spec_dom_logits
            out["domain_logits_inv"] = inv_dom_logits

            support_mask, query_mask = self._split_meta_domains(domains)
            if support_mask is not None and query_mask is not None:
                # Episodic fast meta-learning proxy: optimize performance on held-out source domain.
                support_logits = self.classifier(out["invariant_feature"][support_mask])
                query_logits = self.classifier(out["invariant_feature"][query_mask])
                support_loss = criterion(support_logits, y[support_mask])
                query_loss = criterion(query_logits, y[query_mask])
                meta_loss = 0.5 * (support_loss + query_loss)

        ortho_loss = self._orthogonality_loss(out["invariant_feature"], out["specific_feature"])

        total_loss = (
            self.cls_weight * cls_loss
            + self.aux_cls_weight * aux_cls_loss
            + self.domain_spec_weight * domain_spec_loss
            + self.domain_inv_weight * domain_inv_loss
            + self.ortho_weight * ortho_loss
            + self.meta_weight * meta_loss
        )

        out.update({
            "loss": total_loss,
            "ce_loss": cls_loss.detach(),
            "masfd_aux_cls_loss": aux_cls_loss.detach(),
            "masfd_domain_spec_loss": domain_spec_loss.detach(),
            "masfd_domain_inv_loss": domain_inv_loss.detach(),
            "masfd_ortho_loss": ortho_loss.detach(),
            "masfd_meta_loss": meta_loss.detach(),
            "masfd_num_modes": out["logits"].new_tensor(float(self.num_modes)),
            "masfd_eval_aux_weight": out["logits"].new_tensor(float(self.eval_aux_weight)),
        })
        return out
