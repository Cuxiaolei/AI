# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorOrDict = Union[torch.Tensor, Dict[str, torch.Tensor]]


def _ensure_4d_1d(x: torch.Tensor) -> torch.Tensor:
    """
    time signal: accept (k,n,L) or (k,n,1,L) -> return (B,1,L)
    """
    if x.dim() == 3:
        k, n, L = x.shape
        return x.reshape(k * n, 1, L)
    if x.dim() == 4:
        k, n, c, L = x.shape
        assert c == 1
        return x.reshape(k * n, 1, L)
    raise ValueError(f"time tensor must be 3D/4D, got {x.shape}")


def _ensure_vec(x: torch.Tensor) -> torch.Tensor:
    """
    fft vector: accept (k,n,F) -> (B,F)
    """
    if x.dim() == 3:
        k, n, f = x.shape
        return x.reshape(k * n, f)
    if x.dim() == 2:
        return x
    raise ValueError(f"fft tensor must be 2D/3D, got {x.shape}")


def _ensure_img(x: torch.Tensor) -> torch.Tensor:
    """
    tf image: accept (k,n,3,H,W) -> (B,3,H,W)
    """
    if x.dim() == 5:
        k, n, c, h, w = x.shape
        assert c == 3
        return x.reshape(k * n, c, h, w)
    if x.dim() == 4:
        return x
    raise ValueError(f"tf tensor must be 4D/5D, got {x.shape}")


class MixStyle(nn.Module):
    """
    MixStyle (ICLR'21) 风格混合：在 embedding 上随机混合均值方差，增强DG。
    这里实现成：对 (B,D) embedding 做 channel-wise mean/std 的混合。
    """
    def __init__(self, p: float = 0.5, alpha: float = 0.3, eps: float = 1e-6):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (torch.rand(1, device=x.device).item() > self.p):
            return x
        # x: (B,D)
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()

        x_normed = (x - mu) / sig

        # shuffle for mixing
        perm = torch.randperm(x.size(0), device=x.device)
        mu2, sig2 = mu[perm], sig[perm]

        # Beta(alpha, alpha)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((x.size(0), 1)).to(x.device)
        mu_mix = lam * mu + (1 - lam) * mu2
        sig_mix = lam * sig + (1 - lam) * sig2

        return x_normed * sig_mix + mu_mix


class Conv1DBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 7, s: int = 2):
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Res1DBlock(nn.Module):
    def __init__(self, c: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(c, c, k, padding=p, bias=False)
        self.bn1 = nn.BatchNorm1d(c)
        self.conv2 = nn.Conv1d(c, c, k, padding=p, bias=False)
        self.bn2 = nn.BatchNorm1d(c)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        return F.relu(x + h, inplace=True)


class TimeEncoder1D(nn.Module):
    """
    轻量 1D-ResNet encoder: (B,1,L) -> (B,D)
    """
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.stem = Conv1DBlock(1, 32, k=11, s=2)
        self.b1 = nn.Sequential(Conv1DBlock(32, 64, k=7, s=2), Res1DBlock(64), Res1DBlock(64))
        self.b2 = nn.Sequential(Conv1DBlock(64, 128, k=5, s=2), Res1DBlock(128), Res1DBlock(128))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x


class FFTEncoderMLP(nn.Module):
    """
    FFT 向量 encoder: (B,F) -> (B,D)
    """
    def __init__(self, in_dim: int, out_dim: int = 256, hidden: int = 512, drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TFEncoderCNN(nn.Module):
    """
    时频图 encoder: (B,3,H,W) -> (B,D)
    轻量CNN，避免太重。
    """
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        h = self.features(x).flatten(1)
        return self.fc(h)


class GatedFusion(nn.Module):
    """
    将多个模态 embedding 融合成一个 embedding。
    输入: dict{name: (B,D)} -> 输出: (B,D), 以及 gate weights
    """
    def __init__(self, dim: int, modalities: Tuple[str, ...]):
        super().__init__()
        self.modalities = modalities
        self.gate = nn.Linear(dim * len(modalities), len(modalities))

    def forward(self, emb: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = [emb[m] for m in self.modalities]
        cat = torch.cat(feats, dim=1)  # (B, D*M)
        w = F.softmax(self.gate(cat), dim=1)  # (B, M)
        fused = 0.0
        for i, m in enumerate(self.modalities):
            fused = fused + emb[m] * w[:, i:i+1]
        return fused, w


def episodic_labels(k: int, n_or_q: int, device) -> torch.Tensor:
    """
    生成 episode 内标签：0..k-1，每类重复 n_or_q 次
    """
    y = torch.arange(k, device=device).view(k, 1).repeat(1, n_or_q).view(-1)
    return y


def compute_prototypes(z_s: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """
    z_s: (k*n, D) -> protos: (k, D)
    """
    return z_s.view(k, n, -1).mean(dim=1)


def neg_sq_euclid_logits(z_q: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
    """
    logits = -||z_q - p||^2
    z_q: (k*q, D), protos: (k, D) -> (k*q, k)
    """
    # (N,1,D) - (1,K,D) -> (N,K,D)
    diff = z_q.unsqueeze(1) - protos.unsqueeze(0)
    dist2 = (diff ** 2).sum(dim=2)
    return -dist2


@dataclass
class PCDGConfig:
    # modalities
    use_time: bool = False
    use_fft: bool = True
    use_tf: bool = False

    # dims
    emb_dim: int = 256
    proj_dim: int = 128
    fft_in_dim: int = 256  # 你的 FFT_data 默认 256

    # DG
    mixstyle_p: float = 0.5
    mixstyle_alpha: float = 0.3


class PCDGNet(nn.Module):
    """
    Few-shot DG + Proto + SupCon + (option) Continual Prototype Memory
    forward_episode 接收 support/query（按类分组），输出 embeddings/protos/logits 供 loss 使用。
    """
    def __init__(self, cfg: PCDGConfig):
        super().__init__()
        self.cfg = cfg

        modalities = []
        self.enc_time = None
        self.enc_fft = None
        self.enc_tf = None

        if cfg.use_time:
            self.enc_time = TimeEncoder1D(out_dim=cfg.emb_dim)
            modalities.append("time")
        if cfg.use_fft:
            self.enc_fft = FFTEncoderMLP(in_dim=cfg.fft_in_dim, out_dim=cfg.emb_dim)
            modalities.append("fft")
        if cfg.use_tf:
            self.enc_tf = TFEncoderCNN(out_dim=cfg.emb_dim)
            modalities.append("tf")

        if len(modalities) == 0:
            raise ValueError("PCDGNet: at least one modality must be enabled.")

        self.modalities = tuple(modalities)
        self.fusion = GatedFusion(dim=cfg.emb_dim, modalities=self.modalities)

        # DG augmentation on embedding
        self.mixstyle = MixStyle(p=cfg.mixstyle_p, alpha=cfg.mixstyle_alpha)

        # projection head for SupCon
        self.proj = nn.Sequential(
            nn.Linear(cfg.emb_dim, cfg.emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.emb_dim, cfg.proj_dim),
        )

    def encode(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        x: dict with some keys in {"time","fft","tf"} each already flattened to batch.
        return:
          z_fused: (B,D)
          emb_dict: each modality embedding
          gate_w: (B,M)
        """
        emb = {}
        if self.enc_time is not None and "time" in x:
            emb["time"] = self.enc_time(x["time"])
        if self.enc_fft is not None and "fft" in x:
            emb["fft"] = self.enc_fft(x["fft"])
        if self.enc_tf is not None and "tf" in x:
            emb["tf"] = self.enc_tf(x["tf"])

        # keep only enabled modalities
        emb = {k: v for k, v in emb.items() if k in self.modalities}
        z, gate_w = self.fusion(emb)

        # MixStyle on fused embedding for DG robustness
        z = self.mixstyle(z)

        # normalize embedding helps metric learning
        z = F.normalize(z, dim=1)
        return z, emb, gate_w

    def forward_episode(
        self,
        support: Dict[str, torch.Tensor],
        query: Dict[str, torch.Tensor],
        class_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        support/query: dict of modalities, each shaped by episode grouping:
          time: (k,n,L) or (k,n,1,L)
          fft : (k,n,F)
          tf  : (k,n,3,H,W)
        query:
          time: (k,q,L)...
        class_ids: optional (k,) mapping episodic class index -> global class id
                  用于持续学习 PrototypeMemory（如果你有全局类id）
        """
        # infer k,n,q from one modality
        any_key = next(iter(support.keys()))
        s_shape = support[any_key].shape
        if len(s_shape) < 3:
            raise ValueError("support must be grouped as (k,n,...)")
        k, n = s_shape[0], s_shape[1]
        q = query[any_key].shape[1]

        # flatten & encode
        xs = {}
        xq = {}
        if "time" in support:
            xs["time"] = _ensure_4d_1d(support["time"])
            xq["time"] = _ensure_4d_1d(query["time"].reshape(k, q, -1) if query["time"].dim()==3 else query["time"])
            # if query["time"] is (k,q,1,L), _ensure_4d_1d works directly
            if query["time"].dim() == 4:
                xq["time"] = _ensure_4d_1d(query["time"])
        if "fft" in support:
            xs["fft"] = _ensure_vec(support["fft"])
            xq["fft"] = _ensure_vec(query["fft"].reshape(k*q, -1) if query["fft"].dim()==3 else query["fft"])
        if "tf" in support:
            xs["tf"] = _ensure_img(support["tf"])
            xq["tf"] = _ensure_img(query["tf"])

        z_s, emb_s, gate_s = self.encode(xs)  # (k*n,D)
        z_q, emb_q, gate_q = self.encode(xq)  # (k*q,D)

        protos = compute_prototypes(z_s, k=k, n=n)  # (k,D)
        logits = neg_sq_euclid_logits(z_q, protos)  # (k*q,k)

        y_q = episodic_labels(k, q, device=z_q.device)  # episodic labels

        return {
            "k": torch.tensor(k, device=z_q.device),
            "n": torch.tensor(n, device=z_q.device),
            "q": torch.tensor(q, device=z_q.device),
            "z_s": z_s,
            "z_q": z_q,
            "protos": protos,
            "logits": logits,
            "y_q": y_q,
            "gate_s": gate_s,
            "gate_q": gate_q,
            "class_ids": class_ids if class_ids is not None else None,
            "proj_s": F.normalize(self.proj(z_s), dim=1),
            "proj_q": F.normalize(self.proj(z_q), dim=1),
        }
