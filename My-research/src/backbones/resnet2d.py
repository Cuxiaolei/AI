# -*- coding: utf-8 -*-
"""Reusable 2D ResNet backbones for time-frequency inputs."""

from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn


class Backbone2DBase(nn.Module):
    """Unified interface for 2D backbones."""

    def __init__(self) -> None:
        super().__init__()
        self.out_dim: Optional[int] = None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, return_feature_maps: bool = False):
        raise NotImplementedError



def _import_torchvision_models():
    try:
        import torchvision.models as tv_models
        from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "torchvision is required for 2D backbones, but importing torchvision.models failed. "
            "Please check your PyTorch / torchvision installation compatibility."
        ) from exc
    return tv_models, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights



def _get_weights(name: str, pretrained: bool):
    if not pretrained:
        return None
    _, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights = _import_torchvision_models()
    if name == "resnet18":
        return ResNet18_Weights.DEFAULT
    if name == "resnet34":
        return ResNet34_Weights.DEFAULT
    if name == "resnet50":
        return ResNet50_Weights.DEFAULT
    raise ValueError(f"Unsupported 2D backbone name: {name}")



def _build_2d_resnet(name: str, weights):
    tv_models, _, _, _ = _import_torchvision_models()
    if name == "resnet18":
        return tv_models.resnet18(weights=weights)
    if name == "resnet34":
        return tv_models.resnet34(weights=weights)
    if name == "resnet50":
        return tv_models.resnet50(weights=weights)
    raise ValueError(f"Unsupported 2D backbone name: {name}")


class ResNet2D(Backbone2DBase):
    """
    Generic 2D ResNet backbone.

    Input:
        [B, C, H, W]
    Output:
        [B, out_dim]
    """

    def __init__(
        self,
        name: str = "resnet18",
        in_channels: int = 1,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        weights = _get_weights(name, pretrained)
        model = _build_2d_resnet(name, weights)

        if in_channels != 3:
            self._adapt_input_conv(model, in_channels)

        self.stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.pool = model.avgpool
        self.out_dim = model.fc.in_features

    @staticmethod
    def _adapt_input_conv(model: nn.Module, in_channels: int) -> None:
        old_conv = model.conv1
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            if old_conv.weight.shape[1] == 3 and in_channels == 1:
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            elif old_conv.weight.shape[1] == 3 and in_channels > 3:
                repeat_times = (in_channels + 2) // 3
                w = old_conv.weight.repeat(1, repeat_times, 1, 1)[:, :in_channels, :, :]
                w = w / repeat_times
                new_conv.weight.copy_(w)
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        model.conv1 = new_conv

    def forward_stages(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return {
            "layer1": f1,
            "layer2": f2,
            "layer3": f3,
            "layer4": f4,
        }

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_stages(x)
        return self.pool(feats["layer4"]).flatten(1)

    def forward(self, x: torch.Tensor, return_feature_maps: bool = False):
        feats = self.forward_stages(x)
        pooled = self.pool(feats["layer4"]).flatten(1)
        if return_feature_maps:
            feats["pooled"] = pooled
            return feats
        return pooled
