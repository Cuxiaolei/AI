# -*- coding: utf-8 -*-
"""Reusable 1D ResNet backbones for frequency-domain inputs."""

from __future__ import annotations
from typing import Dict, Optional, Type
import torch
import torch.nn as nn


def get_activation(name: str = "relu", inplace: bool = True) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name == "elu":
        return nn.ELU(inplace=inplace)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def get_norm_1d(name: str, num_features: int) -> nn.Module:
    name = name.lower()
    if name == "bn":
        return nn.BatchNorm1d(num_features)
    if name == "in":
        return nn.InstanceNorm1d(num_features, affine=True)
    if name == "gn":
        num_groups = 8 if num_features % 8 == 0 else 4
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
    raise ValueError(f"Unsupported 1D norm: {name}")


def conv3x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv1d:
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv1d:
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class Backbone1DBase(nn.Module):
    """Unified interface for 1D backbones."""

    def __init__(self) -> None:
        super().__init__()
        self.out_dim: Optional[int] = None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, return_feature_maps: bool = False):
        raise NotImplementedError


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm: str = "bn",
        act: str = "elu",
    ) -> None:
        super().__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = get_norm_1d(norm, planes)
        self.act = get_activation(act)
        self.conv2 = conv3x1(planes, planes, 1)
        self.bn2 = get_norm_1d(norm, planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.act(out)
        return out


class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm: str = "bn",
        act: str = "elu",
    ) -> None:
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = get_norm_1d(norm, planes)
        self.conv2 = conv3x1(planes, planes, stride)
        self.bn2 = get_norm_1d(norm, planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = get_norm_1d(norm, planes * self.expansion)
        self.act = get_activation(act)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.act(out)
        return out


class ResNet1D(Backbone1DBase):
    """
    Generic 1D ResNet backbone.

    Input:
        [B, C, L]
    Output:
        [B, out_dim]
    """

    def __init__(
        self,
        block: Type[nn.Module],
        layers: list[int],
        in_channels: int = 1,
        base_channels: int = 64,
        norm: str = "bn",
        act: str = "elu",
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__()
        self.inplanes = base_channels
        self.norm = norm
        self.act_name = act

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            get_norm_1d(norm, base_channels),
            get_activation(act),
            nn.Conv1d(base_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            get_norm_1d(norm, base_channels),
            get_activation(act),
            nn.Conv1d(base_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            get_norm_1d(norm, base_channels),
            get_activation(act),
        )

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = base_channels * 8 * block.expansion
        self._init_weights(zero_init_residual=zero_init_residual)

    def _make_layer(self, block: Type[nn.Module], planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                get_norm_1d(self.norm, planes * block.expansion),
            )

        layers = [
            block(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                norm=self.norm,
                act=self.act_name,
            )
        ]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes=self.inplanes,
                    planes=planes,
                    stride=1,
                    downsample=None,
                    norm=self.norm,
                    act=self.act_name,
                )
            )

        return nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual: bool = False) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.InstanceNorm1d, nn.GroupNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, Bottleneck1D):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward_stages(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.maxpool(x)
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


# ---------- Builders ----------

def resnet1d18(
    in_channels: int = 1,
    base_channels: int = 64,
    norm: str = "bn",
    act: str = "elu",
) -> ResNet1D:
    return ResNet1D(
        block=BasicBlock1D,
        layers=[2, 2, 2, 2],
        in_channels=in_channels,
        base_channels=base_channels,
        norm=norm,
        act=act,
    )


def resnet1d34(
    in_channels: int = 1,
    base_channels: int = 64,
    norm: str = "bn",
    act: str = "elu",
) -> ResNet1D:
    return ResNet1D(
        block=BasicBlock1D,
        layers=[3, 4, 6, 3],
        in_channels=in_channels,
        base_channels=base_channels,
        norm=norm,
        act=act,
    )


def resnet1d50(
    in_channels: int = 1,
    base_channels: int = 64,
    norm: str = "bn",
    act: str = "elu",
) -> ResNet1D:
    return ResNet1D(
        block=Bottleneck1D,
        layers=[3, 4, 6, 3],
        in_channels=in_channels,
        base_channels=base_channels,
        norm=norm,
        act=act,
    )
