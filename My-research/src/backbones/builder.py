# -*- coding: utf-8 -*-
"""Factory for creating reusable backbone networks."""

from __future__ import annotations
from typing import Union
from .resnet1d import Backbone1DBase, resnet1d18, resnet1d34, resnet1d50
from .resnet2d import Backbone2DBase, ResNet2D


BackboneBase = Union[Backbone1DBase, Backbone2DBase]



def build_backbone(
    name: str,
    in_channels: int = 1,
    pretrained: bool = False,
    **kwargs,
) -> BackboneBase:
    """
    Build one backbone by name.

    Supported names:
        - resnet1d18
        - resnet1d34
        - resnet1d50
        - resnet18
        - resnet34
        - resnet50
    """
    name = name.lower()

    if name == "resnet1d18":
        if pretrained:
            raise ValueError("resnet1d18 does not support ImageNet pretrained weights.")
        return resnet1d18(
            in_channels=in_channels,
            base_channels=kwargs.get("base_channels", 64),
            norm=kwargs.get("norm", "bn"),
            act=kwargs.get("act", "elu"),
        )

    if name == "resnet1d34":
        if pretrained:
            raise ValueError("resnet1d34 does not support ImageNet pretrained weights.")
        return resnet1d34(
            in_channels=in_channels,
            base_channels=kwargs.get("base_channels", 64),
            norm=kwargs.get("norm", "bn"),
            act=kwargs.get("act", "elu"),
        )

    if name == "resnet1d50":
        if pretrained:
            raise ValueError("resnet1d50 does not support ImageNet pretrained weights.")
        return resnet1d50(
            in_channels=in_channels,
            base_channels=kwargs.get("base_channels", 64),
            norm=kwargs.get("norm", "bn"),
            act=kwargs.get("act", "elu"),
        )

    if name in {"resnet18", "resnet34", "resnet50"}:
        return ResNet2D(name=name, in_channels=in_channels, pretrained=pretrained)

    raise ValueError(f"Unsupported backbone name: {name}")
