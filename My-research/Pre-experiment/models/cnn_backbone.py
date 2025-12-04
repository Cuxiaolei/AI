#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNN主入口：2D-ResNet版本
"""

from .resnet2d_tf import ResNet2DAdapter, PrototypeNetwork2D
import torch
import torch.nn as nn

class CNNBackbone(nn.Module):
    """统一接口：2D-ResNet"""

    def __init__(self, feature_dim=128, target_size=64, pretrained=True):
        super().__init__()
        self.backbone = ResNet2DAdapter(
            pretrained=pretrained,
            feature_dim=feature_dim,
            target_size=target_size
        )
        self.feature_dim = feature_dim

    def forward(self, x):
        """
        x: [batch, window_size] 原始振动信号
        """
        return self.backbone(x)

    def forward_with_tf(self, tf_image):
        """
        直接输入时频图（用于调试）
        tf_image: [batch, 1, H, W]
        """
        return self.backbone.forward_with_tf(tf_image)