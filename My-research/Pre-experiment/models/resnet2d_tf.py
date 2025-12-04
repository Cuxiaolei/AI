#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2D-ResNet-18 + 时频转换器（动态原型版）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pywt
import numpy as np


class TimeFrequencyConverter:
    """时频图转换器（用小波包变换）"""

    def __init__(self, wavelet='db4', mode='symmetric', target_size=128):
        self.wavelet = wavelet
        self.mode = mode
        self.target_size = target_size

    def convert(self, signal, level=5):
        """信号转时频图
        Args:
            signal: 振动信号，shape [batch, window_size] 或 [window_size]
            level: 小波包分解层数
        Returns:
            tf_images: 时频图，shape [batch, 1, target_size, target_size]
        """
        if isinstance(signal, torch.Tensor):
            signal = signal.cpu().numpy()

        if signal.ndim == 1:
            signal = signal[None, :]

        batch_size = signal.shape[0]
        tf_images = []

        for i in range(batch_size):
            sig = signal[i]
            wp = pywt.WaveletPacket(sig, self.wavelet, maxlevel=level)
            nodes = wp.get_level(level, order='natural')

            # 提取节点能量
            energies = [np.sum(np.array(node.data) ** 2) for node in nodes]

            # 构建方阵
            n_nodes = len(energies)
            size = int(np.ceil(np.sqrt(n_nodes)))  # 向上取整
            square_len = size * size

            if len(energies) < square_len:
                energies.extend([0] * (square_len - len(energies)))

            tf_2d = np.array(energies[:square_len]).reshape(size, size)

            # 双线性插值到目标尺寸
            tf_tensor = torch.from_numpy(tf_2d).float().unsqueeze(0).unsqueeze(0)
            tf_resized = F.interpolate(tf_tensor, size=(self.target_size, self.target_size),
                                       mode='bilinear', align_corners=False)

            # 归一化
            tf_resized = (tf_resized - tf_resized.min()) / (tf_resized.max() - tf_resized.min() + 1e-8)
            tf_images.append(tf_resized.squeeze(0))

        return torch.stack(tf_images)


class ResNet2DAdapter(nn.Module):
    """2D-ResNet-18适配器"""

    def __init__(self, pretrained=True, feature_dim=128, target_size=128):
        super(ResNet2DAdapter, self).__init__()
        self.tf_converter = TimeFrequencyConverter(target_size=target_size)

        # 加载ResNet-18
        self.resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)

        # 修改输入层: 3通道 -> 1通道
        original_weight = self.resnet.conv1.weight.data
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if pretrained:
            # 将RGB权重平均到单通道
            self.resnet.conv1.weight.data = original_weight.mean(dim=1, keepdim=True)

        # 修改输出层: 1000类 -> feature_dim
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, feature_dim)
        nn.init.kaiming_normal_(self.resnet.fc.weight)
        nn.init.constant_(self.resnet.fc.bias, 0)

    def forward(self, x):
        """前向传播
        Args:
            x: 原始振动信号 [batch, window_size]
        Returns:
            features: 特征向量 [batch, feature_dim]
        """
        tf_image = self.tf_converter.convert(x).to(x.device)
        return self.resnet(tf_image)

    def forward_with_tf(self, tf_image):
        """直接输入时频图"""
        return self.resnet(tf_image)


class PrototypeNetwork2D(nn.Module):
    """动态原型网络：从支持集计算原型，而非可学习参数"""

    def __init__(self, feature_dim=128, temperature=0.07):
        super(PrototypeNetwork2D, self).__init__()
        self.temperature = temperature

    def compute_prototypes(self, features, labels, n_classes=3):
        """动态计算原型
        Args:
            features: 特征向量 [N, feature_dim]
            labels: 标签 [N]
            n_classes: 类别数
        Returns:
            prototypes: 原型向量 [n_classes, feature_dim]
        """
        features = F.normalize(features, dim=1)
        prototypes = []

        for c in range(n_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                # 该类别的均值原型
                proto = features[mask].mean(dim=0)
            else:
                # 无样本时为零向量
                proto = torch.zeros_like(features[0])
            prototypes.append(proto)

        return torch.stack(prototypes)

    def forward(self, query_features, prototypes):
        """基于原型的分类
        Args:
            query_features: 查询集特征 [N, feature_dim]
            prototypes: 原型 [n_classes, feature_dim]
        Returns:
            logits: 相似度得分 [N, n_classes]
        """
        query_features = F.normalize(query_features, dim=1)
        prototypes = F.normalize(prototypes, dim=1)

        # 余弦相似度
        logits = torch.mm(query_features, prototypes.t()) / self.temperature
        return logits