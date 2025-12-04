#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2D-ResNet-18 + 时频转换器
"""

import torch
import torch.nn as nn
import torchvision.models as models
import pywt
import numpy as np


class TimeFrequencyConverter:
    """时频图转换器"""

    def __init__(self, wavelet='db4', mode='symmetric', target_size=64):
        self.wavelet = wavelet
        self.mode = mode
        self.target_size = target_size

    def convert(self, signal, level=5):
        """信号转时频图"""
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

            tf_matrix = []
            for node in nodes:
                energy = np.sum(np.array(node.data) ** 2)
                tf_matrix.append(energy)

            n_nodes = len(tf_matrix)
            size = int(np.sqrt(n_nodes))
            if size * size != n_nodes:
                next_square = int(np.ceil(np.sqrt(n_nodes))) ** 2
                tf_matrix = tf_matrix + [0] * (next_square - n_nodes)
                size = int(np.sqrt(next_square))

            tf_2d = np.array(tf_matrix).reshape(size, size)
            scale = self.target_size // size
            if scale > 0:
                tf_2d = np.kron(tf_2d, np.ones((scale, scale)))
                if tf_2d.shape[0] > self.target_size:
                    tf_2d = tf_2d[:self.target_size, :self.target_size]
                else:
                    pad_h = self.target_size - tf_2d.shape[0]
                    pad_w = self.target_size - tf_2d.shape[1]
                    tf_2d = np.pad(tf_2d, ((0, pad_h), (0, pad_w)), mode='constant')

            tf_2d = (tf_2d - tf_2d.min()) / (tf_2d.max() - tf_2d.min() + 1e-8)
            tf_images.append(tf_2d)

        tf_images = np.array(tf_images)[:, None, :, :]
        return torch.FloatTensor(tf_images)


class ResNet2DAdapter(nn.Module):
    """2D-ResNet-18适配器"""

    def __init__(self, pretrained=True, feature_dim=128, target_size=64):
        super().__init__()
        self.tf_converter = TimeFrequencyConverter(target_size=target_size)
        self.resnet = models.resnet18(pretrained=pretrained)

        original_weight = self.resnet.conv1.weight.data
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if pretrained:
            self.resnet.conv1.weight.data = original_weight.mean(dim=1, keepdim=True)

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, feature_dim)
        nn.init.kaiming_normal_(self.resnet.fc.weight)

    def forward(self, x):
        tf_image = self.tf_converter.convert(x).to(x.device)
        return self.resnet(tf_image)

    def forward_with_tf(self, tf_image):
        return self.resnet(tf_image)


class PrototypeNetwork2D(nn.Module):
    """2D原型网络"""

    def __init__(self, feature_dim=128, n_classes=3):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_classes, feature_dim))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, features, support_labels):
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        sim = torch.mm(features, prototypes.t())
        loss = self.prototype_contrast_loss(features, support_labels)
        return sim, loss

    def prototype_contrast_loss(self, features, labels):
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)

        pos_proto = prototypes[labels]
        pos_dist = torch.norm(features - pos_proto, dim=1)

        neg_dists = []
        for i in range(len(labels)):
            neg_indices = [j for j in range(len(self.prototypes)) if j != labels[i]]
            neg_proto = prototypes[neg_indices]
            neg_dist = torch.min(torch.norm(features[i:i + 1] - neg_proto, dim=1))
            neg_dists.append(neg_dist)
        neg_dists = torch.stack(neg_dists)

        margin = 1.0
        loss = torch.mean(torch.relu(pos_dist - neg_dists + margin))
        return loss