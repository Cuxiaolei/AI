#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
深度学习评估指标
"""

import torch
import torch.nn.functional as F
import numpy as np


class DeepMetrics:
    @staticmethod
    def accuracy(logits, labels):
        preds = logits.argmax(dim=1)
        return (preds == labels).float().mean().item()

    @staticmethod
    def prototype_loss(features, labels, prototypes):
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(prototypes, dim=1)
        sim = torch.mm(features, prototypes.t())
        return F.cross_entropy(sim, labels)

    @staticmethod
    def coral_distance(cov_s, cov_t):
        return torch.norm(cov_s - cov_t, 'fro').item()