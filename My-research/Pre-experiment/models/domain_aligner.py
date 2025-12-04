#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CORAL域对齐模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralAligner(nn.Module):
    """CORAL对齐器：不修改特征，仅计算对齐损失"""

    def __init__(self, lambda_coral=0.3):
        super(CoralAligner, self).__init__()
        self.lambda_coral = lambda_coral

    def forward(self, source_feat, target_feat):
        """前向传播
        Args:
            source_feat: 源域特征 [N, D]
            target_feat: 目标域特征 [M, D]
        Returns:
            source_feat: 保持不变的源域特征
            coral_loss: CORAL对齐损失
        """
        coral_loss = self.compute_coral_loss(source_feat, target_feat)
        return source_feat, coral_loss * self.lambda_coral

    def compute_coral_loss(self, source, target):
        """计算CORAL损失（协方差矩阵的Frobenius范数）

        Args:
            source: 源域特征 [N, D]
            target: 目标域特征 [M, D]
        Returns:
            loss: CORAL损失值
        """
        d = source.size(1)

        # 减去均值
        source_centered = source - source.mean(dim=0, keepdim=True)
        target_centered = target - target.mean(dim=0, keepdim=True)

        # 计算协方差矩阵
        source_cov = (source_centered.T @ source_centered) / (source.size(0) - 1)
        target_cov = (target_centered.T @ target_centered) / (target.size(0) - 1)

        # Frobenius范数
        loss = torch.norm(source_cov - target_cov, p='fro')
        return loss


class EWCRegularizer(nn.Module):
    """弹性权重巩固（EWC）持续学习正则化"""

    def __init__(self, model, lambda_ewc=0.4):
        super(EWCRegularizer, self).__init__()
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_dict = {}
        self.params_dict = {}

    def update_fisher(self, support_set, backbone):
        """在域训练结束后计算Fisher信息矩阵

        Args:
            support_set: 该域的支持集数据
            backbone: 特征提取器
        """
        backbone.eval()
        support_x = support_set['x'].cuda()
        support_y = support_set['y'].cuda()

        # 计算log likelihood
        features = backbone(support_x)
        logits = torch.mm(features, features.T)  # 简化的似然计算
        log_likelihood = F.log_softmax(logits, dim=1).mean()

        # 计算梯度
        grads = torch.autograd.grad(log_likelihood, self.model.parameters(),
                                    retain_graph=False, create_graph=False)

        # 存储Fisher信息和旧参数
        for (name, param), grad in zip(self.model.named_parameters(), grads):
            self.fisher_dict[name] = grad.detach().clone() ** 2 + 1e-6
            self.params_dict[name] = param.detach().clone()

        backbone.train()

    def penalty(self):
        """计算EWC惩罚项"""
        loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher_dict:
                fisher = self.fisher_dict[name]
                param_old = self.params_dict[name]
                loss += (fisher * (param - param_old) ** 2).sum()
        return 0.5 * self.lambda_ewc * loss