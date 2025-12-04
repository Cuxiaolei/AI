#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2D-ResNet主训练脚本
"""

import yaml
import torch
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
from models.cnn_backbone import CNNBackbone
from models.domain_aligner import CoralAligner
from utils.data_loader import EpisodeDataset
from utils.metrics_deep import DeepMetrics


class FSDGDeepPipeline:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"设备: {self.device}")

        # 初始化组件
        self.dataset = EpisodeDataset(self.config['DATA']['path'], self.config)
        self.backbone = CNNBackbone(
            feature_dim=self.config['MODEL']['feature_dim'],
            target_size=self.config['DATA']['target_size'],
            pretrained=self.config['MODEL']['pretrained']
        ).to(self.device)

        self.prototype_head = PrototypeNetwork2D(
            feature_dim=self.config['MODEL']['feature_dim']
        ).to(self.device)

        self.coral_aligner = CoralAligner(
            lambda_coral=self.config['TRAIN']['lambda_coral']
        ).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(
            list(self.backbone.parameters()) +
            list(self.prototype_head.parameters()) +
            list(self.coral_aligner.parameters()),
            lr=self.config['TRAIN']['learning_rate'],
            weight_decay=self.config['TRAIN']['weight_decay']
        )

        self.metrics = DeepMetrics()
        os.makedirs(self.config['OUTPUT']['result_dir'], exist_ok=True)

    def train_episode(self, support_set, query_set, target_set):
        """训练单个episode"""
        support_x = support_set['x'].to(self.device)
        support_y = support_set['y'].to(self.device)
        query_x = query_set['x'].to(self.device)
        target_x = target_set.to(self.device)

        # 前向传播
        support_feat = self.backbone(support_x)
        query_feat = self.backbone(query_x)
        target_feat = self.backbone(target_x)

        # 域对齐
        support_aligned, coral_loss = self.coral_aligner(support_feat, target_feat)
        query_aligned, _ = self.coral_aligner(query_feat, target_feat)

        # 原型对比
        logits, proto_loss = self.prototype_head(support_aligned, support_y)

        # 分类损失
        cls_loss = F.cross_entropy(logits, support_y)

        # 总损失
        total_loss = cls_loss + proto_loss + coral_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'cls_loss': cls_loss.item(),
            'proto_loss': proto_loss.item(),
            'coral_loss': coral_loss.item(),
            'total_loss': total_loss.item()
        }

    def evaluate_episode(self, support_set, query_set):
        """评估episode"""
        with torch.no_grad():
            support_x = support_set['x'].to(self.device)
            support_y = support_set['y'].to(self.device)
            query_x = query_set['x'].to(self.device)
            query_y = query_set['y'].to(self.device)

            support_feat = self.backbone(support_x)
            query_feat = self.backbone(query_x)

            # 计算原型
            prototypes = []
            for c in range(3):
                mask = (support_y == c)
                if mask.sum() > 0:
                    proto = support_feat[mask].mean(dim=0)
                else:
                    proto = torch.zeros_like(support_feat[0])
                prototypes.append(proto)
            prototypes = torch.stack(prototypes)

            # 最近原型分类
            distances = torch.cdist(query_feat, prototypes)
            preds = distances.argmin(dim=1).cpu().numpy()

            acc = (preds == query_y.cpu().numpy()).mean()
            return acc

    def train_and_validate(self):
        """主训练循环"""
        n_domains = 12
        results = []

        for target_idx in tqdm(range(n_domains), desc="域验证"):
            source_domains = [i for i in range(n_domains) if i != target_idx]
            episode_accs = []

            for episode in range(self.config['FEW_SHOT']['n_episodes']):
                support_set, query_set, target_set = self.dataset.generate_episode(
                    source_domains, target_idx,
                    self.config['FEW_SHOT']['k_shot'],
                    self.config['FEW_SHOT']['n_query']
                )

                loss_dict = self.train_episode(support_set, query_set, target_set)
                acc = self.evaluate_episode(support_set, query_set)

                episode_accs.append(acc)

                if episode % 10 == 0:
                    print(f"域{target_idx} Episode{episode}: Acc={acc:.3f}, Loss={loss_dict['total_loss']:.4f}")

            mean_acc = np.mean(episode_accs)
            std_acc = np.std(episode_accs)
            results.append({
                'target_domain': target_idx,
                'mean_acc': mean_acc,
                'std_acc': std_acc
            })

            print(f"域{target_idx}完成: {mean_acc:.4f} ± {std_acc:.4f}")

        return results

    def save_model(self, path):
        torch.save({
            'backbone': self.backbone.state_dict(),
            'prototype_head': self.prototype_head.state_dict(),
            'coral_aligner': self.coral_aligner.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        print(f"✅ 模型已保存: {path}")


def main():
    config_path = './configs/config_resnet2d.yaml'

    if not os.path.exists('./data/Ottawa_Bearing_Dataset'):
        print("❌ 数据路径不存在")
        return

    pipeline = FSDGDeepPipeline(config_path)
    results = pipeline.train_and_validate()

    # 保存结果
    pipeline.save_model('./results/resnet2d_model.pt')

    # 打印总体结果
    overall_mean = np.mean([r['mean_acc'] for r in results])
    overall_std = np.mean([r['std_acc'] for r in results])

    print("\n" + "=" * 60)
    print("📊 总体性能")
    print("=" * 60)
    print(f"平均准确率: {overall_mean:.4f} ± {overall_std:.4f}")

    # 保存详细报告
    report_path = './results/detailed_results.txt'
    with open(report_path, 'w') as f:
        f.write(f"平均准确率: {overall_mean:.4f} ± {overall_std:.4f}\n\n")
        for r in results:
            domain_info = pipeline.dataset.domain_map[r['target_domain']]
            f.write(
                f"域{r['target_domain']} ({domain_info['health']}-{domain_info['speed']}): {r['mean_acc']:.4f} ± {r['std_acc']:.4f}\n")

    print(f"\n✅ 详细报告已保存: {report_path}")


if __name__ == '__main__':
    main()