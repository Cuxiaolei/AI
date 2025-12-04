#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2D-ResNet主训练脚本（持续学习版）
"""

import yaml
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
from models.cnn_backbone import CNNBackbone
from models.domain_aligner import CoralAligner, EWCRegularizer
from models.resnet2d_tf import PrototypeNetwork2D
from utils.data_loader import EpisodeDataset
from utils.metrics_deep import DeepMetrics, ContinualMetrics


class FSDGContinualPipeline:
    """小样本域泛化持续学习流水线"""

    def __init__(self, config_path):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  设备: {self.device}")

        # 初始化组件
        self.dataset = EpisodeDataset(self.config['DATA']['path'], self.config)
        self.backbone = CNNBackbone(
            feature_dim=self.config['MODEL']['feature_dim'],
            target_size=self.config['DATA']['target_size'],
            pretrained=self.config['MODEL']['pretrained']
        ).to(self.device)

        self.prototype_head = PrototypeNetwork2D(
            feature_dim=self.config['MODEL']['feature_dim'],
            temperature=self.config['TRAIN']['temperature']
        ).to(self.device)

        self.coral_aligner = CoralAligner(
            lambda_coral=self.config['TRAIN']['lambda_coral']
        ).to(self.device)

        # 持续学习组件
        self.continual_method = self.config['CONTINUAL']['method']
        if self.continual_method == 'ewc':
            self.ewc_reg = EWCRegularizer(
                self.backbone,
                lambda_ewc=self.config['CONTINUAL']['lambda_ewc']
            )

        # 记忆缓冲区（用于经验回放）
        self.memory_buffer = {}  # {domain_idx: {'x': ..., 'y': ...}}
        self.memory_per_domain = self.config['CONTINUAL']['memory_per_domain']

        # 优化器
        self.optimizer = optim.Adam(
            list(self.backbone.parameters()) +
            list(self.prototype_head.parameters()),
            lr=self.config['TRAIN']['learning_rate'],
            weight_decay=self.config['TRAIN']['weight_decay']
        )

        self.metrics = DeepMetrics()
        self.continual_metrics = ContinualMetrics()

        # 创建输出目录
        os.makedirs(self.config['OUTPUT']['result_dir'], exist_ok=True)
        os.makedirs(self.config['OUTPUT']['log_dir'], exist_ok=True)

    def update_memory(self, domain_idx, support_set):
        """更新经验回放记忆缓冲区
        Args:
            domain_idx: 域索引
            support_set: 支持集数据
        """
        x = support_set['x']
        y = support_set['y']

        # 随机采样存储
        if len(x) > self.memory_per_domain:
            indices = np.random.choice(len(x), self.memory_per_domain, replace=False)
            self.memory_buffer[domain_idx] = {
                'x': x[indices].cpu(),
                'y': y[indices].cpu()
            }
        else:
            self.memory_buffer[domain_idx] = {
                'x': x.cpu(),
                'y': y.cpu()
            }

        print(f"💾 已更新域{domain_idx}的记忆缓冲区 (样本数: {len(self.memory_buffer[domain_idx]['x'])})")

    def sample_from_memory(self):
        """从记忆缓冲区采样用于回放"""
        if not self.memory_buffer:
            return None

        all_x, all_y = [], []
        for domain_data in self.memory_buffer.values():
            all_x.append(domain_data['x'])
            all_y.append(domain_data['y'])

        return {
            'x': torch.cat(all_x).to(self.device),
            'y': torch.cat(all_y).to(self.device)
        }

    def train_episode(self, support_set, query_set, target_set, domain_idx):
        """训练单个Episode"""
        # 数据移至设备
        support_x = support_set['x'].to(self.device)
        support_y = support_set['y'].to(self.device)
        query_x = query_set['x'].to(self.device)
        query_y = query_set['y'].to(self.device)
        target_x = target_set.to(self.device)

        # 1. 前向传播
        support_feat = self.backbone(support_x)
        query_feat = self.backbone(query_x)
        target_feat = self.backbone(target_x)

        # 2. 域对齐（CORAL损失）
        support_aligned, coral_loss = self.coral_aligner(support_feat, target_feat)
        query_aligned, _ = self.coral_aligner(query_feat, target_feat)

        # 3. 计算原型（动态）
        prototypes = self.prototype_head.compute_prototypes(support_aligned, support_y)

        # 4. 原型对比损失（在查询集上计算）
        proto_loss = self.metrics.prototype_loss(query_aligned, query_y, prototypes)

        # 5. 分类损失（最近原型分类）
        logits = self.prototype_head(query_aligned, prototypes)
        cls_loss = F.cross_entropy(logits, query_y)

        # 6. 总损失
        total_loss = cls_loss + proto_loss * self.config['TRAIN']['lambda_proto'] + coral_loss

        # 7. 经验回放损失
        if self.continual_method == 'replay':
            replay_data = self.sample_from_memory()
            if replay_data is not None:
                replay_feat = self.backbone(replay_data['x'])
                replay_protos = self.prototype_head.compute_prototypes(
                    replay_feat, replay_data['y']
                )
                replay_logits = self.prototype_head(replay_feat, replay_protos)
                replay_loss = F.cross_entropy(replay_logits, replay_data['y'])
                total_loss += replay_loss * 0.3

        # 8. EWC正则化
        if self.continual_method == 'ewc':
            ewc_loss = self.ewc_reg.penalty()
            total_loss += ewc_loss

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

    def evaluate_on_domain(self, domain_idx, support_memory):
        """在指定域上评估
        Args:
            domain_idx: 目标域索引
            support_memory: 支持集（来自记忆库）
        """
        # 加载域数据
        domain_x, domain_y = self.dataset.load_domain(domain_idx)

        if len(domain_x) == 0:
            return 0.0

        # 采样测试数据
        n_test = min(100, len(domain_x))
        test_indices = np.random.choice(len(domain_x), n_test, replace=False)
        test_x = domain_x[test_indices].to(self.device)
        test_y = domain_y[test_indices].to(self.device)

        # 特征提取
        with torch.no_grad():
            support_feat = self.backbone(support_memory['x'].to(self.device))
            test_feat = self.backbone(test_x)

            # 计算原型
            prototypes = self.prototype_head.compute_prototypes(
                support_feat, support_memory['y'].to(self.device)
            )

            # 分类
            logits = self.prototype_head(test_feat, prototypes)
            acc = self.metrics.accuracy(logits, test_y)

        return acc

    def train_and_validate(self):
        """持续学习主循环：顺序学习12个域"""
        n_domains = 12
        results = []  # 存储每个域学习后的性能

        # 初始模型在未见过域上的性能（前向迁移基线）
        print("📊 计算初始前向迁移基线...")
        initial_performance = {}
        for target_idx in tqdm(range(n_domains), desc="初始评估"):
            # 用前3个域作为临时支持集
            temp_support, _, _ = self.dataset.generate_fsdg_episode(
                source_domains=[0, 1, 2], target_domain=target_idx,
                k_shot=self.config['FEW_SHOT']['k_shot'],
                n_query=self.config['FEW_SHOT']['n_query']
            )
            acc = self.evaluate_on_domain(target_idx, temp_support)
            initial_performance[f'domain_{target_idx}'] = acc

        results.append(initial_performance)
        print(f"初始平均准确率: {np.mean(list(initial_performance.values())):.4f}")

        # 顺序学习每个域
        print("\n🔄 开始持续学习...")
        for target_idx in tqdm(range(n_domains), desc="持续学习"):
            print(f"\n{'=' * 60}")
            print(f"学习域 {target_idx}: {self.dataset.domain_map[target_idx]}")
            print(f"{'=' * 60}")

            # 源域：除了目标域外的所有域
            source_domains = [i for i in range(n_domains) if i != target_idx]

            # 生成k-shot支持集
            support_set, query_set, target_set = self.dataset.generate_fsdg_episode(
                source_domains, target_idx,
                k_shot=self.config['FEW_SHOT']['k_shot'],
                n_query=self.config['FEW_SHOT']['n_query']
            )

            # 更新记忆缓冲区
            self.update_memory(target_idx, support_set)

            # 训练N个episode
            episode_accs = []
            for episode in range(self.config['FEW_SHOT']['n_episodes']):
                loss_dict = self.train_episode(
                    support_set, query_set, target_set, target_idx
                )

                # 每10个episode评估一次
                if episode % 10 == 0:
                    acc = self.evaluate_episode(support_set, query_set)
                    episode_accs.append(acc)
                    print(f"Episode {episode:3d}: Acc={acc:.3f}, Loss={loss_dict['total_loss']:.4f}")

            # 更新EWC（如果使用）
            if self.continual_method == 'ewc':
                self.ewc_reg.update_fisher(support_set, self.backbone)

            # 评估在所有已见域上的性能
            domain_performance = {}
            for eval_domain in range(target_idx + 1):
                memory_support = {
                    'x': torch.cat([self.memory_buffer[d]['x'] for d in range(target_idx + 1)]),
                    'y': torch.cat([self.memory_buffer[d]['y'] for d in range(target_idx + 1)])
                }
                acc = self.evaluate_on_domain(eval_domain, memory_support)
                domain_performance[f'domain_{eval_domain}'] = acc

            results.append(domain_performance)

            # 打印当前性能
            mean_acc = np.mean(list(domain_performance.values()))
            print(f"📈 域{target_idx}学习后平均准确率: {mean_acc:.4f}")

            # 保存中间结果
            self.save_checkpoint(target_idx)

        return results


def evaluate_episode(self, support_set, query_set):
    """评估单个episode"""
    with torch.no_grad():
        support_x = support_set['x'].to(self.device)
        support_y = support_set['y'].to(self.device)
        query_x = query_set['x'].to(self.device)
        query_y = query_set['y'].to(self.device)

        support_feat = self.backbone(support_x)
        query_feat = self.backbone(query_x)

        prototypes = self.prototype_head.compute_prototypes(support_feat, support_y)
        logits = self.prototype_head(query_feat, prototypes)

        acc = self.metrics.accuracy(logits, query_y)
        return acc


def save_checkpoint(self, domain_idx):
    """保存检查点"""
    checkpoint_path = f"{self.config['OUTPUT']['model_path']}_domain_{domain_idx}.pt"
    torch.save({
        'backbone': self.backbone.state_dict(),
        'prototype_head': self.prototype_head.state_dict(),
        'memory_buffer': self.memory_buffer,
        'optimizer': self.optimizer.state_dict(),
        'config': self.config
    }, checkpoint_path)
    print(f"💾 检查点已保存: {checkpoint_path}")


def save_final_model(self):
    """保存最终模型"""
    torch.save({
        'backbone': self.backbone.state_dict(),
        'prototype_head': self.prototype_head.state_dict(),
        'memory_buffer': self.memory_buffer,
        'optimizer': self.optimizer.state_dict(),
        'config': self.config
    }, self.config['OUTPUT']['model_path'])
    print(f"✅ 最终模型已保存: {self.config['OUTPUT']['model_path']}")


def main():
    config_path = './configs/config_resnet2d.yaml'

    # 检查数据路径
    data_path = './data/Ottawa_Bearing_Dataset'
    if not os.path.exists(data_path):
        print(f"❌ 数据路径不存在: {data_path}")
        print("请下载渥太华轴承数据集并放入该目录")
        print("数据集地址: https://dataverse.nl/dataset.xhtml?persistentId=doi:10.5683/SP2/4U9P7O")
        return

    # 检查数据文件
    required_files = [f"{h}-{s}-{t}.mat" for h in ['H', 'I', 'O']
                      for s in ['A', 'B', 'C', 'D'] for t in [1, 2, 3]]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(data_path, f))]

    if missing_files:
        print(f"⚠️  缺少 {len(missing_files)} 个数据文件")
        print("请确保数据集完整")

    # 启动训练
    print("🚀 启动小样本域泛化持续学习训练...")
    pipeline = FSDGContinualPipeline(config_path)
    results = pipeline.train_and_validate()

    # 保存最终模型
    pipeline.save_final_model()

    # 计算持续学习指标
    print("\n" + "=" * 60)
    print("📊 持续学习性能报告")
    print("=" * 60)

    avg_acc = pipeline.continual_metrics.average_accuracy(results)
    bwt = pipeline.continual_metrics.backward_transfer(results)
    fwt = pipeline.continual_metrics.forward_transfer(results)
    forgetting = pipeline.continual_metrics.forgetting_measure(results)

    print(f"平均准确率: {avg_acc:.4f}")
    print(f"前向迁移 (FWD): {fwt:.4f}")
    print(f"后向迁移 (BWT): {bwt:.4f}")
    print(f"遗忘程度 (FORGET): {forgetting:.4f}")

    # 保存详细报告
    report_path = './results/continual_learning_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("持续学习详细报告\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"配置: {config_path}\n\n")
        f.write(f"平均准确率: {avg_acc:.4f}\n")
        f.write(f"前向迁移 (FWD): {fwt:.4f}\n")
        f.write(f"后向迁移 (BWT): {bwt:.4f}\n")
        f.write(f"遗忘程度 (FORGET): {forgetting:.4f}\n\n")

        f.write("各域性能:\n")
        for idx, r in enumerate(results):
            if idx == 0:
                f.write(f"初始性能: {np.mean(list(r.values())):.4f}\n")
            else:
                domain_info = pipeline.dataset.domain_map[idx - 1]
                f.write(f"学习域{idx - 1} ({domain_info['health']}-{domain_info['speed']})后: ")
                f.write(f"{np.mean(list(r.values())):.4f}\n")

    print(f"\n✅ 详细报告已保存: {report_path}")


if __name__ == '__main__':
    main()