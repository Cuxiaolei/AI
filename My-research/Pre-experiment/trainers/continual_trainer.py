import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import numpy as np
import copy
from collections import deque
from trainers.base_trainer import BaseTrainer
from losses.prototypical_loss import PrototypicalLoss


class ContinualLearningTrainer(BaseTrainer):
    """域增量持续学习训练器"""

    def __init__(self, model, config, device='cuda'):
        super(ContinualLearningTrainer, self).__init__(model, config, device)

        # 记忆回放缓冲区
        self.memory_size = config['training']['memory_size']
        self.rehearsal_ratio = config['training']['rehearsal_ratio']

        # 存储各域的代表性样本
        self.memory_buffer = {}  # domain -> (data, labels, features)

        # 原型损失
        self.criterion = PrototypicalLoss(
            temperature=config['training']['proto_loss_weight'],
            proto_weight=1.0,
            domain_weight=config['training'].get('domain_weight', 0.5)
        )

        # 域顺序
        self.domains = config['data']['source_domains']
        self.current_domain_idx = 0

        # 记录性能
        self.domain_performance = {}

    def train_domain(self, domain: str, train_loader: DataLoader,
                     val_loader: DataLoader, epochs: int) -> Dict:
        """训练单个域"""

        print(f"\n{'=' * 60}")
        print(f"开始训练域: {domain}")
        print(f"{'=' * 60}")

        # 创建优化器
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['lr'],
            weight_decay=self.config['training']['weight_decay']
        )

        # 学习率调度
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config['training']['step_size'],
            gamma=self.config['training']['gamma']
        )

        best_acc = 0
        best_state = None

        for epoch in range(epochs):
            # 训练阶段
            train_metrics = self._train_epoch(
                domain, train_loader, optimizer, scheduler
            )

            # 验证阶段
            if epoch % self.config['evaluation']['eval_freq'] == 0:
                val_metrics = self._evaluate(val_loader)

                print(f"Epoch [{epoch + 1}/{epochs}] - "
                      f"Train Acc: {train_metrics['accuracy']:.4f} - "
                      f"Val Acc: {val_metrics['accuracy']:.4f} - "
                      f"Loss: {train_metrics['total_loss']:.4f}")

                # 保存最佳模型
                if val_metrics['accuracy'] > best_acc:
                    best_acc = val_metrics['accuracy']
                    best_state = copy.deepcopy(self.model.state_dict())

            # 更新记忆缓冲区
            if epoch == epochs // 2:  # 中间时刻更新
                self._update_memory_buffer(domain, train_loader)

        # 恢复最佳模型
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.domain_performance[domain] = best_acc

        return {
            'domain': domain,
            'best_accuracy': best_acc,
            'final_metrics': val_metrics
        }

    def _train_epoch(self, domain: str, train_loader: DataLoader,
                     optimizer: torch.optim.Optimizer,
                     scheduler: torch.optim.lr_scheduler) -> Dict:
        """训练一个epoch"""

        self.model.train()

        total_loss = 0
        total_proto_loss = 0
        total_domain_loss = 0
        correct = 0
        total = 0

        # 如果启用了记忆回放，合并数据
        if self.memory_buffer and self.rehearsal_ratio > 0:
            replay_data = self._get_replay_batch()
        else:
            replay_data = None

        for batch_idx, batch in enumerate(train_loader):
            data = batch['data'].to(self.device)
            labels = batch['label'].to(self.device)

            # 当前批次的域标签
            batch_size = data.shape[0]
            domain_labels = torch.full((batch_size,),
                                       self.domains.index(domain),
                                       device=self.device)

            # 生成episode
            episode_loader = train_loader.dataset.episode_loader
            support_data, support_labels, query_data, query_labels = \
                episode_loader.generate_episode()

            support_data = support_data.to(self.device)
            support_labels = support_labels.to(self.device)
            query_data = query_data.to(self.device)
            query_labels = query_labels.to(self.device)

            # 前向传播和损失计算
            loss, metrics = self.criterion(
                self.model, support_data, support_labels,
                query_data, query_labels,
                domain_labels if self.criterion.domain_weight > 0 else None
            )

            # 记忆回放损失
            if replay_data is not None:
                replay_loss = self._compute_replay_loss(replay_data)
                loss += replay_loss
                metrics['replay_loss'] = replay_loss.item()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config['training'].get('grad_clip', 5.0)
            )

            optimizer.step()

            # 统计
            total_loss += loss.item()
            total_proto_loss += metrics['proto_loss']
            if 'domain_loss' in metrics:
                total_domain_loss += metrics['domain_loss']

            # 计算准确率
            with torch.no_grad():
                pred, _ = self.model.predict(query_data)
                correct += (pred == query_labels).sum().item()
                total += query_labels.size(0)

        scheduler.step()

        return {
            'total_loss': total_loss / len(train_loader),
            'proto_loss': total_proto_loss / len(train_loader),
            'domain_loss': total_domain_loss / len(train_loader),
            'accuracy': correct / total
        }

    def _update_memory_buffer(self, domain: str, train_loader: DataLoader):
        """更新记忆缓冲区"""

        print(f"更新域 {domain} 的记忆缓冲区...")

        self.model.eval()

        all_features = []
        all_labels = []
        all_data = []

        with torch.no_grad():
            for batch in train_loader:
                data = batch['data'].to(self.device)
                labels = batch['label']

                # 提取特征
                _, features = self.model.backbone(data, return_features=True)

                all_data.append(data.cpu())
                all_features.append(features.cpu())
                all_labels.append(labels)

        # 合并
        all_data = torch.cat(all_data, dim=0)
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # 每类选择代表性样本（聚类中心附近）
        selected_data = []
        selected_labels = []
        selected_features = []

        for label in torch.unique(all_labels):
            mask = (all_labels == label)
            class_data = all_data[mask]
            class_features = all_features[mask]
            class_labels = all_labels[mask]

            # 计算类中心
            class_center = class_features.mean(dim=0, keepdim=True)

            # 选择距离中心最近的样本
            distances = torch.norm(class_features - class_center, dim=1)
            _, indices = torch.topk(distances,
                                    min(len(class_data) // 2, 10),
                                    largest=False)

            selected_data.append(class_data[indices])
            selected_labels.append(class_labels[indices])
            selected_features.append(class_features[indices])

        # 存储到缓冲区
        self.memory_buffer[domain] = {
            'data': torch.cat(selected_data, dim=0),
            'labels': torch.cat(selected_labels, dim=0),
            'features': torch.cat(selected_features, dim=0)
        }

        # 限制缓冲区大小
        self._limit_memory_buffer()

    def _limit_memory_buffer(self):
        """限制记忆缓冲区总大小"""

        total_size = sum(buffer['data'].shape[0]
                         for buffer in self.memory_buffer.values())

        if total_size > self.memory_size:
            # 按比例减少每个域的样本
            ratio = self.memory_size / total_size

            for domain in self.memory_buffer:
                buffer = self.memory_buffer[domain]
                current_size = buffer['data'].shape[0]
                new_size = int(current_size * ratio)

                indices = torch.randperm(current_size)[:new_size]

                self.memory_buffer[domain] = {
                    'data': buffer['data'][indices],
                    'labels': buffer['labels'][indices],
                    'features': buffer['features'][indices]
                }

    def _get_replay_batch(self) -> Dict[str, torch.Tensor]:
        """获取回放批次"""

        if not self.memory_buffer:
            return None

        replay_data = []
        replay_labels = []

        # 按比例从每个域采样
        for domain, buffer in self.memory_buffer.items():
            domain_size = buffer['data'].shape[0]
            replay_size = int(domain_size * self.rehearsal_ratio)

            indices = torch.randperm(domain_size)[:replay_size]

            replay_data.append(buffer['data'][indices])
            replay_labels.append(buffer['labels'][indices])

        return {
            'data': torch.cat(replay_data, dim=0).to(self.device),
            'labels': torch.cat(replay_labels, dim=0).to(self.device)
        }

    def _compute_replay_loss(self, replay_data: Dict) -> torch.Tensor:
        """计算回放损失"""

        data = replay_data['data']
        labels = replay_data['labels']

        # 使用知识蒸馏防止灾难性遗忘
        with torch.no_grad():
            old_features = self.model.backbone(data, return_features=True)[1]

        _, new_features = self.model.backbone(data, return_features=True)

        # MSE损失保持特征稳定
        replay_loss = F.mse_loss(new_features, old_features.detach())

        return replay_loss * 0.1  # 权重衰减

    def train_continual(self, train_loaders: Dict[str, DataLoader],
                        val_loaders: Dict[str, DataLoader]) -> Dict:
        """持续学习主流程"""

        results = {}

        for domain_idx, domain in enumerate(self.domains):
            self.current_domain_idx = domain_idx

            train_loader = train_loaders[domain]
            val_loader = val_loaders[domain]

            # 训练当前域
            domain_results = self.train_domain(
                domain, train_loader, val_loader,
                self.config['training']['epochs']
            )

            results[domain] = domain_results

            # 评估之前所有域
            if domain_idx > 0:
                forgetting = self._evaluate_forgetting(val_loaders, domain_idx)
                print(f"遗忘程度: {forgetting:.4f}")
                domain_results['forgetting'] = forgetting

        # 最终评估
        print("\n" + "=" * 60)
        print("持续学习完成！最终结果:")
        print("=" * 60)

        for domain, perf in self.domain_performance.items():
            print(f"域 {domain}: {perf:.4f}")

        return results

    def _evaluate_forgetting(self, val_loaders: Dict[str, DataLoader],
                             current_idx: int) -> float:
        """评估灾难性遗忘"""

        forgettings = []

        for i in range(current_idx):
            domain = self.domains[i]
            val_loader = val_loaders[domain]

            current_acc = self._evaluate(val_loader)['accuracy']
            original_acc = self.domain_performance[domain]

            forgetting = max(0, original_acc - current_acc)
            forgettings.append(forgetting)

        return np.mean(forgettings)