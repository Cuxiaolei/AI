import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import numpy as np
import copy
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

        print(f"ContinualLearningTrainer initialized:")
        print(f"  - Source domains: {len(self.domains)}")
        print(f"  - Memory buffer size: {self.memory_size}")
        print(f"  - Rehearsal ratio: {self.rehearsal_ratio}")

    def train_domain(self, domain: str, episode_loader,
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
                domain, episode_loader, optimizer, scheduler
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

            # 更新记忆缓冲区（epoch中间时刻）
            if epoch == epochs // 2:
                self._update_memory_buffer(domain, episode_loader)

        # 恢复最佳模型
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.domain_performance[domain] = best_acc

        return {
            'domain': domain,
            'best_accuracy': best_acc,
            'final_metrics': val_metrics
        }

    def _train_epoch(self, domain: str, episode_loader,
                     optimizer: torch.optim.Optimizer,
                     scheduler: torch.optim.lr_scheduler) -> Dict:
        """训练一个epoch——核心修改：直接使用episode_loader"""

        self.model.train()

        total_loss = 0
        total_proto_loss = 0
        total_domain_loss = 0
        correct = 0
        total = 0

        # 每个 epoch 训练固定数量的 episodes
        num_episodes = self.config['training'].get('episodes_per_epoch', 50)

        for episode_idx in range(num_episodes):
            # ✨ 核心：从 episode_loader 生成跨域 episode
            try:
                support_data, support_labels, query_data, query_labels = \
                    episode_loader.generate_episode()
            except Exception as e:
                print(f"⚠️  Episode generation failed: {e}")
                continue

            support_data = support_data.to(self.device)
            support_labels = support_labels.to(self.device)
            query_data = query_data.to(self.device)
            query_labels = query_labels.to(self.device)

            # 当前批次的域标签（查询集）
            batch_size = query_data.shape[0]
            domain_label_idx = self.domains.index(domain)
            domain_labels = torch.full((batch_size,), domain_label_idx, device=self.device)

            # 记忆回放数据
            if self.memory_buffer and self.rehearsal_ratio > 0:
                replay_data = self._get_replay_batch()
            else:
                replay_data = None

            # 前向传播和损失计算
            loss, metrics = self.criterion(
                self.model, support_data, support_labels,
                query_data, query_labels,
                domain_labels if self.criterion.domain_weight > 0 else None
            )

            # 回放损失
            if replay_data is not None:
                replay_loss = self._compute_replay_loss(replay_data)
                loss += replay_loss
                metrics['replay_loss'] = replay_loss.item()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            optimizer.step()

            # 统计
            total_loss += loss.item()
            total_proto_loss += metrics['proto_loss']
            if 'domain_loss' in metrics:
                total_domain_loss += metrics['domain_loss']

            with torch.no_grad():
                pred, _ = self.model.predict(query_data)
                correct += (pred == query_labels).sum().item()
                total += query_labels.size(0)

        scheduler.step()

        return {
            'total_loss': total_loss / num_episodes,
            'proto_loss': total_proto_loss / num_episodes,
            'domain_loss': total_domain_loss / num_episodes if total_domain_loss > 0 else 0,
            'accuracy': correct / total if total > 0 else 0
        }

    def _update_memory_buffer(self, domain: str, episode_loader):
        """更新记忆缓冲区——从episode_loader中采样"""

        print(f"更新域 {domain} 的记忆缓冲区...")

        self.model.eval()

        # 使用该域的数据集采样
        domain_dataset = episode_loader.datasets[domain]

        # 简单采样：从该域中随机选样本
        all_data = domain_dataset.data
        all_labels = domain_dataset.labels

        if len(all_data) > 20:
            indices = np.random.choice(len(all_data), 20, replace=False)
        else:
            indices = np.arange(len(all_data))

        selected_data = torch.FloatTensor(all_data[indices]).to(self.device)
        selected_labels = torch.LongTensor(all_labels[indices])

        # 提取特征
        with torch.no_grad():
            _, features = self.model.backbone(selected_data.unsqueeze(1), return_features=True)

        # 存储到缓冲区
        self.memory_buffer[domain] = {
            'data': selected_data.cpu(),
            'labels': selected_labels,
            'features': features.cpu()
        }

        # 限制缓冲区总大小
        self._limit_memory_buffer()

    def _limit_memory_buffer(self):
        """限制记忆缓冲区总大小"""
        total_size = sum(buffer['data'].shape[0]
                         for buffer in self.memory_buffer.values())

        if total_size > self.memory_size:
            ratio = self.memory_size / total_size

            for domain in self.memory_buffer:
                buffer = self.memory_buffer[domain]
                current_size = buffer['data'].shape[0]
                new_size = int(current_size * ratio)

                if new_size > 0:
                    indices = torch.randperm(current_size)[:new_size]
                    self.memory_buffer[domain] = {
                        'data': buffer['data'][indices],
                        'labels': buffer['labels'][indices],
                        'features': buffer['features'][indices]
                    }
                else:
                    # 如果该域样本数被压缩到0，删除该域
                    del self.memory_buffer[domain]

    def _get_replay_batch(self) -> Dict:
        """从记忆缓冲区获取回放批次"""
        if not self.memory_buffer:
            return None

        replay_data = []
        replay_labels = []

        # 按比例从每个域采样
        for domain, buffer in self.memory_buffer.items():
            domain_size = buffer['data'].shape[0]
            replay_size = int(domain_size * self.rehearsal_ratio)

            if replay_size > 0:
                indices = torch.randperm(domain_size)[:replay_size]

                replay_data.append(buffer['data'][indices])
                replay_labels.append(buffer['labels'][indices])

        if not replay_data:
            return None

        return {
            'data': torch.cat(replay_data, dim=0).to(self.device),
            'labels': torch.cat(replay_labels, dim=0).to(self.device)
        }

    def _compute_replay_loss(self, replay_data: Dict) -> torch.Tensor:
        """计算回放损失（知识蒸馏）"""
        data = replay_data['data']
        old_features = replay_data.get('old_features', None)

        if old_features is None:
            # 如果没有旧特征，返回0损失
            return torch.tensor(0.0, device=self.device)

        # 提取新特征
        _, new_features = self.model.backbone(data, return_features=True)

        # MSE损失保持特征稳定
        replay_loss = nn.functional.mse_loss(new_features, old_features.detach())

        return replay_loss * 0.1

    def train_continual(self, episode_loader, val_loaders: Dict, target_loaders: Dict) -> Dict:
        """持续学习主流程"""

        results = {}

        for domain_idx, domain in enumerate(self.domains):
            self.current_domain_idx = domain_idx

            val_loader = val_loaders[domain]

            # 训练当前域
            domain_results = self.train_domain(
                domain, episode_loader, val_loader,
                self.config['training']['epochs']
            )

            results[domain] = domain_results

            # 评估之前所有域的遗忘程度
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

        # 在目标域上评估泛化能力
        if target_loaders:
            self._evaluate_target_domains(target_loaders)

        return results

    def _evaluate_forgetting(self, val_loaders: Dict, current_idx: int) -> float:
        """评估灾难性遗忘"""
        forgettings = []

        for i in range(current_idx):
            domain = self.domains[i]
            val_loader = val_loaders[domain]

            current_acc = self._evaluate(val_loader)['accuracy']
            original_acc = self.domain_performance[domain]

            forgetting = max(0, original_acc - current_acc)
            forgettings.append(forgetting)

        return np.mean(forgettings) if forgettings else 0

    def _evaluate_target_domains(self, target_loaders: Dict):
        """评估在目标域上的泛化性能"""
        print("\n" + "=" * 60)
        print("在目标域上评估泛化能力:")
        print("=" * 60)

        for domain, loader in target_loaders.items():
            metrics = self._evaluate(loader)
            print(f"目标域 {domain}: Accuracy={metrics['accuracy']:.4f}")