# -*- coding: utf-8 -*-
"""Unified trainer for strict domain generalization baselines and meta-learning variants."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.losses import build_classification_loss, compute_class_weights_from_loader
from src.utils.optim import build_optimizer, build_scheduler
from src.utils.runtime import move_batch_to_device, tqdm
from src.utils.metrics import classification_metrics_from_confusion, confusion_matrix_from_arrays
from src.utils.train_utils import (
    build_trainer_logger_and_recorder,
    save_trainer_checkpoint,
    log_train_epoch,
    log_final_test,
    save_final_test_metrics,
    export_final_confusion_matrix,
    clean_up_dataloaders,
)

class Trainer:
    def __init__(
            self,
            cfg: dict,
            model: torch.nn.Module,
            train_loader: DataLoader,
            test_loader: DataLoader,
            device: torch.device,
            output_dir: Path
    ) -> None:
        # 基础配置初始化
        self.cfg = cfg
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.global_step = 0

        # 日志/记录器初始化
        self.logger, self.recorder = build_trainer_logger_and_recorder(self.output_dir)

        # 优化器/调度器初始化
        self.optimizer = build_optimizer(self.model, cfg['optimizer'])
        self.scheduler = self._build_scheduler()

        # 损失函数初始化（含类别权重）
        self.criterion = self._build_criterion()

    def _build_scheduler(self):
        """构建学习率调度器"""
        sched_cfg = dict(self.cfg.get('scheduler', {}))
        sched_cfg.setdefault('epochs', int(self.cfg['train']['epochs']))
        return build_scheduler(self.optimizer, sched_cfg)

    def _build_criterion(self):
        """构建分类损失函数（支持类别权重）"""
        class_weights = None
        if self.cfg.get('loss', {}).get('use_class_weights', False):
            num_classes = int(self.cfg['data']['num_classes'])
            class_weights = compute_class_weights_from_loader(self.train_loader, num_classes).to(self.device)
        return build_classification_loss(self.cfg.get('loss', {}), class_weights=class_weights)

    def _compute_model_step(self, batch: Dict[str, torch.Tensor], epoch: int) -> Dict[str, torch.Tensor]:
        """单次前向传播计算（支持模型自定义loss计算）"""
        return self.model.compute_loss(batch, self.criterion, epoch=epoch, global_step=self.global_step)

    @staticmethod
    def _tensor_to_scalar(v) -> Optional[float]:
        """将单元素tensor转为标量，否则返回None"""
        if torch.is_tensor(v) and v.numel() == 1:
            return float(v.detach().item())
        return None

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """训练单个epoch并返回训练指标"""
        self.model.train()
        total_samples = 0
        total_correct = 0
        total_loss = 0.0
        extras_sum = defaultdict(float)

        pbar = tqdm(self.train_loader, desc=f'Train {epoch}', leave=False)
        for batch in pbar:
            # 数据移至设备 + 梯度清零
            batch = move_batch_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)

            # 前向传播 + 反向传播 + 优化
            out = self._compute_model_step(batch, epoch)
            loss = out['loss']
            loss.backward()
            self.optimizer.step()
            self.global_step += 1

            # 累计指标
            bs = batch['y'].size(0)
            total_loss += float(loss.item()) * bs
            total_samples += bs
            total_correct += int((out['logits'].argmax(dim=1) == batch['y']).sum().item())

            # 累计额外指标
            for k, v in out.items():
                if k in {'loss', 'logits', 'feature', 'feat_freq'}:
                    continue
                scalar = self._tensor_to_scalar(v)
                if scalar is not None:
                    extras_sum[k] += scalar * bs

            # 更新进度条
            pbar.set_postfix(
                loss=f'{loss.item():.4f}',
                acc=f'{total_correct / max(total_samples, 1):.4f}'
            )

        # 计算平均指标
        metrics = {
            'loss': total_loss / max(total_samples, 1),
            'acc': total_correct / max(total_samples, 1),
        }
        for k, v in extras_sum.items():
            metrics[k] = v / max(total_samples, 1)
        return metrics

    @torch.no_grad()
    def evaluate_final(self, desc: str = 'Final Test') -> Tuple[Dict[str, float], np.ndarray]:
        """测试集评估，返回指标和混淆矩阵。desc用于区分逐epoch测试和最终测试。"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_true = []
        all_pred = []

        pbar = tqdm(self.test_loader, desc=desc, leave=False)
        for batch in pbar:
            batch = move_batch_to_device(batch, self.device)
            logits = self.model(batch)['logits']
            y = batch['y']
            loss = self.criterion(logits, y)

            # 累计损失和样本数
            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            total_samples += bs

            # 收集预测/真实标签
            all_true.append(y.detach().cpu().numpy())
            all_pred.append(logits.argmax(dim=1).detach().cpu().numpy())

            pbar.set_postfix(loss=f'{loss.item():.4f}')

        # 拼接标签并计算混淆矩阵/指标
        y_true = np.concatenate(all_true) if all_true else np.empty((0,), dtype=np.int64)
        y_pred = np.concatenate(all_pred) if all_pred else np.empty((0,), dtype=np.int64)
        cm = confusion_matrix_from_arrays(y_true, y_pred, num_classes=int(self.cfg['data']['num_classes']))
        metrics = classification_metrics_from_confusion(cm)
        metrics['loss'] = total_loss / max(total_samples, 1)

        return metrics, cm

    def close(self) -> None:
        """释放资源（关闭dataloader/数据集，清理显存）"""
        clean_up_dataloaders(self.train_loader, self.test_loader)
        self.train_loader = None
        self.test_loader = None

    def fit(self) -> List[Dict]:
        epochs = int(self.cfg['train']['epochs'])
        history: List[Dict] = []
        last_test_metrics: Optional[Dict[str, float]] = None
        last_cm: Optional[np.ndarray] = None

        for epoch in range(1, epochs + 1):
            # 1) 训练一个epoch
            train_metrics = self.train_one_epoch(epoch)
            self.scheduler.step()

            # 2) 每个epoch结束后立即在测试集上评估一次
            test_metrics, cm = self.evaluate_final(desc=f'Test {epoch}')
            last_test_metrics = test_metrics
            last_cm = cm

            # 3) 一行同时记录训练指标和测试指标，便于后续直接画epoch-accuracy曲线
            row = {
                'phase': 'epoch',
                'epoch': epoch,
                'lr': self.optimizer.param_groups[0]['lr'],
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'test_{k}': v for k, v in test_metrics.items()},
            }
            history.append(row)
            self.recorder.append(row)
            self.recorder.flush()

            # 4) 保存checkpoint（如果配置开启）
            if self.cfg.get('output', {}).get('save_checkpoint', False):
                save_trainer_checkpoint(
                    output_dir=self.output_dir,
                    epoch=epoch,
                    history=history,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    cfg=self.cfg
                )

            log_train_epoch(self.logger, epoch, epochs, train_metrics)
            self.logger.info(
                f"[Epoch {epoch}/{epochs}] "
                f"test_acc={test_metrics.get('acc', 0.0):.4f}, "
                f"test_f1_macro={test_metrics.get('f1_macro', 0.0):.4f}"
            )

        # 最后一个epoch已经测试过，直接复用最后一次测试结果保存最终指标和混淆矩阵
        final_metrics = last_test_metrics if last_test_metrics is not None else {}
        cm = last_cm if last_cm is not None else np.zeros(
            (int(self.cfg['data']['num_classes']), int(self.cfg['data']['num_classes'])),
            dtype=np.int64
        )

        export_final_confusion_matrix(
            cm=cm,
            test_loader=self.test_loader,
            num_classes=int(self.cfg['data']['num_classes']),
            output_dir=self.output_dir
        )
        log_final_test(self.logger, final_metrics)
        save_final_test_metrics(self.output_dir, epochs, final_metrics)

        # 释放资源
        self.close()
        return history
