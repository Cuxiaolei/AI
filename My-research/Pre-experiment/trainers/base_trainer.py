import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
import os
import time
from ..utils.metrics import compute_metrics
from ..utils.visualizer import plot_confusion_matrix


class BaseTrainer:
    """基础训练器"""

    def __init__(self, model, config, device='cuda'):
        self.model = model
        self.config = config
        self.device = device

        self.model.to(device)

        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def _evaluate(self, dataloader: DataLoader) -> Dict:
        """评估模型"""

        self.model.eval()

        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                data = batch['data'].to(self.device)
                labels = batch['label'].to(self.device)

                # 前向传播
                logits, features = self.model.backbone(data, return_features=True)
                loss = nn.CrossEntropyLoss()(logits, labels)

                total_loss += loss.item()

                # 预测
                pred = torch.argmax(logits, dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        metrics = compute_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(dataloader)

        return metrics

    def save_checkpoint(self, save_path: str, epoch: int,
                        is_best: bool = False):
        """保存模型"""

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': self.history
        }

        # 保存最新模型
        torch.save(checkpoint, os.path.join(save_path, 'latest_model.pth'))

        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, os.path.join(save_path, 'best_model.pth'))

        # 定期保存
        if epoch % self.config['evaluation']['save_freq'] == 0:
            torch.save(checkpoint,
                       os.path.join(save_path, f'model_epoch_{epoch}.pth'))

    def load_checkpoint(self, checkpoint_path: str):
        """加载模型"""

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']

        return checkpoint['epoch']