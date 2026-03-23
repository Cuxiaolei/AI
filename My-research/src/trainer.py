# -*- coding: utf-8 -*-
"""Unified trainer for strict domain generalization baselines and meta-learning variants."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.losses import build_classification_loss, compute_class_weights_from_loader
from src.utils.confusion import build_label_names, export_confusion_matrix
from src.utils.logger import ResultRecorder, build_logger
from src.utils.metrics import classification_metrics_from_confusion, confusion_matrix_from_arrays
from src.utils.optim import build_optimizer, build_scheduler
from src.utils.runtime import move_batch_to_device, tqdm


class Trainer:
    def __init__(self, cfg: dict, model: torch.nn.Module, train_loader: DataLoader, test_loader: DataLoader, device: torch.device, output_dir: str | Path) -> None:
        self.cfg = cfg
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = build_logger(self.output_dir)
        self.recorder = ResultRecorder(self.output_dir)

        self.optimizer = build_optimizer(self.model, cfg['optimizer'])
        sched_cfg = dict(cfg.get('scheduler', {}))
        sched_cfg.setdefault('epochs', int(cfg['train']['epochs']))
        self.scheduler = build_scheduler(self.optimizer, sched_cfg)

        class_weights = None
        if bool(cfg.get('loss', {}).get('use_class_weights', False)):
            class_weights = compute_class_weights_from_loader(self.train_loader, int(cfg['data']['num_classes'])).to(device)
        self.criterion = build_classification_loss(cfg.get('loss', {}), class_weights=class_weights)
        self.global_step = 0

    def _compute_model_step(self, batch: Dict[str, torch.Tensor], epoch: int) -> Dict[str, torch.Tensor]:
        if hasattr(self.model, 'compute_loss'):
            return self.model.compute_loss(batch, self.criterion, epoch=epoch, global_step=self.global_step)
        out = self.model(batch)
        logits = out['logits']
        y = batch['y']
        loss = self.criterion(logits, y)
        out['loss'] = loss
        return out

    @staticmethod
    def _tensor_scalar(v) -> float | None:
        if torch.is_tensor(v) and v.numel() == 1:
            return float(v.detach().item())
        return None

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_samples = 0
        total_correct = 0
        total_loss = 0.0
        extras_sum = defaultdict(float)

        pbar = tqdm(self.train_loader, desc=f'Train {epoch}', leave=False)
        for batch in pbar:
            batch = move_batch_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            out = self._compute_model_step(batch, epoch)
            logits = out['logits']
            y = batch['y']
            loss = out['loss']
            loss.backward()
            self.optimizer.step()
            self.global_step += 1

            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            total_samples += bs
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            for k, v in out.items():
                if k in {'loss', 'logits', 'feature', 'feat_freq', 'feat_tf'}:
                    continue
                scalar = self._tensor_scalar(v)
                if scalar is not None:
                    extras_sum[k] += scalar * bs
            pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{total_correct/max(total_samples,1):.4f}')

        metrics = {
            'loss': total_loss / max(total_samples, 1),
            'acc': total_correct / max(total_samples, 1),
        }
        for k, v in extras_sum.items():
            metrics[k] = v / max(total_samples, 1)
        return metrics

    @torch.no_grad()
    def evaluate_final(self) -> Tuple[Dict[str, float], np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_true = []
        all_pred = []

        pbar = tqdm(self.test_loader, desc='Final Test', leave=False)
        for batch in pbar:
            batch = move_batch_to_device(batch, self.device)
            logits = self.model(batch)['logits']
            y = batch['y']
            loss = self.criterion(logits, y)
            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            total_samples += bs
            all_true.append(y.detach().cpu().numpy())
            all_pred.append(logits.argmax(dim=1).detach().cpu().numpy())
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        y_true = np.concatenate(all_true, axis=0) if all_true else np.empty((0,), dtype=np.int64)
        y_pred = np.concatenate(all_pred, axis=0) if all_pred else np.empty((0,), dtype=np.int64)
        cm = confusion_matrix_from_arrays(y_true, y_pred, num_classes=int(self.cfg['data']['num_classes']))
        metrics = classification_metrics_from_confusion(cm)
        metrics['loss'] = total_loss / max(total_samples, 1)
        return metrics, cm

    def save_checkpoint(self, epoch: int, history: List[Dict]) -> None:
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'history': history,
            'config': self.cfg,
        }
        torch.save(ckpt, self.output_dir / 'last.pth')

    def fit(self) -> List[Dict]:
        epochs = int(self.cfg['train']['epochs'])
        history: List[Dict] = []
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_one_epoch(epoch)
            if self.scheduler is not None:
                self.scheduler.step()
            row = {
                'phase': 'train',
                'epoch': epoch,
                'lr': self.optimizer.param_groups[0]['lr'],
                **{f'train_{k}': v for k, v in train_metrics.items()},
            }
            history.append(row)
            self.recorder.append(row)
            self.recorder.flush()
            if bool(self.cfg.get('output', {}).get('save_checkpoint', False)):
                self.save_checkpoint(epoch, history)

            extra_msg = ' '.join([f'{k}={v:.4f}' for k, v in train_metrics.items() if k not in {'loss','acc'}])
            self.logger.info(
                'epoch=%d/%d train_loss=%.4f train_acc=%.4f%s%s',
                epoch, epochs,
                train_metrics['loss'], train_metrics['acc'],
                ' | ' if extra_msg else '',
                extra_msg,
            )

        final_metrics, cm = self.evaluate_final()
        final_row = {
            'phase': 'final_test',
            'epoch': epochs,
            **{f'test_{k}': v for k, v in final_metrics.items()},
        }
        history.append(final_row)
        self.recorder.append(final_row)
        self.recorder.flush()

        print("before export_confusion_matrix")
        label_map = None
        if hasattr(self.test_loader.dataset, 'get_label_map'):
            label_map = self.test_loader.dataset.get_label_map()
        label_names = build_label_names(int(self.cfg['data']['num_classes']), label_map=label_map)
        export_confusion_matrix(cm, label_names, self.output_dir / 'confusion_matrices', stem='confusion_matrix_last')
        print("after export_confusion_matrix")


        self.logger.info(
            'final_target_test loss=%.4f acc=%.4f precision_macro=%.4f recall_macro=%.4f f1_macro=%.4f',
            final_metrics['loss'], final_metrics['acc'], final_metrics['precision_macro'],
            final_metrics['recall_macro'], final_metrics['f1_macro'],
        )

        print("before save_checkpoint")
        # 单独保存最终测试指标
        final_test_metrics = {
            'phase': 'final_test',
            'epoch': epochs,
            **{k: float(v) for k, v in final_metrics.items()}
        }
        with open(self.output_dir / 'final_test_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(final_test_metrics, f, indent=2, ensure_ascii=False)
        print("after save_checkpoint")

        return history


