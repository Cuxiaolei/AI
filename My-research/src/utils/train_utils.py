# src/utils/trainer_utils.py
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.confusion import build_label_names, export_confusion_matrix
from src.utils.logger import ResultRecorder, build_logger


def build_trainer_logger_and_recorder(output_dir: Path):
    """构建训练器的logger和结果记录器"""
    logger = build_logger(output_dir)
    recorder = ResultRecorder(output_dir)
    return logger, recorder


def save_trainer_checkpoint(
    output_dir: Path,
    epoch: int,
    history: List[Dict],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    cfg: dict
):
    """保存训练器checkpoint"""
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'history': history,
        'config': cfg,
    }
    torch.save(ckpt, output_dir / 'last.pth')


def log_train_epoch(
    logger,
    epoch: int,
    total_epochs: int,
    train_metrics: Dict[str, float]
):
    """记录训练epoch的日志"""
    extra_msg = ' '.join([
        f'{k}={v:.4f}' for k, v in train_metrics.items() 
        if k not in {'loss','acc'}
    ])
    logger.info(
        'epoch=%d/%d train_loss=%.4f train_acc=%.4f%s%s',
        epoch, total_epochs,
        train_metrics['loss'], train_metrics['acc'],
        ' | ' if extra_msg else '',
        extra_msg,
    )


def log_final_test(logger, final_metrics: Dict[str, float]):
    """记录最终测试的日志"""
    logger.info(
        'final_target_test loss=%.4f acc=%.4f precision_macro=%.4f recall_macro=%.4f f1_macro=%.4f',
        final_metrics['loss'], final_metrics['acc'], final_metrics['precision_macro'],
        final_metrics['recall_macro'], final_metrics['f1_macro'],
    )


def save_final_test_metrics(output_dir: Path, epochs: int, final_metrics: Dict[str, float]):
    """保存最终测试指标到JSON文件"""
    final_test_metrics = {
        'phase': 'final_test',
        'epoch': epochs,
        **{k: float(v) for k, v in final_metrics.items()}
    }
    with open(output_dir / 'final_test_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(final_test_metrics, f, indent=2, ensure_ascii=False)


def export_final_confusion_matrix(
    cm: np.ndarray,
    test_loader: DataLoader,
    num_classes: int,
    output_dir: Path
):
    """导出最终测试的混淆矩阵"""
    label_map = None
    if hasattr(test_loader.dataset, 'get_label_map'):
        label_map = test_loader.dataset.get_label_map()
    label_names = build_label_names(num_classes, label_map=label_map)
    export_confusion_matrix(
        cm, 
        label_names, 
        output_dir / 'confusion_matrices', 
        stem='confusion_matrix_last'
    )