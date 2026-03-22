# -*- coding: utf-8 -*-
"""Common classification metrics."""
from __future__ import annotations

from typing import Dict

import numpy as np


def confusion_matrix_from_arrays(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(np.int64), y_pred.astype(np.int64)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def classification_metrics_from_confusion(cm: np.ndarray) -> Dict[str, float]:
    total = cm.sum()
    acc = float(np.trace(cm) / total) if total > 0 else 0.0

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    support = cm.sum(axis=1).astype(np.float64)

    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / np.maximum(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)

    return {
        'acc': acc,
        'precision_macro': float(np.mean(precision)),
        'recall_macro': float(np.mean(recall)),
        'f1_macro': float(np.mean(f1)),
        'precision_weighted': float(np.sum(precision * support) / np.maximum(support.sum(), 1.0)),
        'recall_weighted': float(np.sum(recall * support) / np.maximum(support.sum(), 1.0)),
        'f1_weighted': float(np.sum(f1 * support) / np.maximum(support.sum(), 1.0)),
    }


__all__ = ['confusion_matrix_from_arrays', 'classification_metrics_from_confusion']
