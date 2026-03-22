# -*- coding: utf-8 -*-
"""Confusion matrix export utilities with table headers."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

try:
    from openpyxl import Workbook
except Exception:  # pragma: no cover
    Workbook = None


def build_label_names(num_classes: int, label_map: dict | None = None) -> List[str]:
    """Return ordered label names by class id."""
    if not label_map:
        return [f'class_{i}' for i in range(num_classes)]

    ordered = [None] * num_classes
    # expected format usually {name: id}
    for k, v in label_map.items():
        try:
            idx = int(v)
        except Exception:
            continue
        if 0 <= idx < num_classes:
            ordered[idx] = str(k)

    for i in range(num_classes):
        if ordered[i] is None:
            ordered[i] = f'class_{i}'
    return ordered


def confusion_matrix_to_rows(cm: np.ndarray, label_names: Sequence[str], normalize: bool = False) -> List[List]:
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError('Confusion matrix must be square.')
    if len(label_names) != cm.shape[0]:
        raise ValueError('label_names length must equal confusion matrix size.')

    mat = cm.astype(np.float64) if normalize else cm.astype(np.int64)
    if normalize:
        row_sum = mat.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        mat = mat / row_sum

    headers = ['Actual\\Pred'] + list(label_names)
    rows = [headers]
    for i, name in enumerate(label_names):
        row = [name]
        for j in range(len(label_names)):
            val = mat[i, j]
            row.append(round(float(val), 6) if normalize else int(val))
        rows.append(row)
    return rows


def save_rows_to_csv(rows: Iterable[Sequence], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(list(row))
    return path


def save_confusion_matrix_csv(cm: np.ndarray, label_names: Sequence[str], path: str | Path, normalize: bool = False) -> Path:
    rows = confusion_matrix_to_rows(cm, label_names, normalize=normalize)
    return save_rows_to_csv(rows, path)


def save_confusion_matrix_xlsx(cm: np.ndarray, label_names: Sequence[str], path: str | Path) -> Path | None:
    if Workbook is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()

    ws_counts = wb.active
    ws_counts.title = 'counts'
    for row in confusion_matrix_to_rows(cm, label_names, normalize=False):
        ws_counts.append(row)

    ws_norm = wb.create_sheet(title='normalized')
    for row in confusion_matrix_to_rows(cm, label_names, normalize=True):
        ws_norm.append(row)

    wb.save(path)
    return path


def export_confusion_matrix(cm: np.ndarray, label_names: Sequence[str], output_dir: str | Path, stem: str = 'confusion_matrix') -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_confusion_matrix_csv(cm, label_names, output_dir / f'{stem}.csv', normalize=False)
    csv_norm_path = save_confusion_matrix_csv(cm, label_names, output_dir / f'{stem}_normalized.csv', normalize=True)
    xlsx_path = save_confusion_matrix_xlsx(cm, label_names, output_dir / f'{stem}.xlsx')
    return {
        'csv': str(csv_path),
        'csv_normalized': str(csv_norm_path),
        'xlsx': str(xlsx_path) if xlsx_path is not None else None,
    }


__all__ = [
    'build_label_names',
    'confusion_matrix_to_rows',
    'save_confusion_matrix_csv',
    'save_confusion_matrix_xlsx',
    'export_confusion_matrix',
]
