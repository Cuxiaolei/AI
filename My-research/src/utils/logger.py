# -*- coding: utf-8 -*-
"""Reusable experiment logger and tabular recorder."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List

try:
    from openpyxl import Workbook
except Exception:  # pragma: no cover
    Workbook = None


class ResultRecorder:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rows: List[Dict] = []

    def append(self, row: Dict) -> None:
        self.rows.append(dict(row))

    def _collect_fieldnames(self) -> List[str]:
        fieldnames: List[str] = []
        for row in self.rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        return fieldnames

    def save_csv(self, filename: str = 'results.csv') -> Path:
        path = self.output_dir / filename
        if not self.rows:
            return path
        fieldnames = self._collect_fieldnames()
        with open(path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        return path

    def save_xlsx(self, filename: str = 'results.xlsx') -> Path | None:
        if Workbook is None or not self.rows:
            return None
        path = self.output_dir / filename
        wb = Workbook()
        ws = wb.active
        ws.title = 'results'
        headers = self._collect_fieldnames()
        ws.append(headers)
        for row in self.rows:
            ws.append([row.get(h) for h in headers])
        wb.save(path)
        return path

    def flush(self) -> None:
        self.save_csv()
        self.save_xlsx()


def build_logger(output_dir: str | Path, filename: str = 'train.log', name: str = 'train') -> logging.Logger:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f'{name}_{output_dir.as_posix()}')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(output_dir / filename, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


__all__ = ['ResultRecorder', 'build_logger']
