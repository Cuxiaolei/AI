# -*- coding: utf-8 -*-
"""Unified training entry with YAML config."""
from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from src.datasets import build_dataset
from src.models import build_method
from src.datasets.samplers import build_train_batch_sampler
from src.trainer import Trainer
from src.utils.condition_utils import build_condition_table_from_datasets
from src.utils.config import dump_yaml, load_config
from src.utils.runtime import get_device, set_seed



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.configs)

    # 打印相关信息
    config_path = Path(args.configs)
    device = get_device(cfg['train'].get('device', 'auto'))
    print(f"开始训练，当前使用的配置文件：{config_path}")
    print('Using device:', device)

    set_seed(cfg['train']['seed'])
    output_dir = Path(cfg['output']['root']) / str(cfg['method']['name']).lower() / str(cfg['output']['exp_name'])
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(cfg, output_dir / 'merged_config.yaml')

    num_workers = int(cfg['data'].get('num_workers', 0))
    pin_memory = bool(cfg['data'].get('pin_memory', True)) and device.type == 'cuda'
    batch_size = int(cfg['train']['batch_size'])
    test_batch_size = int(cfg['train'].get('test_batch_size', batch_size))




    train_dataset = build_dataset(
        h5_path=cfg['data']['train_h5'],
        dataset_name=cfg['data']['dataset_name'],
        to_tensor=True
    )

    test_dataset = build_dataset(
        h5_path=cfg['data']['test_h5'],
        dataset_name=cfg['data']['dataset_name'],
        to_tensor=True
    )

    batch_sampler = build_train_batch_sampler(
        dataset=train_dataset,
        sampler_cfg=cfg.get('sampler', None),
        batch_size=batch_size,
        seed=int(cfg['train'].get('seed', 42)),
    )

    if batch_sampler is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=bool(cfg['data'].get('drop_last_train')),
            persistent_workers=(num_workers > 0),
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    cfg['data']['num_domains'] = int(train_dataset.get_num_domains())
    print(cfg['data']['num_domains'])
    model = build_method(cfg)

    method_name = str(cfg['method']['name']).lower()
    if method_name == 'mcpdg':
        condition_table, meta = build_condition_table_from_datasets(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            dataset_name=cfg['data']['dataset_name'],
        )
        model.set_condition_lookup(condition_table)
        print(f"[Condition] table shape={tuple(condition_table.shape)} dataset={meta['dataset_name']}")
    trainer = Trainer(cfg, model, train_loader, test_loader, device, output_dir)
    trainer.fit()


if __name__ == '__main__':
    main()
