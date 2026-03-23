# -*- coding: utf-8 -*-
"""Unified training entry with YAML config."""
from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from src.datasets import build_dataset
from src.models import build_method
from src.samplers import build_train_batch_sampler
from src.trainer import Trainer
from src.utils.condition_utils import build_condition_table_from_datasets
from src.utils.config import dump_yaml, load_config
from src.utils.runtime import get_device, set_seed




def parse_args():
    parser = argparse.ArgumentParser(description='Unified strict DG training entry.')
    parser.add_argument('--configs', type=str, required=True, help='Path to model config yaml.')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.configs)

    # 打印当前使用的配置文件路径
    config_path = Path(args.configs)
    print(f"开始训练，当前使用的配置文件：")
    print(f"{config_path}")

    set_seed(int(cfg['train'].get('seed', 42)))
    device = get_device(cfg['train'].get('device', 'auto'))
    print('Using device:', device)


    output_dir = Path(cfg['output']['root']) / str(cfg['method']['name']).lower() / str(cfg['output']['exp_name'])
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(cfg, output_dir / 'merged_config.yaml')

    dataset_kwargs = {
        'feature_mode': cfg['model']['feature_mode'],
        'to_tensor': True,
    }
    num_workers = int(cfg['data'].get('num_workers', 0))
    pin_memory = bool(cfg['data'].get('pin_memory', True)) and device.type == 'cuda'
    batch_size = int(cfg['train']['batch_size'])
    test_batch_size = int(cfg['train'].get('test_batch_size', batch_size))

    train_dataset = build_dataset(
        h5_path=cfg['data']['train_h5'],
        dataset_name=cfg['data']['dataset_name'],
        **dataset_kwargs,
    )
    test_dataset = build_dataset(
        h5_path=cfg['data']['test_h5'],
        dataset_name=cfg['data']['dataset_name'],
        **dataset_kwargs,
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
            drop_last=bool(cfg['data'].get('drop_last_train', False)),
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

    model = build_method(cfg)
    if hasattr(model, 'set_condition_lookup'):
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
