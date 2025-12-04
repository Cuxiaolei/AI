import argparse
import yaml
import torch
import os
import numpy as np
import random
from torch.utils.data import DataLoader

from data.dataset import OttawaBearingDataset
from data.preprocessor import SignalPreprocessor
from models.backbone import resnet18_1d
from models.prototypical_net import PrototypicalNetwork
from trainers.continual_trainer import ContinualLearningTrainer
from utils.metrics import print_metrics


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='轴承故障诊断 - 小样本域泛化'
    )
    parser.add_argument(
        '--config', '-c',
        default='configs/config.yaml',
        type=str,
        help='配置文件路径'
    )
    parser.add_argument(
        '--data_dir', '-d',
        type=str,
        help='数据集根目录'
    )
    parser.add_argument(
        '--save_dir', '-s',
        default='./outputs',
        type=str,
        help='保存目录'
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'test', 'continual'],
        default='train',
        help='运行模式'
    )
    parser.add_argument(
        '--resume',
        type=str,
        help='恢复的checkpoint路径'
    )

    return parser.parse_args()


def setup_data(config: dict):
    """设置数据加载器"""

    data_config = config['data']

    # 预处理器
    preprocessor = SignalPreprocessor(
        filtering=data_config.get('filtering', True),
        normalize=data_config.get('normalize', True),
        denoise=data_config.get('denoise', False)
    )

    # 创建数据集
    train_loaders = {}
    val_loaders = {}

    # 源域
    for domain in data_config['source_domains']:
        # 训练集
        train_dataset = OttawaBearingDataset(
            data_dir=data_config['root_dir'],
            domains=[domain],
            window_size=data_config['window_size'],
            overlap=data_config['overlap'],
            channels=data_config['channels'],
            k_shot=data_config['k_shot'],
            n_query=data_config['n_query'],
            mode='train',
            preprocessor=preprocessor
        )

        train_loaders[domain] = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )

        # 验证集
        val_dataset = OttawaBearingDataset(
            data_dir=data_config['root_dir'],
            domains=[domain],
            window_size=data_config['window_size'],
            overlap=0.0,  # 验证集无重叠
            channels=data_config['channels'],
            mode='test',
            preprocessor=preprocessor
        )

        val_loaders[domain] = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )

    # 目标域测试集
    target_loaders = {}
    for domain in data_config['target_domains']:
        target_dataset = OttawaBearingDataset(
            data_dir=data_config['root_dir'],
            domains=[domain],
            window_size=data_config['window_size'],
            overlap=0.0,
            channels=data_config['channels'],
            mode='test',
            preprocessor=preprocessor
        )

        target_loaders[domain] = DataLoader(
            target_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )

    return train_loaders, val_loaders, target_loaders


def setup_model(config: dict):
    """设置模型"""

    # 主干网络
    backbone = resnet18_1d(
        in_channels=1,
        num_classes=config['model']['num_classes'],
        feature_dim=config['model']['feature_dim']
    )

    # 原型网络
    model = PrototypicalNetwork(
        backbone=backbone,
        feature_dim=config['model']['feature_dim'],
        num_classes=config['model']['num_classes']
    )

    return model


def main():
    """主函数"""

    args = parse_args()

    # 加载配置
    config = load_config(args.config)

    # 覆盖数据路径
    if args.data_dir:
        config['data']['root_dir'] = args.data_dir

    # 设置随机种子
    set_seed(config['seed'])

    # 设备
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建输出目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 设置数据
    print("设置数据加载器...")
    train_loaders, val_loaders, target_loaders = setup_data(config)

    # 设置模型
    print("设置模型...")
    model = setup_model(config)

    # 打印模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")

    # 训练模式
    if args.mode in ['train', 'continual']:
        # 创建训练器
        trainer = ContinualLearningTrainer(model, config, device)

        # 恢复checkpoint
        if args.resume:
            print(f"恢复训练: {args.resume}")
            trainer.load_checkpoint(args.resume)

        # 训练
        if args.mode == 'continual':
            print("开始持续学习...")
            results = trainer.train_continual(train_loaders, val_loaders)

            # 保存结果
            import json
            with open(os.path.join(args.save_dir, 'continual_results.json'), 'w') as f:
                json.dump(results, f, indent=2)

        else:
            # 单域训练（传统方式）
            domain = config['data']['source_domains'][0]
            results = trainer.train_domain(
                domain, train_loaders[domain], val_loaders[domain],
                config['training']['epochs']
            )

    # 评估模式
    if args.mode in ['test', 'continual']:
        print("\n" + "=" * 60)
        print("在目标域上评估...")
        print("=" * 60)

        model.eval()

        # 评估每个目标域
        for domain, loader in target_loaders.items():
            print(f"\n评估域: {domain}")

            metrics = trainer._evaluate(loader)

            print_metrics(metrics, label_names=['健康', '内圈缺陷', '外圈缺陷'])

            # 保存结果
            np.save(os.path.join(args.save_dir, f'metrics_{domain}.npy'),
                    metrics)

    print(f"\n所有结果已保存到: {args.save_dir}")


if __name__ == '__main__':
    main()