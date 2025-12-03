#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试不同特征组合效果的脚本
直接保存为 test_features.py 后运行
"""

import sys
import os
import yaml
import numpy as np
from utils.data_loader import OttawaDataset
from models.feature_extractor import FeatureExtractor
from models.classifier import ClassifierFactory

# 基础配置
BASE_CONFIG = {
    'DATA': {
        'path': '/root/data/Ottawa_Bearing_Dataset',
        'window_size': 2048,
        'overlap': 0.5,
        'sample_rate': 200000
    },
    'FEW_SHOT': {
        'k_shot': 5,
        'n_query': 15,
        'n_episodes': 5  # 每个特征组合只测5个episode加快速度
    },
    'MODEL': {
        'classifier': 'KNN',
        'n_neighbors': 1
    }
}

# 四种特征组合测试
FEATURE_CONFIGS = [
    {
        'name': '仅时域统计',
        'use_statistical': True,
        'use_spectral': False,
        'use_time_freq': False,
        'n_features': 10
    },
    {
        'name': '仅频域',
        'use_statistical': False,
        'use_spectral': True,
        'use_time_freq': False,
        'n_features': 8
    },
    {
        'name': '仅时频(小波包)',
        'use_statistical': False,
        'use_spectral': False,
        'use_time_freq': True,
        'n_features': 8
    },
    {
        'name': '全部特征',
        'use_statistical': True,
        'use_spectral': True,
        'use_time_freq': True,
        'n_features': 26
    }
]


def test_feature_combination(feature_cfg, config):
    """测试单一特征组合"""
    print(f"\n{'-' * 60}")
    print(f"测试: {feature_cfg['name']}")
    print(f"特征维度: {feature_cfg['n_features']}")
    print('-' * 60)

    # 更新配置
    config['FEATURE'] = feature_cfg

    # 初始化组件
    dataset = OttawaDataset(config['DATA']['path'], config)
    feature_extractor = FeatureExtractor(config)

    # 留一域测试
    source_domains = list(range(1, 12))
    target_domain = 0

    # 运行5个episode
    episode_accs = []

    for episode in range(config['FEW_SHOT']['n_episodes']):
        support_set, query_set, _ = dataset.generate_episode(
            source_domains, target_domain,
            config['FEW_SHOT']['k_shot'],
            config['FEW_SHOT']['n_query']
        )

        # 特征提取
        support_features = feature_extractor.extract_features(support_set['X'])
        query_features = feature_extractor.extract_features(query_set['X'])

        # 训练分类器
        classifier = ClassifierFactory.create_classifier(config)
        classifier.fit(support_features, support_set['y'])

        # 评估
        eval_results = classifier.evaluate(query_features, query_set['y'])
        episode_accs.append(eval_results['accuracy'])

        print(f"  Episode {episode + 1}: Acc={eval_results['accuracy']:.3f}")

    mean_acc = np.mean(episode_accs)
    std_acc = np.std(episode_accs)

    print(f"  平均性能: {mean_acc:.4f} ± {std_acc:.4f}")

    return {
        'name': feature_cfg['name'],
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'n_features': feature_cfg['n_features']
    }


def main():
    print("=" * 60)
    print("🔍 特征组合对比测试")
    print("=" * 60)
    print(f"测试组合数: {len(FEATURE_CONFIGS)}")
    print(f"每组合episodes: {BASE_CONFIG['FEW_SHOT']['n_episodes']}")
    print("-" * 60)

    results = []

    # 测试每种特征组合
    for feature_cfg in FEATURE_CONFIGS:
        result = test_feature_combination(feature_cfg, BASE_CONFIG.copy())
        results.append(result)

    # 排序并输出最佳
    results.sort(key=lambda x: x['mean_acc'], reverse=True)

    print("\n" + "=" * 60)
    print("📊 特征组合性能排名")
    print("=" * 60)

    for i, r in enumerate(results, 1):
        print(f"{i}. {r['name']:20s} | "
              f"Acc={r['mean_acc']:.4f}±{r['std_acc']:.4f} | "
              f"维度={r['n_features']:2d}")

    # 最佳建议
    best = results[0]
    print("\n" + "=" * 60)
    print("💡 最佳特征组合")
    print("=" * 60)
    print(f"推荐配置: {best['name']}")
    print(f"平均准确率: {best['mean_acc']:.4f}")
    print(f"特征维度: {best['n_features']}")

    if best['name'] != '全部特征':
        print("\n💡 配置修改建议:")
        print(f"  修改 config.yaml 中 FEATURE 部分:")
        for key in ['use_statistical', 'use_spectral', 'use_time_freq']:
            value = FEATURE_CONFIGS[results.index(best)].get(key, False)
            print(f"    {key}: {value}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()