#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
禁用域对齐的诊断测试脚本
直接复制保存为 test_no_alignment.py 后运行
"""

import sys
import os
import yaml
import numpy as np
from utils.data_loader import OttawaDataset
from models.feature_extractor import FeatureExtractor
from models.classifier import ClassifierFactory
from utils.metrics import FSMetrics

# 配置参数（已修复：添加 sample_rate）
CONFIG = {
    'DATA': {
        'path': '/root/data/Ottawa_Bearing_Dataset',
        'window_size': 2048,
        'overlap': 0.5,
        'sample_rate': 200000  # ✅ 修复：添加采样率
    },
    'FEW_SHOT': {
        'k_shot': 5,
        'n_query': 15,
        'n_episodes': 10  # 仅测试10个episode加快速度
    },
    'MODEL': {
        'classifier': 'KNN',
        'n_neighbors': 1
    },
    'FEATURE': {
        'use_statistical': True,
        'use_spectral': True,
        'use_time_freq': True
    }
}


def main():
    print("=" * 60)
    print("🔍 禁用域对齐的诊断测试")
    print("=" * 60)

    # 初始化组件
    dataset = OttawaDataset(CONFIG['DATA']['path'], CONFIG)
    feature_extractor = FeatureExtractor(CONFIG)
    metrics = FSMetrics()

    # 留一域测试（目标域=0）
    source_domains = list(range(1, 12))
    target_domain = 0

    # 加载目标域（仅用于加载，不参与对齐）
    target_data = dataset.load_domain(target_domain)
    target_features = feature_extractor.extract_features(target_data['vibration'])
    target_labels = target_data['labels']

    # 运行10个episode
    episode_accs = []
    episode_f1s = []

    print(f"测试配置: K={CONFIG['FEW_SHOT']['k_shot']}, n_query={CONFIG['FEW_SHOT']['n_query']}")
    print(f"源域数量: {len(source_domains)}, 目标域: {target_domain}")
    print("-" * 60)

    for episode in range(CONFIG['FEW_SHOT']['n_episodes']):
        # 生成episode
        support_set, query_set, _ = dataset.generate_episode(
            source_domains, target_domain,
            CONFIG['FEW_SHOT']['k_shot'],
            CONFIG['FEW_SHOT']['n_query']
        )

        # 特征提取（不进行域对齐）
        support_features = feature_extractor.extract_features(support_set['X'])
        query_features = feature_extractor.extract_features(query_set['X'])

        # 训练分类器
        classifier = ClassifierFactory.create_classifier(CONFIG)
        classifier.fit(support_features, support_set['y'])

        # 评估
        eval_results = classifier.evaluate(query_features, query_set['y'])
        episode_accs.append(eval_results['accuracy'])
        episode_f1s.append(eval_results['f1_macro'])

        print(f"Episode {episode + 1:2d}: Acc={eval_results['accuracy']:.3f}, "
              f"F1={eval_results['f1_macro']:.3f}")

    # 统计结果
    mean_acc = np.mean(episode_accs)
    std_acc = np.std(episode_accs)
    mean_f1 = np.mean(episode_f1s)
    std_f1 = np.std(episode_f1s)

    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    print(f"平均准确率: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"平均F1分数: {mean_f1:.4f} ± {std_f1:.4f}")

    # 判断与建议
    if mean_acc > 0.6:
        print("\n✅ 测试通过！禁用对齐后性能提升")
        print("💡 建议: 将 config.yaml 中 DOMAIN_ALIGN.method 设为 'None'")
    elif mean_acc > 0.4:
        print("\n⚠️  性能中等，需进一步优化")
        print("💡 建议: 增大 k_shot 或 n_query，或尝试其他分类器")
    else:
        print("\n❌ 性能较差，需检查特征提取环节")
        print("💡 建议: 运行 test_features.py 测试不同特征组合")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()