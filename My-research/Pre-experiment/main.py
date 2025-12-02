import yaml
import numpy as np
from utils.data_loader import OttawaDataset
from models.feature_extractor import FeatureExtractor
from models.domain_aligner import CoralAligner, MMDAligner
from models.classifier import PrototypicalClassifier, SVMClassifier
import matplotlib.pyplot as plt
import os


class FSDGTrainer:
    """小样本域泛化训练器"""

    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 初始化组件
        self.dataset = OttawaDataset(self.config['DATA']['path'], self.config)
        self.feature_extractor = FeatureExtractor(self.config)

        # 域对齐方法
        align_method = self.config['DOMAIN_ALIGN']['method']
        if align_method == 'CORAL':
            self.domain_aligner = CoralAligner()
        elif align_method == 'MMD':
            self.domain_aligner = MMDAligner()
        else:
            self.domain_aligner = None

        # 分类器
        classifier_type = self.config['MODEL']['classifier']
        if classifier_type == 'KNN':
            self.classifier = PrototypicalClassifier(
                n_neighbors=self.config['MODEL']['n_neighbors']
            )
        elif classifier_type == 'SVM':
            self.classifier = SVMClassifier()

        # 结果保存
        self.results = {'train_loss': [], 'test_acc': []}

    def train_epoch(self, source_domains, target_domain):
        """单个episode训练"""
        # 生成episode
        support_set, query_set, target_data = self.dataset.generate_episode(
            source_domains, target_domain,
            self.config['FEW_SHOT']['k_shot'],
            self.config['FEW_SHOT']['n_query']
        )

        # 特征提取
        support_features = self.feature_extractor.extract_features(
            support_set['X']
        )
        query_features = self.feature_extractor.extract_features(
            query_set['X']
        )
        target_features = self.feature_extractor.extract_features(
            target_data['vibration']
        )

        # 域对齐
        if self.config['DOMAIN_ALIGN']['method'] == 'CORAL':
            support_aligned, target_aligned = self.domain_aligner.align(
                support_features, target_features
            )
            query_aligned, _ = self.domain_aligner.align(
                query_features, target_features
            )
        else:
            support_aligned = support_features
            query_aligned = query_features

        # 训练分类器
        self.classifier.fit(support_aligned, support_set['y'])

        # 在查询集上评估
        predictions = self.classifier.predict(query_aligned)
        accuracy = np.mean(predictions == query_set['y'])

        return accuracy

    def run_cross_domain_validation(self):
        """留一域交叉验证"""
        n_domains = self.config['DATA']['n_domains']
        results = []

        print("开始留一域交叉验证...")
        for target_idx in range(n_domains):
            source_domains = [i for i in range(n_domains) if i != target_idx]
            print(f"\n目标域: {self.dataset.domain_map[target_idx]}")

            # 在目标域上测试
            target_data = self.dataset.load_domain(target_idx)
            target_features = self.feature_extractor.extract_features(
                target_data['vibration']
            )

            episode_accs = []
            for episode in range(self.config['FEW_SHOT']['n_episodes']):
                # 生成episode并训练
                acc = self.train_epoch(source_domains, target_idx)
                episode_accs.append(acc)

            mean_acc = np.mean(episode_accs)
            std_acc = np.std(episode_accs)
            results.append({
                'target_domain': target_idx,
                'mean_acc': mean_acc,
                'std_acc': std_acc
            })

            print(f"  平均准确率: {mean_acc:.4f} ± {std_acc:.4f}")

        # 总体结果
        overall_mean = np.mean([r['mean_acc'] for r in results])
        overall_std = np.mean([r['std_acc'] for r in results])

        print(f"\n=== 总体性能 ===")
        print(f"平均准确率: {overall_mean:.4f} ± {overall_std:.4f}")

        return results, overall_mean

    def save_results(self, results, overall_mean):
        """保存结果和模型"""
        os.makedirs(self.config['OUTPUT']['result_dir'], exist_ok=True)

        # 保存结果
        result_file = os.path.join(
            self.config['OUTPUT']['result_dir'], 'results.txt'
        )
        with open(result_file, 'w') as f:
            f.write("渥太华轴承数据集小样本域泛化结果\n")
            f.write(f"总体准确率: {overall_mean:.4f}\n\n")
            for r in results:
                domain_info = self.dataset.domain_map[r['target_domain']]
                f.write(f"域{r['target_domain']} ({domain_info['health']}-{domain_info['speed']}): ")
                f.write(f"{r['mean_acc']:.4f} ± {r['std_acc']:.4f}\n")

        print(f"\n结果已保存至: {result_file}")

        # 可视化
        if self.config['OUTPUT']['visualize']:
            self._visualize_results(results)

    def _visualize_results(self, results):
        """可视化结果"""
        domains = [r['target_domain'] for r in results]
        accs = [r['mean_acc'] for r in results]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(domains, accs, alpha=0.7)
        ax.set_xlabel('目标域')
        ax.set_ylabel('准确率')
        ax.set_title('各目标域测试准确率')
        ax.set_ylim(0, 1)

        # 添加数值标签
        for bar, acc in zip(bars, accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{acc:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(
            self.config['OUTPUT']['result_dir'], 'accuracy_plot.png'
        ))
        plt.show()


def main():
    # 配置文件路径
    config_path = './configs/config.yaml'

    # 检查数据路径
    if not os.path.exists('./data/Ottawa Bearing Dataset'):
        print("错误: 数据集路径不存在!")
        print("请确保数据在 './data/Ottawa Bearing Dataset' 目录下")
        return

    # 运行实验
    trainer = FSDGTrainer(config_path)
    results, overall_mean = trainer.run_cross_domain_validation()
    trainer.save_results(results, overall_mean)

    print("\n实验完成!")


if __name__ == '__main__':
    main()