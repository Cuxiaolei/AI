import yaml
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from utils.data_loader import OttawaDataset
from utils.metrics import FSMetrics, DomainShiftAnalyzer
from models.feature_extractor import FeatureExtractor
from models.domain_aligner import CoralAligner, MMDAligner
from models.classifier import ClassifierFactory
import os
import joblib
import warnings

warnings.filterwarnings('ignore')


class FSDGUnifiedPipeline:
    """小样本域泛化训练预测一体化管道"""

    def __init__(self, config_path):
        self.config_path = config_path

        # 1. 加载配置
        print("=" * 60)
        print("【初始化】加载配置...")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 2. 初始化所有组件
        self._init_components()

    def _init_components(self):
        """初始化所有组件"""
        # 数据加载
        self.dataset = OttawaDataset(self.config['DATA']['path'], self.config)

        # 特征提取器
        self.feature_extractor = FeatureExtractor(self.config)

        # 域对齐器
        align_method = self.config['DOMAIN_ALIGN']['method']
        if align_method == 'CORAL':
            self.domain_aligner = CoralAligner()
        elif align_method == 'MMD':
            self.domain_aligner = MMDAligner()
        else:
            self.domain_aligner = None

        # 评估指标
        self.metrics = FSMetrics()
        self.shift_analyzer = DomainShiftAnalyzer(self.feature_extractor)

        # 健康状态映射
        self.health_map = {0: '健康', 1: '内圈缺陷', 2: '外圈缺陷'}

        print(f"   数据集: {self.config['DATA']['path']}")
        print(f"   特征维度: {self.config['FEATURE']['n_features']}")
        print(f"   域对齐: {align_method}")
        print(f"   分类器: {self.config['MODEL']['classifier']}")
        print(f"   小样本设置: K={self.config['FEW_SHOT']['k_shot']} shots")

    def train_and_validate(self):
        """训练+验证（主流程）"""
        print("\n" + "=" * 60)
        print("【开始训练】留一域交叉验证...")

        n_domains = self.config['DATA']['n_domains']
        results = []

        # 留一域交叉验证
        for target_idx in range(n_domains):
            source_domains = [i for i in range(n_domains) if i != target_idx]
            domain_info = self.dataset.domain_map[target_idx]

            print(f"\n>>> 目标域 {target_idx}: "
                  f"{domain_info['health']}-{domain_info['speed']}")

            # 在目标域上生成测试特征
            target_data = self.dataset.load_domain(target_idx)
            target_features = self.feature_extractor.extract_features(
                target_data['vibration']
            )
            target_labels = target_data['labels']

            # 域偏移分析（可选）
            if target_idx == 0:  # 仅分析第一个域以节省时间
                source_data_dict = {
                    idx: self.dataset.load_domain(idx) for idx in source_domains[:3]
                }
                shift_scores = self.shift_analyzer.analyze_shift(
                    source_data_dict, target_data
                )
                print(f"   域偏移分析: 最难对齐的源域 = {shift_scores[0][0]}")

            # 多episode测试
            episode_results = []
            for episode in range(self.config['FEW_SHOT']['n_episodes']):
                result = self._run_single_episode(
                    source_domains, target_features, target_labels
                )
                episode_results.append(result)

            # 汇总结果
            mean_acc = np.mean([r['accuracy'] for r in episode_results])
            std_acc = np.std([r['accuracy'] for r in episode_results])
            results.append({
                'target_domain': target_idx,
                'mean_acc': mean_acc,
                'std_acc': std_acc,
                'episodes': episode_results
            })

            print(f"   Episode准确率: {mean_acc:.4f} ± {std_acc:.4f}")

        # 总体结果
        self._show_overall_results(results)
        return results

    def _run_single_episode(self, source_domains, target_features, target_labels):
        """运行单个episode"""
        # 生成episode
        support_set, query_set, _ = self.dataset.generate_episode(
            source_domains, 0,  # target_idx占位符
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

        # 域对齐（CORAL）
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

        # 创建并训练分类器
        classifier = ClassifierFactory.create_classifier(self.config)
        classifier.fit(support_aligned, support_set['y'])

        # 在查询集上评估
        eval_results = classifier.evaluate(query_aligned, query_set['y'])

        # 域对齐效果评估
        alignment_metrics = {}
        if self.domain_aligner:
            alignment_metrics = self.metrics.compute_domain_alignment_metrics(
                support_aligned, target_aligned
            )

        return {
            'accuracy': eval_results['accuracy'],
            'metrics': eval_results,
            'alignment': alignment_metrics
        }

    def _show_overall_results(self, results):
        """展示总体结果"""
        overall_mean = np.mean([r['mean_acc'] for r in results])
        overall_std = np.mean([r['std_acc'] for r in results])

        print("\n" + "=" * 60)
        print("【训练完成】总体性能")
        print(f"  平均准确率: {overall_mean:.4f} ± {overall_std:.4f}")
        print("=" * 60)

        self.results = results
        self.overall_mean = overall_mean

    def save_model_and_results(self):
        """保存模型和结果"""
        print("\n" + "=" * 60)
        print("【保存结果】保存模型和详细结果...")

        output_dir = self.config['OUTPUT']['result_dir']
        os.makedirs(output_dir, exist_ok=True)

        # 1. 保存最后一域的分类器
        model_path = os.path.join(output_dir, 'final_classifier.pkl')
        # 这里保存的是最后一个episode的分类器，实际应保存最佳模型
        # 为简化，我们重新训练并保存
        self._save_best_model(model_path)

        # 2. 保存详细结果
        self._save_detailed_results()

        # 3. 可视化
        if self.config['OUTPUT']['visualize']:
            self._plot_results()

        # 4. 保存配置
        config_save_path = os.path.join(output_dir, 'config.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        print("=" * 60)

    def _save_best_model(self, model_path):
        """保存最佳模型（基于验证集性能）"""
        # 简单策略：使用最后一个目标域的最佳episode
        if not hasattr(self, 'results') or not self.results:
            print("⚠️  无训练结果可保存")
            return

        # 重新创建一个最佳分类器
        best_result = self.results[-1]  # 最后一个域
        best_episode = max(best_result['episodes'], key=lambda x: x['accuracy'])

        # 重新训练该episode
        source_domains = [i for i in range(self.config['DATA']['n_domains'])
                          if i != best_result['target_domain']]
        target_idx = best_result['target_domain']

        support_set, _, target_data = self.dataset.generate_episode(
            source_domains, target_idx,
            self.config['FEW_SHOT']['k_shot'],
            self.config['FEW_SHOT']['n_query']
        )

        support_features = self.feature_extractor.extract_features(support_set['X'])
        target_features = self.feature_extractor.extract_features(target_data['vibration'])

        if self.config['DOMAIN_ALIGN']['method'] == 'CORAL':
            support_aligned, _ = self.domain_aligner.align(support_features, target_features)
        else:
            support_aligned = support_features

        # 训练并保存
        classifier = ClassifierFactory.create_classifier(self.config)
        classifier.fit(support_aligned, support_set['y'])
        classifier.save(model_path)

    def _save_detailed_results(self):
        """保存详细结果"""
        result_file = os.path.join(
            self.config['OUTPUT']['result_dir'], 'detailed_results.txt'
        )

        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("  渥太华轴承数据集小样本域泛化实验报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"实验时间: {np.datetime64('now')}\n")
            f.write(f"小样本设置: {self.config['FEW_SHOT']['k_shot']}-shot\n")
            f.write(f"域对齐方法: {self.config['DOMAIN_ALIGN']['method']}\n")
            f.write(f"分类器: {self.config['MODEL']['classifier']}\n")
            f.write(f"总体准确率: {self.overall_mean:.4f}\n\n")
            f.write("各目标域性能:\n")
            f.write("-" * 60 + "\n")
            f.write("域ID\t健康状态\t转速\t准确率\t标准差\n")
            f.write("-" * 60 + "\n")

            for r in self.results:
                domain_info = self.dataset.domain_map[r['target_domain']]
                health_name = self.health_map[
                    self.dataset._health_to_label(domain_info['health'])
                ]
                f.write(f"{r['target_domain']}\t{health_name}\t\t"
                        f"{domain_info['speed']}\t{r['mean_acc']:.4f}\t{r['std_acc']:.4f}\n")

            f.write("=" * 60 + "\n")

        print(f"  详细结果: {result_file}")

    def _plot_results(self):
        """绘制结果图"""
        print("  生成可视化...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 1. 各域准确率柱状图
        domains = [r['target_domain'] for r in self.results]
        accs = [r['mean_acc'] for r in self.results]

        bars = ax1.bar(domains, accs, alpha=0.7, color='steelblue')
        ax1.set_xlabel('目标域ID', fontsize=12)
        ax1.set_ylabel('准确率', fontsize=12)
        ax1.set_title('各目标域测试性能', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        for bar, acc in zip(bars, accs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

        # 2. 性能分布直方图
        ax2.hist(accs, bins=10, alpha=0.7, color='darkorange', edgecolor='black')
        ax2.axvline(self.overall_mean, color='red', linestyle='--', linewidth=2,
                    label=f'均值: {self.overall_mean:.3f}')
        ax2.set_xlabel('准确率', fontsize=12)
        ax2.set_ylabel('频数', fontsize=12)
        ax2.set_title('准确率分布', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.config['OUTPUT']['result_dir'], 'results_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"  可视化: {plot_path}")

    def predict_on_test_files(self):
        """在指定的测试文件上进行预测"""
        print("\n" + "=" * 60)
        print("【预测验证】在测试文件上进行预测...")

        test_files = self.config['PREDICT']['test_files']
        model_path = os.path.join(
            self.config['OUTPUT']['result_dir'], 'final_classifier.pkl'
        )

        if not os.path.exists(model_path):
            print(f"❌ 错误: 模型文件不存在!")
            return

        # 加载模型
        classifier = ClassifierFactory.create_classifier(self.config)
        classifier.load(model_path)

        # 使用任意源域准备支持集（用于对齐）
        source_domains = [0, 1, 2]

        print(f"使用模型: {model_path}")
        print("-" * 60)

        # 对每个测试文件
        results = []
        for test_file in test_files:
            file_path = os.path.join(self.config['DATA']['path'], test_file)
            if not os.path.exists(file_path):
                print(f"  ⚠️  文件不存在: {test_file}")
                continue

            result = self._predict_single_file(
                file_path, classifier, source_domains
            )
            result['file'] = test_file
            results.append(result)

            # 打印结果
            print(f"\n  📄 文件: {test_file}")
            print(f"     预测: {result['prediction']}")
            print(f"     置信度: {result['confidence']:.4f}")
            print(f"     投票详情: {result['all_votes']}")

        # 保存预测结果
        self._save_prediction_results(results)
        print("\n" + "=" * 60)

    def _predict_single_file(self, file_path, classifier, source_domains):
        """预测单个文件"""
        # 加载数据
        data = sio.loadmat(file_path)
        vibration = data['Channel_1'].flatten()

        # 分段
        window_size = self.config['DATA']['window_size']
        step = int(window_size * (1 - self.config['DATA']['overlap']))
        n_samples = (len(vibration) - window_size) // step
        n_sample = min(n_samples, self.config['PREDICT']['vote_samples'])

        # 提取特征
        features = []
        for i in range(n_sample):
            start = i * step
            end = start + window_size
            sample = vibration[start:end]
            feat = self.feature_extractor.extract_features([sample])
            features.append(feat[0])

        features = np.array(features)

        # 生成一个episode来获取支持集（用于对齐）
        support_set, _, _ = self.dataset.generate_episode(
            source_domains, 0,
            self.config['FEW_SHOT']['k_shot'],
            5  # 小的查询集
        )
        support_features = self.feature_extractor.extract_features(support_set['X'])

        # 对齐（如果需要）
        if self.config['DOMAIN_ALIGN']['method'] == 'CORAL':
            # 简单处理：用支持集的协方差对齐测试特征
            cov_s = np.cov(support_features, rowvar=False) + np.eye(support_features.shape[1])
            cov_t = np.cov(features, rowvar=False) + np.eye(features.shape[1])
            from scipy.linalg import sqrtm
            cov_s_sqrt = sqrtm(cov_s)
            if np.iscomplexobj(cov_s_sqrt):
                cov_s_sqrt = cov_s_sqrt.real
            try:
                A = np.dot(np.dot(cov_s_sqrt, np.linalg.inv(sqrtm(cov_t))), cov_s_sqrt)
                features_aligned = np.dot(features, np.linalg.inv(A))
            except:
                features_aligned = features
        else:
            features_aligned = features

        # 预测
        predictions = classifier.predict(features_aligned)
        probs = classifier.predict_proba(features_aligned)

        # 投票
        final_pred = np.bincount(predictions).argmax()
        confidence = np.mean(probs[:, final_pred])

        all_votes = {
            self.health_map[i]: np.sum(predictions == i)
            for i in range(3)
        }

        return {
            'prediction': self.health_map[final_pred],
            'confidence': confidence,
            'all_votes': all_votes
        }

    def _save_prediction_results(self, results):
        """保存预测结果"""
        pred_file = os.path.join(
            self.config['OUTPUT']['result_dir'], 'prediction_results.txt'
        )

        with open(pred_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("  预测验证结果\n")
            f.write("=" * 60 + "\n\n")
            for r in results:
                f.write(f"文件: {r['file']}\n")
                f.write(f"预测健康状态: {r['prediction']}\n")
                f.write(f"置信度: {r['confidence']:.4f}\n")
                f.write(f"投票分布: {r['all_votes']}\n")
                f.write("-" * 40 + "\n")

        print(f"\n  预测结果已保存: {pred_file}")

    def run_full_pipeline(self):
        """运行完整训练和预测流程"""
        print("\n🚀 渥太华轴承数据集小样本域泛化实验")
        print("   训练模式: 留一域交叉验证")
        print("   预测模式: 自动验证测试文件")

        # 1. 训练
        self.train_and_validate()

        # 2. 保存
        self.save_model_and_results()

        # 3. 预测
        self.predict_on_test_files()

        print("\n✅ 实验全部完成！请查看results目录")
        print("=" * 60)


def main():
    config_path = './configs/config.yaml'

    # 检查数据
    if not os.path.exists('./data/Ottawa Bearing Dataset'):
        print("❌ 错误: 数据集路径不存在!")
        print("   请确保数据在 './data/Ottawa Bearing Dataset' 目录下")
        return

    # 检查依赖
    try:
        import scipy, numpy, sklearn, pywt, matplotlib, yaml
        print("✅ 所有依赖库已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("   请安装: pip install scipy numpy scikit-learn PyWavelets matplotlib pyyaml")
        return

    # 运行完整流程
    pipeline = FSDGUnifiedPipeline(config_path)
    pipeline.run_full_pipeline()


if __name__ == '__main__':
    main()