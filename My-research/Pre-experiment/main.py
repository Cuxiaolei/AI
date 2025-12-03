import yaml
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import joblib
import time
import warnings
from tqdm import tqdm
from utils.data_loader import OttawaDataset
from utils.metrics import FSMetrics, DomainShiftAnalyzer
from models.feature_extractor import FeatureExtractor
from models.domain_aligner import CoralAligner, MMDAligner
from models.classifier import ClassifierFactory

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

        # 检查数据
        if not os.path.exists(self.config['DATA']['path']):
            print("❌ 错误: 数据集路径不存在!")
            print("   请确保数据在" + self.config['DATA']['path'] + "目录下")
            return

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
        """训练+验证（带详细进度显示）"""
        print("\n" + "=" * 60)
        print("【开始训练】留一域交叉验证...")

        n_domains = self.config['DATA']['n_domains']
        results = []

        # 留一域交叉验证（域级进度条）
        for target_idx in tqdm(range(n_domains), desc="域验证进度", ncols=80):
            source_domains = [i for i in range(n_domains) if i != target_idx]
            domain_info = self.dataset.domain_map[target_idx]

            print(f"\n{'=' * 60}")
            print(f"🎯 目标域 {target_idx}: {domain_info['health']}-{domain_info['speed']}")
            print(f"   源域数量: {len(source_domains)}")
            print(f"   开始时间: {time.strftime('%H:%M:%S')}")

            # 加载目标域数据
            print("   📂 加载目标域数据...")
            target_data = self.dataset.load_domain(target_idx)
            target_features = self.feature_extractor.extract_features(
                target_data['vibration']
            )
            print(f"   ✅ 目标域加载完成: {target_features.shape[0]}个样本")

            # 多episode测试（episode级进度条）
            episode_results = []
            episode_pbar = tqdm(
                range(self.config['FEW_SHOT']['n_episodes']),
                desc=f"Episode进度[域{target_idx}]",
                ncols=80,
                leave=False
            )

            for episode in episode_pbar:
                # 运行单个episode
                result = self._run_single_episode(
                    source_domains, target_features, target_data['labels']
                )
                episode_results.append(result)

                # 实时更新进度条后缀
                episode_pbar.set_postfix({
                    'acc': f"{result['metrics']['accuracy']:.3f}",
                    'f1': f"{result['metrics']['f1_macro']:.3f}"
                })

            # 汇总结果
            mean_acc = np.mean([r['metrics']['accuracy'] for r in episode_results])
            std_acc = np.std([r['metrics']['accuracy'] for r in episode_results])
            results.append({
                'target_domain': target_idx,
                'mean_acc': mean_acc,
                'std_acc': std_acc,
                'episodes': episode_results
            })

            print(f"\n   📊 域{target_idx}完成: {mean_acc:.4f} ± {std_acc:.4f}")
            print(f"   ⏱️  耗时: {time.strftime('%H:%M:%S')}")

        # 总体结果
        self._show_overall_results(results)
        return results

    def _run_single_episode(self, source_domains, target_features, target_labels):
        """运行单个episode（带详细步骤日志）"""
        start_time = time.time()

        # 1. 生成episode
        print(f"   🔄 生成episode...", end=" ")
        support_set, query_set, _ = self.dataset.generate_episode(
            source_domains, 0,  # target_idx占位符
            self.config['FEW_SHOT']['k_shot'],
            self.config['FEW_SHOT']['n_query']
        )
        print(f"✅ Support:{support_set['X'].shape}, Query:{query_set['X'].shape}")

        # 2. 特征提取
        print(f"   🔧 提取特征...", end=" ")
        support_features = self.feature_extractor.extract_features(support_set['X'])
        query_features = self.feature_extractor.extract_features(query_set['X'])
        print(f"✅ 完成 (支持集{support_features.shape}, 查询集{query_features.shape})")

        # 3. 域对齐
        if self.config['DOMAIN_ALIGN']['method'] == 'CORAL':
            print(f"   🎯 CORAL域对齐...", end=" ")
            support_aligned, target_aligned = self.domain_aligner.align(
                support_features, target_features
            )
            query_aligned, _ = self.domain_aligner.align(
                query_features, target_features
            )
            print(f"✅ 完成")
        else:
            print(f"   ⏭️  跳过域对齐")
            support_aligned = support_features
            query_aligned = query_features

        # 4. 分类器训练
        print(f"   🎓 训练分类器...", end=" ")
        classifier = ClassifierFactory.create_classifier(self.config)
        classifier.fit(support_aligned, support_set['y'])
        print(f"✅ {self.config['MODEL']['classifier']}训练完成")

        # 5. 评估
        print(f"   📏 评估查询集...", end=" ")
        eval_results = classifier.evaluate(query_aligned, query_set['y'])
        print(f"✅ Acc={eval_results['accuracy']:.3f}, F1={eval_results['f1_macro']:.3f}")

        # 6. 域对齐效果评估
        alignment_metrics = {}
        if self.domain_aligner:
            print(f"   📐 计算域对齐距离...", end=" ")
            alignment_metrics = self.metrics.compute_domain_alignment_metrics(
                support_aligned, target_aligned
            )
            print(f"✅ CORAL距离={alignment_metrics['coral_distance']:.2f}")

        episode_time = time.time() - start_time
        print(f"   ⏱️  Episode耗时: {episode_time:.2f}s")

        return {
            'accuracy': eval_results['accuracy'],
            'metrics': eval_results,
            'alignment': alignment_metrics,
            'time': episode_time
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

        # 保存最佳模型
        model_path = os.path.join(output_dir, 'final_classifier.pkl')
        self._save_best_model(model_path)

        # 保存详细结果
        self._save_detailed_results()

        # 可视化
        if self.config['OUTPUT']['visualize']:
            self._plot_results()

        # 保存配置
        config_save_path = os.path.join(output_dir, 'config.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        print("=" * 60)

    def _save_best_model(self, model_path):
        """保存最佳模型（基于验证集性能）"""
        if not hasattr(self, 'results') or not self.results:
            print("⚠️  无训练结果可保存")
            return

        # 使用最后一个域的最佳episode
        best_result = self.results[-1]
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
        """在测试文件上预测（带进度显示）"""
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
        print(f"   📦 加载模型: {model_path}")
        classifier = ClassifierFactory.create_classifier(self.config)
        classifier.load(model_path)

        # 使用源域准备对齐参考
        source_domains = [0, 1, 2]
        print(f"   🎯 使用源域 {source_domains} 作为对齐参考")

        # 对每个测试文件（文件级进度条）
        results = []
        file_pbar = tqdm(test_files, desc="预测进度", ncols=80)

        for test_file in file_pbar:
            file_path = os.path.join(self.config['DATA']['path'], test_file)
            if not os.path.exists(file_path):
                file_pbar.set_postfix({'status': '文件不存在'})
                continue

            # 更新进度条描述
            file_pbar.set_description(f"预测: {test_file}")

            # 预测
            result = self._predict_single_file(
                file_path, classifier, source_domains
            )
            result['file'] = test_file
            results.append(result)

            # 显示结果
            print(f"\n   📄 {test_file}")
            print(f"      预测: {result['prediction']}")
            print(f"      置信度: {result['confidence']:.4f}")

            # 更新进度条后缀
            file_pbar.set_postfix({
                'pred': result['prediction'][:2],
                'conf': f"{result['confidence']:.2f}"
            })

        # 保存预测结果
        self._save_prediction_results(results)

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

        # 生成episode来获取支持集（用于对齐）
        support_set, _, _ = self.dataset.generate_episode(
            source_domains, 0,
            self.config['FEW_SHOT']['k_shot'],
            5
        )
        support_features = self.feature_extractor.extract_features(support_set['X'])

        # 对齐
        if self.config['DOMAIN_ALIGN']['method'] == 'CORAL':
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

        total_start = time.time()

        # 1. 训练
        print(f"\n{'=' * 60}")
        print("⏱️  开始训练...")
        self.train_and_validate()

        # 2. 保存
        print(f"\n{'=' * 60}")
        print("💾 保存结果...")
        self.save_model_and_results()

        # 3. 预测
        print(f"\n{'=' * 60}")
        print("🔮 开始预测...")
        self.predict_on_test_files()

        total_time = time.time() - total_start
        print("\n" + "=" * 60)
        print(f"✅ 实验全部完成！总耗时: {total_time / 60:.2f}分钟")
        print("📁 请查看results目录")
        print("=" * 60)


def main():
    config_path = './configs/config.yaml'

    # 检查依赖
    try:
        import scipy, numpy, sklearn, pywt, matplotlib, yaml, joblib, tqdm
        print("✅ 所有依赖库已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("   请安装: pip install scipy numpy scikit-learn PyWavelets matplotlib pyyaml joblib tqdm")
        return

    # 运行完整流程
    pipeline = FSDGUnifiedPipeline(config_path)
    pipeline.run_full_pipeline()


if __name__ == '__main__':
    main()