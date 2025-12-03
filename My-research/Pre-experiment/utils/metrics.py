import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, classification_report
)
from scipy.spatial.distance import cdist


class FSMetrics:
    """
    小样本域泛化专用评估指标
    支持多维度评估：分类性能 + 域对齐效果
    """

    def __init__(self, n_domains=12, n_classes=3):
        self.n_domains = n_domains
        self.n_classes = n_classes
        self.health_map = {0: '健康', 1: '内圈缺陷', 2: '外圈缺陷'}

    def compute_classification_metrics(self, y_true, y_pred, y_proba=None):
        """
        计算分类指标
        Returns: dict with acc, precision, recall, f1, per-class metrics
        """
        # 基础指标
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )

        # 每类指标
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
        )

        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
        }

        # 如果有概率输出，计算AUC-ROC
        if y_proba is not None:
            try:
                # 多分类AUC需要one-hot编码
                metrics['auc_roc'] = roc_auc_score(
                    y_true, y_proba, multi_class='ovo', average='macro'
                )
            except:
                metrics['auc_roc'] = None

        return metrics

    def compute_confusion_matrix(self, y_true, y_pred, normalize='true'):
        """
        计算混淆矩阵
        normalize: 'true' (按行), 'pred' (按列), None (原始计数)
        """
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        if normalize == 'true':
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        elif normalize == 'pred':
            cm = cm.astype('float') / cm.sum(axis=0)
        return cm

    def compute_domain_alignment_metrics(self, source_feat, target_feat):
        """
        计算域对齐效果
        Returns: CORAL距离, MMD距离, 均值差异
        """
        metrics = {}

        # CORAL距离（协方差差异）
        cov_s = np.cov(source_feat, rowvar=False)
        cov_t = np.cov(target_feat, rowvar=False)
        coral_dist = np.linalg.norm(cov_s - cov_t, 'fro')
        metrics['coral_distance'] = coral_dist

        # 最大均值差异（MMD）近似
        mean_s = np.mean(source_feat, axis=0)
        mean_t = np.mean(target_feat, axis=0)
        mmd_dist = np.linalg.norm(mean_s - mean_t)
        metrics['mmd_distance'] = mmd_dist

        # 特征分布重叠度（基于类内距离）
        intra_domain_dist = np.mean(cdist(source_feat, source_feat, 'euclidean'))
        inter_domain_dist = np.mean(cdist(source_feat, target_feat, 'euclidean'))
        metrics['domain_overlap_ratio'] = intra_domain_dist / (inter_domain_dist + 1e-8)

        return metrics

    def generate_report(self, results_dict, save_path=None):
        """
        生成详细评估报告
        results_dict: 包含所有episode的结果
        """
        import pandas as pd

        # 汇总所有episode
        all_accuracies = []
        all_precisions = []
        all_recalls = []
        all_f1s = []

        for result in results_dict:
            all_accuracies.append(result['metrics']['accuracy'])
            all_precisions.append(result['metrics']['precision_macro'])
            all_recalls.append(result['metrics']['recall_macro'])
            all_f1s.append(result['metrics']['f1_macro'])

        # 构建报告
        report = {
            '总体性能': {
                '准确率': f"{np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}",
                '精确率': f"{np.mean(all_precisions):.4f} ± {np.std(all_precisions):.4f}",
                '召回率': f"{np.mean(all_recalls):.4f} ± {np.std(all_recalls):.4f}",
                'F1分数': f"{np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}",
            },
            '每类F1分数': {},
            '样本数量': len(all_accuracies)
        }

        # 计算每类平均F1
        for class_idx, class_name in self.health_map.items():
            class_f1s = [r['metrics']['f1_per_class'][class_idx] for r in results_dict]
            report['每类F1分数'][class_name] = \
                f"{np.mean(class_f1s):.4f} ± {np.std(class_f1s):.4f}"

        # 保存到文件
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("  详细性能评估报告\n")
                f.write("=" * 60 + "\n\n")
                for key, value in report.items():
                    if isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for sub_key, sub_value in value.items():
                            f.write(f"  {sub_key}: {sub_value}\n")
                    else:
                        f.write(f"{key}: {value}\n")

        return report


class DomainShiftAnalyzer:
    """
    域偏移分析器
    量化源域到目标域的分布差异
    """

    def __init__(self, feature_extractor):
        self.extractor = feature_extractor

    def analyze_shift(self, source_data_dict, target_data):
        """
        分析源域到目标域的偏移
        source_data_dict: {domain_idx: data_dict}
        target_data: data_dict
        Returns: 各域对齐难度评分
        """
        shift_scores = {}

        target_features = self.extractor.extract_features(
            target_data['vibration']
        )

        for src_idx, src_data in source_data_dict.items():
            src_features = self.extractor.extract_features(
                src_data['vibration']
            )

            # 计算多维度距离
            # 1. 均值差异
            mean_shift = np.linalg.norm(
                np.mean(src_features, axis=0) - np.mean(target_features, axis=0)
            )

            # 2. 方差差异
            std_shift = np.linalg.norm(
                np.std(src_features, axis=0) - np.std(target_features, axis=0)
            )

            # 3. 最大均值差异（MMD）
            mmd_score = self._mmd_rbf(src_features, target_features)

            shift_scores[src_idx] = {
                'mean_shift': mean_shift,
                'std_shift': std_shift,
                'mmd_score': mmd_score,
                'overall_shift': mean_shift + std_shift + mmd_score * 0.1
            }

        # 按总体偏移排序
        sorted_shifts = sorted(
            shift_scores.items(),
            key=lambda x: x[1]['overall_shift'],
            reverse=True
        )

        return sorted_shifts

    def _mmd_rbf(self, X, Y, gamma=1.0):
        """RBF核MMD计算"""
        K_xx = self._rbf_kernel(X, X, gamma)
        K_yy = self._rbf_kernel(Y, Y, gamma)
        K_xy = self._rbf_kernel(X, Y, gamma)

        mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        return mmd

    def _rbf_kernel(self, X1, X2, gamma=1.0):
        """RBF核函数"""
        dists = cdist(X1, X2, 'sqeuclidean')
        return np.exp(-gamma * dists)