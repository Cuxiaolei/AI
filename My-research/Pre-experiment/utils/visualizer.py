import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os
from typing import List, Optional


def plot_confusion_matrix(y_true: List, y_pred: List,
                          classes: List[str] = None,
                          save_path: str = None,
                          title: str = 'Confusion Matrix',
                          normalize: bool = False):
    """
    绘制混淆矩阵

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        classes: 类别名称
        save_path: 保存路径
        title: 标题
        normalize: 是否归一化
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=classes, yticklabels=classes,
                cbar_kws={"shrink": 0.8})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_training_history(history: dict,
                          save_dir: str = './outputs',
                          show_plot: bool = False):
    """
    绘制训练历史曲线

    Args:
        history: 历史记录字典，包含 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_dir: 保存目录
        show_plot: 是否显示图像
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_ylim([0, 1])
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存
    save_path = os.path.join(save_dir, 'training_history.png')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练历史图已保存到: {save_path}")

    if show_plot:
        plt.show()
    plt.close()


def plot_domain_performance(results: dict,
                            save_path: str = None,
                            show_plot: bool = False):
    """
    绘制各域性能对比柱状图

    Args:
        results: 结果字典，格式 {domain: {'best_accuracy': float, ...}}
        save_path: 保存路径
        show_plot: 是否显示图像
    """
    domains = list(results.keys())
    accuracies = [results[dom]['best_accuracy'] for dom in domains]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(domains, accuracies,
                   color='skyblue', edgecolor='black', linewidth=1.5)

    plt.title('Domain Performance Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Domains', fontsize=12)
    plt.ylabel('Best Accuracy', fontsize=12)
    plt.ylim(0, 1)

    # 在柱子上显示数值
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"域性能对比图已保存到: {save_path}")

    if show_plot:
        plt.show()
    plt.close()


def plot_feature_distribution(features: np.ndarray,
                              labels: np.ndarray,
                              domains: Optional[np.ndarray] = None,
                              save_path: str = None,
                              method: str = 'tsne',
                              show_plot: bool = False):
    """
    可视化特征分布（t-SNE或PCA降维）

    Args:
        features: 特征数组 [N, feature_dim]
        labels: 标签数组 [N]
        domains: 域标签数组 [N]（可选）
        save_path: 保存路径
        method: 降维方法 'tsne' 或 'pca'
        show_plot: 是否显示图像
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    print(f"正在使用 {method.upper()} 降维可视化 {features.shape[0]} 个样本...")

    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 降维
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) // 4))
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError("method must be 'tsne' or 'pca'")

    features_2d = reducer.fit_transform(features_scaled)

    plt.figure(figsize=(10, 8))

    # 按标签着色
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                    c=[colors[i]], label=f'Class {label}', alpha=0.7, s=30)

    plt.title(f'Feature Distribution ({method.upper()})', fontsize=16, fontweight='bold')
    plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
    plt.legend(loc='best')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征分布图已保存到: {save_path}")

    if show_plot:
        plt.show()
    plt.close()


def plot_continual_learning_curve(domain_results: dict,
                                  save_path: str = None,
                                  show_plot: bool = False):
    """
    绘制持续学习过程中的性能变化曲线

    Args:
        domain_results: 持续学习结果字典
        save_path: 保存路径
        show_plot: 是否显示图像
    """
    domains = list(domain_results.keys())
    accuracies = [domain_results[dom]['best_accuracy'] for dom in domains]

    # 绘制学习曲线
    plt.figure(figsize=(12, 6))

    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(domains, accuracies, 'b-o', linewidth=2, markersize=8)
    plt.title('Continual Learning Performance', fontsize=14, fontweight='bold')
    plt.xlabel('Training Domain', fontsize=12)
    plt.ylabel('Best Accuracy', fontsize=12)
    plt.ylim(0.5, 1.0)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')

    # 遗忘程度
    if any('forgetting' in results for results in domain_results.values()):
        forgettings = [domain_results[dom].get('forgetting', 0) for dom in domains]

        plt.subplot(1, 2, 2)
        plt.bar(domains, forgettings, color='orange', alpha=0.7)
        plt.title('Catastrophic Forgetting', fontsize=14, fontweight='bold')
        plt.xlabel('Domain', fontsize=12)
        plt.ylabel('Forgetting', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"持续学习曲线已保存到: {save_path}")

    if show_plot:
        plt.show()
    plt.close()


def save_feature_visualization_data(features: np.ndarray,
                                    labels: np.ndarray,
                                    domains: np.ndarray,
                                    save_dir: str = './outputs'):
    """
    保存特征可视化数据（用于后续分析）

    Args:
        features: 特征数组
        labels: 标签数组
        domains: 域数组
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, 'features.npy'), features)
    np.save(os.path.join(save_dir, 'labels.npy'), labels)
    np.save(os.path.join(save_dir, 'domains.npy'), domains)

    print(f"特征可视化数据已保存到: {save_dir}/{{features,labels,domains}}.npy")