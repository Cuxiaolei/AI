import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os


def plot_single_confusion_matrix(csv_file, title=None, figsize=(10, 8),
                                 cmap=None, annot_size=12, fmt='.4f', auto_save=True):
    """绘制单个混淆矩阵"""
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file, index_col=0)

        # 提取类别名称（去掉括号里的内容）
        classes = [label.split('(')[0] for label in df.index]

        # 提取数据值
        data = df.values

        # 设置默认颜色映射
        if cmap is None:
            cmap = LinearSegmentedColormap.from_list(
                'custom_cmap', ['#ffffff', '#3498db', '#e74c3c'], N=100)

        # 创建图形和轴
        plt.figure(figsize=figsize)
        ax = plt.gca()

        # 绘制热图
        sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                    xticklabels=classes, yticklabels=classes,
                    annot_kws={"size": annot_size}, cbar=True)

        # 设置标题和标签
        if title is None:
            # 从文件名提取标题
            title = os.path.splitext(os.path.basename(csv_file))[0]
        plt.title(f"Confusion Matrix - {title}", fontsize=16)
        plt.xlabel('Predicted Class', fontsize=14)
        plt.ylabel('Actual Class', fontsize=14)

        # 调整刻度标签大小
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12, rotation=0)

        # 调整布局
        plt.tight_layout()

        # 自动保存图片
        if auto_save:
            # 获取文件名（不含扩展名）
            file_name = os.path.splitext(csv_file)[0]
            # 保存图片
            save_path = f"{file_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存为: {save_path}")

        # 显示图形
        plt.show()
        # 关闭当前图形，释放内存
        plt.close()

    except Exception as e:
        print(f"处理文件 {csv_file} 时出错: {str(e)}")


def batch_plot_confusion_matrices(file_dir, file_names, ext=".csv"):
    """批量处理多个文件"""
    # 遍历所有文件名
    for name in file_names:
        # 构建完整文件路径
        file_path = os.path.join(file_dir, f"{name}{ext}")

        # 检查文件是否存在
        if os.path.exists(file_path):
            print(f"正在处理: {file_path}")
            plot_single_confusion_matrix(file_path)
        else:
            print(f"文件不存在: {file_path}")


if __name__ == "__main__":
    # 设置文件所在目录（请根据实际情况修改）
    # 例如: r"D:\桌面1\训练结果\scannet\混淆矩阵"
    file_directory = r"D:\桌面1\训练结果\scannet\混淆矩阵"

    # 要处理的文件名列表（不带扩展名）
    file_names = [
        "MinkUNet",
        "SPUNet",
        "Swin3D",
        "StratTrans",
        "OACNNs",
        "HybridTM",
        "PT-V3",
        "OURS"
    ]

    # 批量绘制混淆矩阵
    batch_plot_confusion_matrices(file_directory, file_names)
