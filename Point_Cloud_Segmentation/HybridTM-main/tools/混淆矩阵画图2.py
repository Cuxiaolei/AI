import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os


def plot_single_confusion_matrix(csv_file, title=None, figsize=(10, 8),
                                 cmap=None, base_annot_size=24,
                                 cbar_font_size=24, fmt='.4f', auto_save=True):
    """
    绘制单个混淆矩阵，支持调整右侧颜色刻度尺的字体大小
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file, index_col=0)

        # 提取类别名称
        classes = [label.split('(')[0] for label in df.index]

        # 提取数据值
        data = df.values

        # 设置绿色系颜色映射（浅绿到深绿渐变）
        if cmap is None:
            cmap = LinearSegmentedColormap.from_list(
                'green_gradient',
                ['#f7fcF5', '#e5f5e0', '#c7e9c0', '#a1d99b',
                 '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b'],
                N=100)

        # 创建图形和轴
        plt.figure(figsize=figsize)
        ax = plt.gca()

        # 绘制热图并获取颜色条对象
        heatmap = sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                              xticklabels=classes, yticklabels=classes,
                              annot_kws={"size": base_annot_size},
                              cbar=True,
                              ax=ax)

        # 获取颜色条并调整其刻度标签字体大小
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=cbar_font_size)

        # 调整每个注释的颜色和大小
        for text in ax.texts:
            value = float(text.get_text())
            # 大于0.8的数值使用白色字体
            if value > 0.8:
                text.set_color('white')
            else:
                text.set_color('#003300')
            # 增大字体大小
            text.set_fontsize(base_annot_size)

        # 设置标题和标签
        if title is None:
            title = os.path.splitext(os.path.basename(csv_file))[0]
        # plt.title(f"Confusion Matrix - {title}", fontsize=16, color="#00441b")
        plt.xlabel('Predicted Class', fontsize=20, color="#00441b")
        plt.ylabel('Actual Class', fontsize=20, color="#00441b")

        # 调整刻度标签
        plt.xticks(fontsize=24, color="#00441b")
        plt.yticks(fontsize=24, rotation=0, color="#00441b")

        # 调整布局
        plt.tight_layout()

        # 自动保存图片
        if auto_save:
            file_name = os.path.splitext(csv_file)[0]
            save_path = f"{file_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存为: {save_path}")

        # 显示图形
        plt.show()
        plt.close()

    except Exception as e:
        print(f"处理文件 {csv_file} 时出错: {str(e)}")


def batch_plot_confusion_matrices(file_dir, file_names, ext=".csv"):
    """批量处理多个文件"""
    for name in file_names:
        file_path = os.path.join(file_dir, f"{name}{ext}")
        if os.path.exists(file_path):
            print(f"正在处理: {file_path}")
            # 可在此处调整颜色刻度尺字体大小，默认为12
            plot_single_confusion_matrix(file_path, cbar_font_size=16)
        else:
            print(f"文件不存在: {file_path}")


if __name__ == "__main__":
    # 设置文件所在目录
    file_directory = r"D:\桌面1\训练结果\scannet\混淆矩阵"

    # 要处理的文件名列表
    file_names = [
        "MinkUNet", "SPUNet", "Swin3D", "StratTrans",
        "OACNNs", "HybridTM", "PT-V3", "OURS"
    ]

    # 批量绘制混淆矩阵
    batch_plot_confusion_matrices(file_directory, file_names)
