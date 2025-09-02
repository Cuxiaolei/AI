import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator
import os
from datetime import datetime

# 设置字体和样式
plt.rcParams['font.family'] = ['Times New Roman', 'serif']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11  # 保持较小字体以节省空间
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['lines.markersize'] = 0

# 高质量输出参数
plt.rcParams['savefig.format'] = 'png'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1


def plot_miou_curve(models_data, num_epochs=100, save_path=None,
                    title="mIoU Convergence Curves"):
    """绘制mIoU曲线（确保图例完整显示）"""
    # 增大图表尺寸，特别是宽度和高度
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    # 8个清晰区分的颜色
    colors = [
        '#1f77b4', '#9467bd', '#2ca02c', '#7f7fff',
        '#ff7f0e', '#8c564b', '#e377c2', '#d62728',
    ]

    epochs = np.arange(1, num_epochs + 1)

    # 遍历模型绘制曲线
    for i, (model_name, metrics) in enumerate(models_data.items()):
        color = colors[i % len(colors)]

        # 第5个(索引4)和第8个(索引7)颜色加重，其他变浅色
        if i == 4 or i == 7:
            # 加重：更粗的线条和不透明
            plt.plot(epochs, metrics['miou'] * 100, label=model_name,
                     color=color, linestyle='-', linewidth=1.5, alpha=1.0)
        else:
            # 浅色：较细的线条和半透明
            plt.plot(epochs, metrics['miou'] * 100, label=model_name,
                     color=color, linestyle='-', linewidth=1.0, alpha=0.8)

    # 设置图表属性
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('mIoU (%)')
    plt.xlim(0, num_epochs)
    plt.ylim(0, 100)

    # 设置刻度
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    # 仅保留左下坐标轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.grid(False)

    # 调整子图参数，为右下角图例留出更多空间
    plt.subplots_adjust(right=0.75, bottom=0.2)

    # 图例设置：右下角，两列，确保不被裁剪
    plt.legend(
        loc='lower right',
        ncol=2,
        frameon=False,
        columnspacing=0.8,
        bbox_to_anchor=(1.0, 0.0)
    )

    # 保存图片时确保包含完整图例
    if save_path:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"miou_{timestamp}.png"
        full_path = os.path.join(save_path, file_name)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        plt.savefig(
            full_path,
            dpi=600,
            bbox_inches='tight',
            pad_inches=0.3,
            pil_kwargs=dict(quality=95)
        )
        print(f"mIoU图表已保存至: {full_path}")

    plt.show()


def plot_oa_curve(models_data, num_epochs=100, save_path=None,
                  title=""):
    """绘制OA曲线（确保图例完整显示）"""
    # 增大图表尺寸
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # 8个清晰区分的颜色
    colors = [
        '#1f77b4', '#9467bd', '#2ca02c', '#7f7fff',
        '#ff7f0e', '#8c564b', '#e377c2', '#d62728',
    ]

    epochs = np.arange(1, num_epochs + 1)

    # 遍历模型绘制曲线
    for i, (model_name, metrics) in enumerate(models_data.items()):
        color = colors[i % len(colors)]

        # 第5个(索引4)和第8个(索引7)颜色加重，其他变浅色
        if i == 4 or i == 7:
            # 加重：更粗的线条和不透明
            plt.plot(epochs, metrics['oa'] * 100, label=model_name,
                     color=color, linestyle='-', linewidth=1.5, alpha=1.0)
        else:
            # 浅色：较细的线条和半透明
            plt.plot(epochs, metrics['oa'] * 100, label=model_name,
                     color=color, linestyle='-', linewidth=1.0, alpha=0.8)

    # 设置图表属性
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('OA (%)')
    plt.xlim(0, num_epochs)
    plt.ylim(0, 100)

    # 设置刻度
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    # 仅保留左下坐标轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.grid(False)

    # 调整子图参数，为右下角图例留出更多空间
    plt.subplots_adjust(right=0.75, bottom=0.2)

    # 图例设置：右下角，两列
    plt.legend(
        loc='lower right',
        ncol=2,
        frameon=False,
        columnspacing=0.8,
        bbox_to_anchor=(1.0, 0.0)
    )

    # 保存图片
    if save_path:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"oa_{timestamp}.png"
        full_path = os.path.join(save_path, file_name)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        plt.savefig(
            full_path,
            dpi=600,
            bbox_inches='tight',
            pad_inches=0.3,
            pil_kwargs=dict(quality=95)
        )
        print(f"OA图表已保存至: {full_path}")

    plt.show()


def load_data_from_csv(file_path):
    """从CSV加载模型的数据"""
    df = pd.read_csv(file_path)

    models_data = {}
    model_names = []

    # 按顺序遍历列名，按出现顺序收集模型名称（不重复）
    for col in df.columns:
        if '_miou' in col:
            model_name = col.split('_')[0]
            if model_name not in model_names:
                model_names.append(model_name)

    # 检查模型数量
    if len(model_names) != 8:
        print(f"警告: 检测到{len(model_names)}个模型，而不是预期的8个")

    for model in model_names:
        miou_col = f"{model}_miou"
        oa_col = f"{model}_oa"

        if miou_col in df.columns and oa_col in df.columns:
            models_data[model] = {
                'miou': df[miou_col].values,
                'oa': df[oa_col].values
            }

    return models_data, len(df)


def main():
    csv_path = "../z_picture/train_metrics.csv"  # 替换为你的CSV路径
    models_data, num_epochs = load_data_from_csv(csv_path)

    # 绘制图表
    plot_miou_curve(
        models_data,
        num_epochs,
        save_path="../z_picture/",
        title=""
    )

    plot_oa_curve(
        models_data,
        num_epochs,
        save_path="../z_picture/",
        title=""
    )
if __name__ == "__main__":
    main()
