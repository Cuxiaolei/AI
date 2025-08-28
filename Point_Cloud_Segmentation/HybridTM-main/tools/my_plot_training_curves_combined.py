import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator
import os
from datetime import datetime

# 设置字体和样式
plt.rcParams['font.family'] = ['Times New Roman','serif']
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


def generate_sample_data(num_epochs=100, num_models=7):
    """生成7个模型的示例数据"""
    np.random.seed(42)

    models_data = {}

    for i in range(num_models):
        model_name = f"Model {i + 1}"

        # 生成miou数据
        base_miou = 0.45 + (i * 0.03)
        final_miou = 0.8 + (i * 0.02)
        if final_miou > 0.95:
            final_miou = 0.95
        miou = np.linspace(base_miou, final_miou, num_epochs)
        miou += np.random.normal(0, 0.007, num_epochs).cumsum() * 0.1
        miou = np.clip(miou, 0, 1)

        # 生成oa数据
        base_oa = base_miou + 0.05 + (i * 0.02)
        final_oa = final_miou + 0.05 + (i * 0.01)
        if final_oa > 0.98:
            final_oa = 0.98
        oa = np.linspace(base_oa, final_oa, num_epochs)
        oa += np.random.normal(0, 0.005, num_epochs).cumsum() * 0.1
        oa = np.clip(oa, 0, 1)

        models_data[model_name] = {
           'miou': miou,
            'oa': oa
        }

    return models_data


def plot_miou_curve(models_data, num_epochs=100, save_path=None,
                    title="mIoU Convergence Curves"):
    """绘制mIoU曲线（确保图例完整显示）"""
    # 增大图表尺寸，特别是宽度和高度
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    # 7个清晰区分的颜色
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2'
    ]

    epochs = np.arange(1, num_epochs + 1)

    # 遍历模型绘制曲线
    for i, (model_name, metrics) in enumerate(models_data.items()):
        color = colors[i % len(colors)]
        plt.plot(epochs, metrics['miou'] * 100, label=model_name, color=color, linestyle='-')

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
    plt.subplots_adjust(right=0.75, bottom=0.2)  # 关键修改：减小右和底部边距

    # 图例设置：右下角，两列，确保不被裁剪
    plt.legend(
        loc='lower right',
        ncol=2,
        frameon=False,
        columnspacing=0.8,
        bbox_to_anchor=(1.0, 0.0)  # 精确控制图例位置
    )

    # 保存图片时确保包含完整图例
    if save_path:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"miou_{timestamp}.png"
        full_path = os.path.join(save_path, file_name)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        # 使用bbox_inches确保图例不被裁剪
        plt.savefig(
            full_path,
            dpi=600,
            bbox_inches='tight',
            pad_inches=0.3,  # 增加边距确保图例完整
            pil_kwargs=dict(quality=95)
        )
        print(f"mIoU图表已保存至: {full_path}")

    plt.show()


def plot_oa_curve(models_data, num_epochs=100, save_path=None,
                  title=""):
    """绘制OA曲线（确保图例完整显示）"""
    # 增大图表尺寸
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    # 与mIoU保持一致的颜色
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2'
    ]

    epochs = np.arange(1, num_epochs + 1)

    # 遍历模型绘制曲线
    for i, (model_name, metrics) in enumerate(models_data.items()):
        color = colors[i % len(colors)]
        plt.plot(epochs, metrics['oa'] * 100, label=model_name, color=color, linestyle='-')

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
    """从CSV加载7个模型的数据（按顺序加载）"""
    df = pd.read_csv(file_path)

    models_data = {}
    model_names = []

    # 按顺序遍历列名，按出现顺序收集模型名称（不重复）
    for col in df.columns:
        if '_miou' in col:
            model_name = col.split('_')[0]
            if model_name not in model_names:
                model_names.append(model_name)

    # 确保读取到7个模型
    if len(model_names) != 7:
        print(f"警告: 检测到{len(model_names)}个模型，而不是预期的7个")

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