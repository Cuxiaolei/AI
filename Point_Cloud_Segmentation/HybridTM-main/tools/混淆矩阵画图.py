import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import pandas as pd
import os
from datetime import datetime


def load_and_transform_data(csv_path):
    """加载CSV数据并进行正确转换"""
    df = pd.read_csv(csv_path, index_col=0)

    original_actual_labels = df.index.tolist()  # 原实际类别（行）
    original_pred_labels = df.columns.tolist()   # 原预测类别（列）
    original_data = df.values

    # 数据转换：仅反转实际类别（行）顺序（保持原逻辑）
    transformed_actual_labels = original_actual_labels[::-1]
    transformed_pred_labels = original_pred_labels
    transformed_data = original_data[::-1, :]

    # 简化标签：只保留Class后的数字（如Class_0→"0"）
    simplified_actual_labels = [label.split("_")[1].split("(")[0] for label in transformed_actual_labels]
    simplified_pred_labels = [label.split("_")[1].split("(")[0] for label in transformed_pred_labels]

    return transformed_data, simplified_actual_labels, simplified_pred_labels


def create_3d_visualization(data, actual_labels, pred_labels, csv_filename):
    """创建3D混淆矩阵：调整x/y轴顺序 + 左上角柱子数值白色"""
    # 字体配置
    plt.rcParams["font.family"] = ["Arial", "SimHei", "WenQuanYi Micro Hei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10

    # 创建图形（保持原有尺寸）
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 获取类别数量（混淆矩阵为方阵，行/列数一致）
    n_classes = data.shape[0]

    # -------------------------- 1. 调整x/y轴顺序：交换True/Pred角色 --------------------------
    # 原x轴：Predicted Class → 新x轴：True Class；原y轴：True Class → 新y轴：Predicted Class
    # 同步交换轴标签、数据矩阵（转置）、网格坐标
    new_x_labels = actual_labels       # 新x轴标签：原True Class标签
    new_y_labels = pred_labels         # 新y轴标签：原Predicted Class标签
    adjusted_data = data.T             # 数据矩阵转置（确保True/Pred对应关系正确）

    # 柱形底部中心点坐标（轴顺序交换后，x/y中心仍基于类别数量，但对应标签变了）
    x_centers = np.arange(n_classes)   # 新x轴（True Class）中心点
    y_centers = np.arange(n_classes)   # 新y轴（Predicted Class）中心点

    # 生成网格坐标（基于新x/y轴）
    x_pos, y_pos = np.meshgrid(x_centers, y_centers)
    x_pos_flat = x_pos.flatten()       # 新x轴（True Class）扁平化坐标
    y_pos_flat = y_pos.flatten()       # 新y轴（Predicted Class）扁平化坐标
    z_pos_flat = np.zeros_like(x_pos_flat)  # 柱子底部z坐标（均为0）
    # ------------------------------------------------------------------------------------------

    # 柱形高度（调整后的数据）和绿色系颜色映射（保持原有风格）
    bar_heights = adjusted_data.flatten()
    norm = colors.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap('Greens')
    bar_colors = cmap(norm(bar_heights))

    # 绘制3D柱状图（宽度保持0.6，位置计算适配新轴）
    dx = dy = 0.6
    ax.bar3d(
        x_pos_flat - dx / 2,  # 新x轴起始位置（确保中心点对齐刻度）
        y_pos_flat - dy / 2,  # 新y轴起始位置（确保中心点对齐刻度）
        z_pos_flat,
        dx, dy, bar_heights,
        color=bar_colors,
        alpha=0.9,
        edgecolor='darkgreen',
        linewidth=0.6
    )

    # -------------------------- 2. 调整坐标轴标签与刻度（适配新轴顺序） --------------------------
    ax.set_xlabel('True Class', labelpad=15, fontsize=12)    # 新x轴标签：True Class
    ax.set_ylabel('Predicted Class', labelpad=15, fontsize=12)# 新y轴标签：Predicted Class
    ax.set_zlabel('Probability', labelpad=15, fontsize=12)    # Z轴不变

    # 新x/y轴刻度与标签（对应交换后的角色）
    ax.set_xticks(x_centers)
    ax.set_xticklabels(new_x_labels, rotation=0, ha='center', fontsize=10)  # 新x轴标签
    ax.set_yticks(y_centers)
    ax.set_yticklabels(new_y_labels, rotation=0, va='center', fontsize=10)  # 新y轴标签
    # ------------------------------------------------------------------------------------------

    # 调整坐标轴范围（适配新轴和柱子宽度）
    ax.set_xlim(-dx / 2, n_classes - dx / 2)
    ax.set_ylim(-dy / 2, n_classes - dy / 2)
    ax.set_zlim(0, 1.1)

    # -------------------------- 3. 初始角度左上角柱子数值显示为白色 --------------------------
    # 初始视角（elev=35, azim=45）下，"左上角柱子"定义：x最小（True Class最小索引）且y最大（Predicted Class最大索引）
    min_x = np.min(x_pos_flat)                  # 新x轴最小坐标（左上角x）
    min_y = np.max(y_pos_flat)                  # 新y轴最大坐标（左上角y）
    # 找到左上角柱子对应的索引（支持任意类别数量，非硬编码）
    top_left_idx = np.where((x_pos_flat == min_x) & (y_pos_flat == min_y))[0][0]
    # ------------------------------------------------------------------------------------------

    # 添加数值标注（核心：左上角柱子强制白色文字）
    for i in range(len(x_pos_flat)):
        x = x_pos_flat[i]
        y = y_pos_flat[i]
        z = bar_heights[i] + 0.02
        prob_value = bar_heights[i]

        # 文字颜色逻辑：1. 左上角柱子→白色；2. 其他柱子→原逻辑（概率>0.5白，否则黑）
        if i == top_left_idx:
            text_color = 'white'
        else:
            text_color = 'white' if prob_value > 0.5 else 'black'

        ax.text(
            x, y, z,
            f'{prob_value:.4f}',
            ha='center', va='bottom',
            fontsize=10,
            color=text_color,
            fontweight='bold'
        )

    # 颜色条（保持无标签，原有风格）
    colorbar_mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    colorbar_mappable.set_array(bar_heights)
    cbar = fig.colorbar(
        colorbar_mappable,
        ax=ax,
        shrink=0.7,
        aspect=15,
        pad=0.1
    )
    cbar.ax.tick_params(labelsize=9)

    # 初始视角（保持原有35°仰角、45°方位角，确保左上角柱子位置固定）
    ax.view_init(elev=35, azim=45)

    return fig


def save_figure(fig, csv_path):
    """保存图像（保持原有逻辑：CSV同目录+时间戳防覆盖）"""
    csv_dir = os.path.dirname(csv_path)
    csv_filename = os.path.basename(csv_path)
    csv_base_name = os.path.splitext(csv_filename)[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_name = f"{csv_base_name}_{timestamp}.png"
    img_save_path = os.path.join(csv_dir, img_name)

    # 保存高分辨率图片（300dpi适合论文，无底部标题则无需额外留边）
    fig.savefig(
        img_save_path,
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )

    return img_save_path


def main():
    """主函数（保持原有逻辑，仅路径需确认）"""
    # 设置CSV文件路径（请确认路径正确性）
    csv_path = "../z_hunxiao/minkunet.csv"

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found - {csv_path}")
        return

    try:
        print("Loading and transforming data...")
        data, actual_labels, pred_labels = load_and_transform_data(csv_path)

        csv_filename = os.path.basename(csv_path)

        print("Generating 3D confusion matrix...")
        fig = create_3d_visualization(data, actual_labels, pred_labels, csv_filename)

        print("Saving image...")
        save_path = save_figure(fig, csv_path)
        print(f"Image saved successfully to: {save_path}")

        plt.show()

    except Exception as e:
        print(f"Error during processing: {str(e)}")


if __name__ == "__main__":
    main()
