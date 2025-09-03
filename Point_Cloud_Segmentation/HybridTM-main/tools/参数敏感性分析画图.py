import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 设置中文字体，确保中文正常显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def get_unique_filename(directory, base_name, suffix):
    """生成目录下不重复的文件名，避免覆盖已有文件"""
    file_path = os.path.join(directory, f"{base_name}{suffix}")
    counter = 1
    while os.path.exists(file_path):
        file_path = os.path.join(directory, f"{base_name}_{counter}{suffix}")
        counter += 1
    return file_path


def plot_sensitivity_analysis(csv_file_path):
    """绘制无顶部/右侧边框且无背景虚线的参数敏感性分析图"""
    try:
        # 1. 解析CSV文件路径
        csv_path = Path(csv_file_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV文件不存在：{csv_file_path}")
        csv_directory = str(csv_path.parent)
        csv_filename = csv_path.name

        # 2. 读取并校验数据
        df = pd.read_csv(csv_file_path)
        required_columns = ['angle_weight', 'Class_0_IOU(class_0)',
                           'Class_1_IOU(class_1)', 'Class_2_IOU(class_2)',
                           'Scene_mIoU']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV缺少必要列：{', '.join(missing_cols)}")

        # 3. 创建图表与轴
        fig, ax = plt.subplots(figsize=(10, 6))

        # 隐藏顶部和右侧边框及刻度
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(top=False, right=False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

        # 4. 绘制折线
        ax.plot(df['angle_weight'], df['Class_0_IOU(class_0)'],
                marker='o', color='#1f77b4', label='Tower IOU', linewidth=2, markersize=6, linestyle='--')
        ax.plot(df['angle_weight'], df['Class_1_IOU(class_1)'],
                marker='s', color='#ff7f0e', label='Background IOU', linewidth=2, markersize=6, linestyle='--')
        ax.plot(df['angle_weight'], df['Class_2_IOU(class_2)'],
                marker='^', color='#2ca02c', label='Conductor IOU', linewidth=2, markersize=6, linestyle='--')
        ax.plot(df['angle_weight'], df['Scene_mIoU'],
                marker='D', color='#ff0000', label='mIoU', linewidth=2, markersize=6)

        # 5. 坐标轴设置
        y_max = df[required_columns[1:]].max().max()
        ax.set_ylim(0.8, y_max + 0.01)
        ax.set_xlim(df['angle_weight'].min() - 0.05, df['angle_weight'].max() + 0.05)

        # 6. 图表标签与美化（已移除网格线）
        ax.set_xlabel('angle_weight', fontsize=12, fontweight='bold')
        ax.set_ylabel('IOU Value', fontsize=12, fontweight='bold')
        ax.set_title('', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right', frameon=False)  # 图例无边框

        # 7. 保存图片
        unique_img_path = get_unique_filename(csv_directory, "sensitivity_analysis", ".png")
        plt.tight_layout()
        plt.savefig(unique_img_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ CSV文件：{csv_filename}（目录：{csv_directory}）")
        print(f"✅ 无网格图表已保存为：{unique_img_path}")

        # 显示图表
        plt.show()

    except Exception as e:
        print(f"❌ 绘图失败：{str(e)}")


if __name__ == "__main__":
    # 填写你的CSV文件路径
    csv_file = "../z_canshu/test_results.csv"
    plot_sensitivity_analysis(csv_file)
