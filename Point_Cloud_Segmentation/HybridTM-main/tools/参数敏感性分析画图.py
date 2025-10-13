import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
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
    """绘制参数敏感性分析图，减小图表宽度使整体更紧凑"""
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
                            'Scene_mIoU', 'Class_0_ACC(class_0)',
                            'Class_1_ACC(class_1)', 'Class_2_ACC(class_2)',
                            'Scene_OA']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV缺少必要列：{', '.join(missing_cols)}")

        # 定义类别样式
        class_styles = {
            'Class_0': ('#1f77b4', 'o', 'Tower'),  # 蓝色, 圆形
            'Class_1': ('#ff7f0e', 's', 'Background'),  # 橙色, 正方形
            'Class_2': ('#2ca02c', '^', 'Conductor'),  # 绿色, 三角形
        }
        # 总体指标使用红色
        overall_color = '#ff0000'  # 红色用于mIoU和Overall Accuracy
        overall_marker = 'D'  # 菱形标记用于总体指标

        # 3. 绘制IOU图表 - 减小宽度到8
        fig_iou, ax_iou = plt.subplots(figsize=(8, 7))  # 宽度从10减小到8

        # 隐藏顶部和右侧边框及刻度
        ax_iou.spines['top'].set_visible(False)
        ax_iou.spines['right'].set_visible(False)
        ax_iou.tick_params(top=False, right=False)
        ax_iou.spines['left'].set_linewidth(1.2)
        ax_iou.spines['bottom'].set_linewidth(1.2)

        # 横坐标设置（保持0.5-1.5范围，0.2间隔显示刻度）
        ax_iou.set_xticks(np.arange(0.5, 1.6, 0.1))
        ax_iou.set_xlim(0.5, 1.5)
        # 调整刻度标签字体大小，避免拥挤
        ax_iou.tick_params(axis='x', labelsize=10)

        # 绘制IOU折线
        ax_iou.plot(df['angle_weight'], df['Class_0_IOU(class_0)'],
                    marker=class_styles['Class_0'][1], color=class_styles['Class_0'][0],
                    label=f'{class_styles["Class_0"][2]} IOU', linewidth=2, markersize=6,
                    linestyle='--')

        ax_iou.plot(df['angle_weight'], df['Class_1_IOU(class_1)'],
                    marker=class_styles['Class_1'][1], color=class_styles['Class_1'][0],
                    label=f'{class_styles["Class_1"][2]} IOU', linewidth=2, markersize=6,
                    linestyle='--')

        ax_iou.plot(df['angle_weight'], df['Class_2_IOU(class_2)'],
                    marker=class_styles['Class_2'][1], color=class_styles['Class_2'][0],
                    label=f'{class_styles["Class_2"][2]} IOU', linewidth=2, markersize=6,
                    linestyle='--')

        ax_iou.plot(df['angle_weight'], df['Scene_mIoU'],
                    marker=overall_marker, color=overall_color,
                    label='mIoU', linewidth=2, markersize=6,
                    linestyle='-')

        # IOU坐标轴设置
        iou_columns = ['Class_0_IOU(class_0)', 'Class_1_IOU(class_1)',
                       'Class_2_IOU(class_2)', 'Scene_mIoU']
        y_max_iou = df[iou_columns].max().max()
        ax_iou.set_ylim(0.85, y_max_iou + 0.01)

        # IOU图表标签与美化
        ax_iou.set_xlabel('α', fontsize=14, fontweight='bold')
        ax_iou.set_ylabel('IOU', fontsize=12, fontweight='bold')
        ax_iou.legend(fontsize=10, loc='lower right', frameon=False)

        # 保存IOU图片
        iou_img_path = get_unique_filename(csv_directory, "sensitivity_analysis_iou", ".png")
        plt.tight_layout()
        fig_iou.savefig(iou_img_path, dpi=300, bbox_inches='tight', facecolor='white')

        # 4. 绘制ACC图表 - 减小宽度到8
        fig_acc, ax_acc = plt.subplots(figsize=(8, 7))  # 宽度从10减小到8

        # 隐藏顶部和右侧边框及刻度
        ax_acc.spines['top'].set_visible(False)
        ax_acc.spines['right'].set_visible(False)
        ax_acc.tick_params(top=False, right=False)
        ax_acc.spines['left'].set_linewidth(1.2)
        ax_acc.spines['bottom'].set_linewidth(1.2)

        # 横坐标设置
        ax_acc.set_xticks(np.arange(0.5, 1.6, 0.1))
        ax_acc.set_xlim(0.5, 1.5)
        # 调整刻度标签字体大小
        ax_acc.tick_params(axis='x', labelsize=10)

        # 绘制ACC折线
        ax_acc.plot(df['angle_weight'], df['Class_0_ACC(class_0)'] * 100,
                    marker=class_styles['Class_0'][1], color=class_styles['Class_0'][0],
                    label=f'{class_styles["Class_0"][2]} ACC', linewidth=2, markersize=6,
                    linestyle='--')

        ax_acc.plot(df['angle_weight'], df['Class_1_ACC(class_1)'] * 100,
                    marker=class_styles['Class_1'][1], color=class_styles['Class_1'][0],
                    label=f'{class_styles["Class_1"][2]} ACC', linewidth=2, markersize=6,
                    linestyle='--')

        ax_acc.plot(df['angle_weight'], df['Class_2_ACC(class_2)'] * 100,
                    marker=class_styles['Class_2'][1], color=class_styles['Class_2'][0],
                    label=f'{class_styles["Class_2"][2]} ACC', linewidth=2, markersize=6,
                    linestyle='--')

        ax_acc.plot(df['angle_weight'], df['Scene_OA'] * 100,
                    marker=overall_marker, color=overall_color,
                    label='Overall Accuracy', linewidth=2, markersize=6,
                    linestyle='-')

        # ACC坐标轴设置
        acc_columns = ['Class_0_ACC(class_0)', 'Class_1_ACC(class_1)',
                       'Class_2_ACC(class_2)', 'Scene_OA']
        y_max_acc = df[acc_columns].max().max() * 100
        ax_acc.set_ylim(90, y_max_acc + 0.5)

        # ACC图表标签与美化
        ax_acc.set_xlabel('α', fontsize=14, fontweight='bold')
        ax_acc.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax_acc.legend(fontsize=10, loc='lower right', frameon=False)

        # 保存ACC图片
        acc_img_path = get_unique_filename(csv_directory, "sensitivity_analysis_acc", ".png")
        plt.tight_layout()
        fig_acc.savefig(acc_img_path, dpi=300, bbox_inches='tight', facecolor='white')

        # 输出结果信息
        print(f"✅ CSV文件：{csv_filename}（目录：{csv_directory}）")
        print(f"✅ IOU图表已保存为：{iou_img_path}")
        print(f"✅ ACC图表已保存为：{acc_img_path}")

        # 显示图表
        plt.show()

    except Exception as e:
        print(f"❌ 绘图失败：{str(e)}")


if __name__ == "__main__":
    # 填写你的CSV文件路径
    csv_file = r"D:\桌面1\训练结果\scannet\参数敏感性分析\z_canshu\test_results.csv"
    plot_sensitivity_analysis(csv_file)
