import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import os
from datetime import datetime


def load_all_models_data(csv_path, expected_total_models=8):
    """加载CSV中所有8个模型（E1-E8）的数据，供后续实验筛选"""
    df = pd.read_csv(csv_path)
    all_models = {}
    model_names = []

    # 提取所有模型名称（按E1-E8的列名顺序）
    for col in df.columns:
        if '_miou' in col:
            model_name = col.split('_')[0]  # 列名格式需为 "E1_miou", "E1_oa"
            if model_name not in model_names:
                model_names.append(model_name)

    # 校验总模型数量（确保CSV包含E1-E8）
    if len(model_names) != expected_total_models:
        raise ValueError(f"CSV文件仅检测到{len(model_names)}个模型，需包含E1-E8共8个模型！")

    # 加载每个模型的mIoU和OA数据
    for model in model_names:
        miou_col = f"{model}_miou"
        oa_col = f"{model}_oa"
        if miou_col not in df.columns or oa_col not in df.columns:
            raise KeyError(f"CSV缺少模型{model}的列：{miou_col}或{oa_col}")
        all_models[model] = {
            'miou': df[miou_col].values,
            'oa': df[oa_col].values
        }
    return all_models, len(df)  # 返回所有模型数据 + 训练轮次


def filter_experiment_models(all_models, target_models):
    """根据当前实验的目标模型（如E1,E2,E7,E8）筛选数据"""
    filtered = {}
    for model in target_models:
        if model not in all_models:
            raise ValueError(f"目标模型{model}不在所有模型列表中（需为E1-E8）")
        filtered[model] = all_models[model]
    return filtered


def plot_experiment_curve(filtered_models, config, metric_type, is_zoom=False):
    """绘制单个实验的曲线（支持mIoU或OA），增加is_zoom参数用于处理放大图"""
    # 初始化图表
    plt.figure(figsize=config['fig_size'])
    ax = plt.gca()

    # 获取当前实验的核心配置
    exp_name = config['exp_name']
    colors = config['colors']
    epochs = np.arange(1, config['num_epochs'] + 1)

    # 绘制每个模型的曲线（颜色与模型一一对应）
    for idx, (model_name, metrics) in enumerate(filtered_models.items()):
        if idx >= len(colors):
            raise ValueError(f"实验{exp_name}的颜色数量（{len(colors)}）少于模型数量（{len(filtered_models)}）")

        color = colors[idx]
        metric_data = metrics[metric_type] * 100  # 转为百分比

        # 如果是放大图，只绘制70-100 epoch的数据
        if is_zoom:
            # 确保不超出实际数据范围
            end_idx = min(100, len(epochs))
            plot_epochs = epochs[69:end_idx]  # 70-100 epoch（索引69到99）
            plot_data = metric_data[69:end_idx]
        else:
            plot_epochs = epochs
            plot_data = metric_data

        plt.plot(
            plot_epochs, plot_data,
            label=model_name,
            color=color,
            linestyle='-',
            linewidth=1,
            alpha=0.8
        )

    # 图表基础配置
    plt.title('', fontsize=config['font']['title_size'])
    plt.xlabel(config['axis_labels']['x'], fontsize=config['font']['label_size'])
    plt.ylabel(f"mIoU(%)", fontsize=config['font']['label_size'])

    # 设置坐标轴范围
    if is_zoom:
        # 放大图：x为70-100，y为85-100
        plt.xlim(70, min(100, config['num_epochs']))
        plt.ylim(94, 100)

        # X轴刻度设置：整数刻度显示值，半刻度线不显示值
        ax.xaxis.set_major_locator(MultipleLocator(5))  # x主刻度间隔5
        ax.xaxis.set_minor_locator(MultipleLocator(2.5))  # x半刻度间隔2.5
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))  # 主刻度显示整数
        ax.xaxis.set_minor_formatter(plt.NullFormatter())  # 次刻度不显示值

        # Y轴刻度设置：整数刻度显示值，半刻度线不显示值
        ax.yaxis.set_major_locator(MultipleLocator(1))  # y主刻度间隔5
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))  # y半刻度间隔2.5
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))  # 主刻度显示整数
        ax.yaxis.set_minor_formatter(plt.NullFormatter())  # 次刻度不显示值
    else:
        # 原图使用配置的坐标轴范围
        plt.xlim(config['axis_lim']['x'])
        plt.ylim(config['axis_lim']['y'])

        # X轴刻度设置：整数刻度显示值，半刻度线不显示值
        ax.xaxis.set_major_locator(MultipleLocator(config['ticker']['x_major']))
        ax.xaxis.set_minor_locator(MultipleLocator(config['ticker']['x_major'] / 2))  # 半刻度
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))  # 主刻度显示整数
        ax.xaxis.set_minor_formatter(plt.NullFormatter())  # 次刻度不显示值

        # Y轴刻度设置：整数刻度显示值，半刻度线不显示值
        ax.yaxis.set_major_locator(MultipleLocator(config['ticker']['y_major']))
        ax.yaxis.set_minor_locator(MultipleLocator(config['ticker']['y_major'] / 2))  # 半刻度
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))  # 主刻度显示整数
        ax.yaxis.set_minor_formatter(plt.NullFormatter())  # 次刻度不显示值

    # 设置刻度标签大小
    ax.tick_params(axis='both', which='major', labelsize=config['font']['tick_size'])
    ax.tick_params(axis='both', which='minor', labelsize=0)  # 次刻度无标签

    # 边框样式优化
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # 图例配置
    plt.legend(
        loc='lower right',
        ncol=1,
        frameon=True,
        framealpha=0.9,
        fontsize=config['font']['legend_size'],
        bbox_to_anchor=(0.98, 0.02)
    )

    # 生成图片名称（放大图添加zoom标识）
    csv_basename = os.path.splitext(os.path.basename(config['csv_path']))[0]
    zoom_suffix = "_zoom" if is_zoom else ""
    file_name = f"{exp_name}_{metric_type}_{csv_basename}{zoom_suffix}.png"
    save_path = os.path.join(config['save_root'], file_name)

    # 保存图片
    plt.savefig(
        save_path,
        dpi=config['save_dpi'],
        bbox_inches='tight',
        pad_inches=0.3,
        pil_kwargs=dict(quality=95)
    )
    print(f"✅ 实验{exp_name}的{metric_type.upper()}{'放大' if is_zoom else ''}图已保存：{save_path}")
    plt.close()


def main():
    # ==============================
    # 消融实验集中配置区（所有参数在此修改）
    # ==============================
    cf = {
        # 1. 基础路径配置
        'csv_path': r"D:\user\Documents\ai\三维重建\我的论文-三维重建\训练结果\scannet\消融\z_xiaorong\1.csv",  # 8个模型（E1-E8）数据CSV路径
        'base_save_root': r"D:\user\Documents\ai\三维重建\我的论文-三维重建\训练结果\scannet\消融\z_xiaorong\消融实验结果",  # 基础保存根目录
        'save_dpi': 600,  # 图片分辨率

        # 2. 全局图表样式（所有实验共用）
        'font': {
            'family': ['Times New Roman', 'serif'],
            'title_size': 16,
            'label_size': 14,
            'tick_size': 12,
            'legend_size': 11
        },
        'fig_size': (10, 7),
        'axis_labels': {'x': 'Training Epochs'},
        'axis_lim': {'x': (0, None), 'y': (0, 100)},
        'ticker': {
            'x_major': 10,
            'x_minor': 5,
            'y_major': 10,
            'y_minor': 5
        },

        # 3. 三组消融实验具体配置
        'experiments': [
            {
                'exp_name': "消融实验1",
                'target_models': ['E1', 'E2', 'E7', 'E8'],
                'colors': ['orange', 'blue', 'green', 'red']
            },
            {
                'exp_name': "消融实验2",
                'target_models': ['E1', 'E3', 'E6', 'E8'],
                'colors': ['orange', 'blue', 'green', 'red']
            },
            {
                'exp_name': "消融实验3",
                'target_models': ['E1', 'E4', 'E5', 'E8'],
                'colors': ['orange', 'blue', 'green', 'red']
            }
        ]
    }

    # ==============================
    # 创建时间命名的独立文件夹
    # ==============================
    time_folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    cf['save_root'] = os.path.join(cf['base_save_root'], time_folder_name)
    os.makedirs(cf['save_root'], exist_ok=True)
    print(f"📂 本次结果保存目录：{cf['save_root']}")

    # ==============================
    # 执行消融实验
    # ==============================
    # 1. 加载所有8个模型数据
    all_models, num_epochs = load_all_models_data(
        csv_path=cf['csv_path'],
        expected_total_models=8
    )
    # 补充X轴范围（自动适配实际训练轮次）
    default_x_lim = cf['axis_lim']['x']
    cf['axis_lim']['x'] = (default_x_lim[0], num_epochs)

    # 2. 循环生成每组实验的图表（原图+放大图）
    for exp_config in cf['experiments']:
        # 筛选当前实验的模型数据
        filtered_models = filter_experiment_models(
            all_models=all_models,
            target_models=exp_config['target_models']
        )
        # 整合完整配置
        current_full_config = {
            **cf,
            **exp_config,
            'num_epochs': num_epochs
        }

        # 为每个实验生成三种图：mIoU原图、OA原图、mIoU放大图
        # 1. 生成mIoU原图
        plot_experiment_curve(filtered_models, current_full_config, 'miou', is_zoom=False)
        # 2. 生成OA原图
        plot_experiment_curve(filtered_models, current_full_config, 'oa', is_zoom=False)
        # 3. 生成mIoU放大图（70-100 epoch，85-100%）
        plot_experiment_curve(filtered_models, current_full_config, 'miou', is_zoom=True)

    print("\n🎉 所有消融实验图表生成完成！")


if __name__ == "__main__":
    main()
