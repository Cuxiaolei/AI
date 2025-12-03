import os
import sys
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# 健康状态映射
HEALTH_MAP = {'H': '健康', 'I': '内圈缺陷', 'O': '外圈缺陷'}
SPEED_MAP = {'A': '增加速度', 'B': '减小速度', 'C': '增加后减小', 'D': '减小后增加'}
SAMPLE_RATE = 200000  # Hz
EXPECTED_DURATION = 10  # 秒
EXPECTED_LENGTH = SAMPLE_RATE * EXPECTED_DURATION


def check_file_structure(file_path):
    """检查单个MAT文件结构"""
    try:
        data = sio.loadmat(file_path, verify_compressed_data_integrity=False)

        # 检查必要通道
        if 'Channel_1' not in data or 'Channel_2' not in data:
            return False, "缺少Channel_1或Channel_2"

        vibration = data['Channel_1']
        speed = data['Channel_2']

        # 检查数据类型和形状
        if vibration.ndim > 2 or speed.ndim > 2:
            return False, "数据维度异常 > 2D"

        vibration = vibration.flatten()
        speed = speed.flatten()

        # 检查长度
        if len(vibration) != len(speed):
            return False, f"双通道长度不匹配: vib={len(vibration)}, speed={len(speed)}"

        if len(vibration) < EXPECTED_LENGTH * 0.9:  # 允许10%误差
            return False, f"数据长度不足: {len(vibration)} < {EXPECTED_LENGTH}"

        # 检查数据完整性
        if np.isnan(vibration).any() or np.isinf(vibration).any():
            return False, "振动信号包含NaN或Inf"

        if np.isnan(speed).any() or np.isinf(speed).any():
            return False, "转速信号包含NaN或Inf"

        return True, {
            'length': len(vibration),
            'vibration_mean': np.mean(vibration),
            'vibration_std': np.std(vibration),
            'speed_mean': np.mean(speed),
            'speed_std': np.std(speed),
            'vibration_max': np.max(np.abs(vibration)),
            'speed_max': np.max(np.abs(speed))
        }

    except Exception as e:
        return False, f"加载失败: {str(e)}"


def analyze_domain_samples(data_path):
    """分析每个域的样本分布"""
    health_conditions = ['H', 'I', 'O']
    speed_conditions = ['A', 'B', 'C', 'D']

    domain_stats = {}
    sample_counts = defaultdict(lambda: defaultdict(int))

    total_samples = 0
    total_files = 0
    valid_files = 0

    print("=" * 80)
    print("📊 数据集样本分布分析")
    print("=" * 80)

    for health in health_conditions:
        for speed in speed_conditions:
            domain_id = len(domain_stats)
            file_prefix = f"{health}-{speed}"
            files = [f for f in os.listdir(data_path)
                     if f.startswith(file_prefix) and f.endswith('.mat')]

            if len(files) != 3:
                print(f"⚠️  {file_prefix}: 文件数量异常，期望3个，实际{len(files)}个")
                for f in files:
                    print(f"   {f}")

            domain_sample_count = 0
            domain_file_info = []

            for file_name in files:
                file_path = os.path.join(data_path, file_name)
                total_files += 1

                is_valid, info = check_file_structure(file_path)

                if is_valid:
                    valid_files += 1
                    # 计算样本数（滑动窗口切分）
                    n_samples = (info['length'] - 2048) // (2048 * 0.5) + 1
                    domain_sample_count += n_samples
                    total_samples += n_samples

                    domain_file_info.append({
                        'file': file_name,
                        'samples': n_samples,
                        'vib_std': info['vibration_std'],
                        'speed_mean': info['speed_mean']
                    })

                else:
                    print(f"❌ 文件错误 {file_name}: {info}")

            domain_stats[domain_id] = {
                'health': health,
                'speed': speed,
                'files': domain_file_info,
                'total_samples': domain_sample_count,
                'health_name': HEALTH_MAP[health],
                'speed_name': SPEED_MAP[speed]
            }

            # 每类样本统计
            sample_counts[health][speed] = domain_sample_count

    return domain_stats, sample_counts, total_samples, total_files, valid_files


def visualize_distribution(domain_stats, save_path='dataset_distribution.png'):
    """可视化数据分布"""
    fig = plt.figure(figsize=(18, 12))

    # 1. 热力图：每域样本数
    ax1 = plt.subplot(2, 3, 1)
    health_order = ['H', 'I', 'O']
    speed_order = ['A', 'B', 'C', 'D']
    heatmap_data = np.zeros((3, 4))

    for domain_id, stats in domain_stats.items():
        h_idx = health_order.index(stats['health'])
        s_idx = speed_order.index(stats['speed'])
        heatmap_data[h_idx, s_idx] = stats['total_samples']

    im = ax1.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(4))
    ax1.set_xticklabels([SPEED_MAP[s] for s in speed_order], rotation=45, ha='right')
    ax1.set_yticks(range(3))
    ax1.set_yticklabels([HEALTH_MAP[h] for h in health_order])
    ax1.set_title('每域总样本数', fontsize=14, fontweight='bold')

    # 添加数值标注
    for i in range(3):
        for j in range(4):
            text = ax1.text(j, i, f'{int(heatmap_data[i, j]):d}',
                            ha="center", va="center", color="black")

    plt.colorbar(im, ax=ax1)

    # 2. 箱线图：每类健康状态的样本分布
    ax2 = plt.subplot(2, 3, 2)
    health_samples = []
    health_labels = []
    for health in ['H', 'I', 'O']:
        samples = [domain_stats[d]['total_samples']
                   for d in domain_stats if domain_stats[d]['health'] == health]
        health_samples.append(samples)
        health_labels.append(HEALTH_MAP[health])

    ax2.boxplot(health_samples, labels=health_labels)
    ax2.set_title('健康状态样本分布', fontsize=14, fontweight='bold')
    ax2.set_ylabel('样本数量')

    # 3. 柱状图：每转速条件的样本分布
    ax3 = plt.subplot(2, 3, 3)
    speed_samples = []
    speed_labels = []
    for speed in ['A', 'B', 'C', 'D']:
        samples = [domain_stats[d]['total_samples']
                   for d in domain_stats if domain_stats[d]['speed'] == speed]
        speed_samples.append(sum(samples))
        speed_labels.append(SPEED_MAP[speed])

    ax3.bar(range(len(speed_samples)), speed_samples, color='skyblue')
    ax3.set_xticks(range(len(speed_labels)))
    ax3.set_xticklabels(speed_labels, rotation=45, ha='right')
    ax3.set_title('转速条件总样本数', fontsize=14, fontweight='bold')
    ax3.set_ylabel('样本数量')

    # 4. 异常值检测：振动信号标准差
    ax4 = plt.subplot(2, 3, 4)
    vib_stds = []
    labels = []
    for domain_id, stats in domain_stats.items():
        for file_info in stats['files']:
            if 'vib_std' in file_info:
                vib_stds.append(file_info['vib_std'])
                labels.append(f"{stats['health']}-{stats['speed']}")

    ax4.hist(vib_stds, bins=20, color='lightcoral', edgecolor='black')
    ax4.set_title('振动信号标准差分布', fontsize=14, fontweight='bold')
    ax4.set_xlabel('标准差')
    ax4.set_ylabel('频数')

    # 5. 转速信号统计
    ax5 = plt.subplot(2, 3, 5)
    speed_means = []
    for domain_id, stats in domain_stats.items():
        for file_info in stats['files']:
            if 'speed_mean' in file_info:
                speed_means.append(file_info['speed_mean'])

    ax5.boxplot(speed_means)
    ax5.set_title('转速信号均值分布', fontsize=14, fontweight='bold')
    ax5.set_ylabel('转速均值')

    # 6. 样本数统计表
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')

    table_data = []
    for health in ['H', 'I', 'O']:
        row = [HEALTH_MAP[health]]
        for speed in ['A', 'B', 'C', 'D']:
            domain_id = next(d for d in domain_stats
                             if domain_stats[d]['health'] == health
                             and domain_stats[d]['speed'] == speed)
            row.append(f"{domain_stats[domain_id]['total_samples']:,}")
        table_data.append(row)

    table = ax6.table(cellText=table_data,
                      colLabels=['健康状态'] + [SPEED_MAP[s] for s in ['A', 'B', 'C', 'D']],
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax6.set_title('样本数统计表', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return save_path


def check_class_balance(sample_counts):
    """检查类别平衡性"""
    print("\n" + "=" * 80)
    print("⚖️  类别平衡性分析")
    print("=" * 80)

    for health in ['H', 'I', 'O']:
        total = sum(sample_counts[health].values())
        print(f"\n{HEALTH_MAP[health]}: {total:,} 样本")
        for speed in ['A', 'B', 'C', 'D']:
            count = sample_counts[health][speed]
            percentage = count / total * 100 if total > 0 else 0
            print(f"  {SPEED_MAP[speed]}: {count:>6,} ({percentage:>5.1f}%)")


def check_speed_profile(data_path):
    """检查转速信号的有效性"""
    print("\n" + "=" * 80)
    print("🔄 转速信号质量分析")
    print("=" * 80)

    files = [f for f in os.listdir(data_path) if f.endswith('.mat')]

    for file_name in files[:3]:  # 只检查前3个文件作为示例
        file_path = os.path.join(data_path, file_name)
        data = sio.loadmat(file_path)
        speed = data['Channel_2'].flatten()

        # 转速信号统计
        speed_min, speed_max = np.min(speed), np.max(speed)
        speed_mean = np.mean(speed)
        speed_std = np.std(speed)

        print(f"\n📁 {file_name}")
        print(f"   转速范围: {speed_min:.2f} - {speed_max:.2f} RPM")
        print(f"   均值: {speed_mean:.2f} RPM")
        print(f"   标准差: {speed_std:.2f} RPM")
        print(f"   变化率: {((speed_max - speed_min) / speed_mean * 100):.1f}%")

        # 检查是否符合预期模式
        if 'A' in file_name:  # 增加速度
            trend = "上升" if speed_max > speed_min * 1.1 else "平稳"
            print(f"   趋势: {trend}")
        elif 'B' in file_name:  # 减小速度
            trend = "下降" if speed_max < speed_min * 0.9 else "平稳"
            print(f"   趋势: {trend}")


def generate_diagnostic_report(domain_stats, sample_counts,
                               total_samples, total_files, valid_files):
    """生成诊断报告"""
    report = []
    report.append("=" * 80)
    report.append("📋 渥太华轴承数据集诊断报告")
    report.append("=" * 80)
    report.append(f"检查时间: {np.datetime64('now')}")
    report.append(f"文件总数: {total_files}")
    report.append(f"有效文件: {valid_files} ({valid_files / total_files * 100:.1f}%)")
    report.append(f"无效文件: {total_files - valid_files}")
    report.append(f"总样本数: {total_samples:,}")
    report.append(f"域数量: 12")
    report.append(f"每域预期文件: 3")
    report.append("")

    # 样本分布
    report.append("-" * 80)
    report.append("样本分布统计:")
    report.append("-" * 80)

    for health in ['H', 'I', 'O']:
        health_total = sum(sample_counts[health].values())
        report.append(f"\n{HEALTH_MAP[health]}: {health_total:,} 样本")

        for speed in ['A', 'B', 'C', 'D']:
            domain_id = next(d for d in domain_stats
                             if domain_stats[d]['health'] == health
                             and domain_stats[d]['speed'] == speed)
            count = domain_stats[domain_id]['total_samples']
            report.append(f"  条件{speed} ({SPEED_MAP[speed][:4]}): {count:>6,} 样本")

    # 问题诊断
    report.append("\n" + "-" * 80)
    report.append("问题诊断:")
    report.append("-" * 80)

    # 1. 样本过少的域
    min_samples = float('inf')
    min_domain = None
    for domain_id, stats in domain_stats.items():
        if stats['total_samples'] < min_samples:
            min_samples = stats['total_samples']
            min_domain = f"{stats['health']}-{stats['speed']}"

    report.append(f"\n1. 最少样本域: {min_domain} ({min_samples:,} 样本)")

    # 2. 每域最小需求分析
    min_needed_per_domain = 3 * 20  # 3类 × 每类至少20样本（5支持+15查询）
    insufficient_domains = []
    for domain_id, stats in domain_stats.items():
        if stats['total_samples'] < min_needed_per_domain:
            insufficient_domains.append(f"{stats['health']}-{stats['speed']}")

    if insufficient_domains:
        report.append(f"\n2. ⚠️  样本不足的域: {', '.join(insufficient_domains)}")
        report.append(f"   最小需求: {min_needed_per_domain} 样本/域")
    else:
        report.append("\n2. ✅ 所有域样本充足")

    # 3. 类别不平衡
    class_totals = [sum(sample_counts[h].values()) for h in ['H', 'I', 'O']]
    max_class = max(class_totals)
    min_class = min(class_totals)
    imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')

    report.append(f"\n3. 类别不平衡比: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 2:
        report.append("   ⚠️  类别不平衡严重！")
    else:
        report.append("   ✅ 类别基本平衡")

    return "\n".join(report)


def main():
    """主函数"""
    data_path = "/root/data/Ottawa_Bearing_Dataset"

    if not os.path.exists(data_path):
        print(f"❌ 数据路径不存在: {data_path}")
        sys.exit(1)

    print("🔍 开始诊断数据集...")
    print(f"数据路径: {data_path}")

    # 1. 分析域样本分布
    domain_stats, sample_counts, total_samples, total_files, valid_files = \
        analyze_domain_samples(data_path)

    # 2. 可视化分布
    plot_path = visualize_distribution(domain_stats)
    print(f"\n📊 可视化已保存: {plot_path}")

    # 3. 检查类别平衡
    check_class_balance(sample_counts)

    # 4. 检查转速信号
    check_speed_profile(data_path)

    # 5. 生成诊断报告
    report = generate_diagnostic_report(
        domain_stats, sample_counts, total_samples, total_files, valid_files
    )

    # 保存报告
    report_path = "dataset_diagnostic_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print("\n" + report)
    print(f"\n💾 完整报告已保存: {report_path}")

    # 6. 关键结论
    print("\n" + "=" * 80)
    print("💡 关键结论与建议")
    print("=" * 80)

    # 样本充足性检查
    min_samples_needed = 3 * 20  # 5支持+15查询
    insufficient_domains = [
        f"{d['health']}-{d['speed']}"
        for d in domain_stats.values()
        if d['total_samples'] < min_samples_needed
    ]

    if insufficient_domains:
        print(f"\n⚠️  发现样本不足的域: {len(insufficient_domains)}个")
        print("   建议方案:")
        print("   1. 减小 config.yaml 中的 k_shot (如从5降到3)")
        print("   2. 减小 n_query (如从15降到10)")
        print("   3. 增大 window_size (如从2048降到1024) 以增加样本数")
        print("   4. 增大 overlap (如从0.5增加到0.75)")
    else:
        print("\n✅ 所有域样本充足，可直接使用当前配置")

    # 文件完整性
    if valid_files == total_files:
        print("\n✅ 所有文件完整且有效")
    else:
        print(f"\n⚠️  有 {total_files - valid_files} 个无效文件，建议重新下载")


if __name__ == '__main__':
    main()
