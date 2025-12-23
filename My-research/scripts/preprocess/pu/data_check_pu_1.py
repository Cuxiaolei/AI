import os
import re
from collections import defaultdict

# TODO: 把这里改成你自己的 PU 数据集根目录
ROOT_DIR = r"D:\user\dataSet\！！工业旋转轴承数据集\德国帕德博恩轴承数据集"  # 举例：r"D:\datasets\PU"

# 匹配类似：N15_M07_F10_KA01_1.mat
FNAME_RE = re.compile(
    r"(N\d{2}_M\d{2}_F\d{2})_(K[A-Z]?\d{2,3})_(\d+)\.mat$",
    re.IGNORECASE
)


def main():
    total_mat_files = 0
    unmatched_files = []  # 文件名不符合规范的 .mat
    non_mat_files = []  # 不是 .mat 的文件

    by_bearing = defaultdict(int)  # 轴承编号 -> 文件数
    by_condition = defaultdict(int)  # 工况 -> 文件数
    by_bearing_condition = defaultdict(int)  # (轴承, 工况) -> 文件数
    indices_by_bc = defaultdict(list)  # (轴承, 工况) -> [测量编号列表]

    for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if not fname.lower().endswith(".mat"):
                non_mat_files.append(fpath)
                continue

            total_mat_files += 1
            m = FNAME_RE.search(fname)
            if not m:
                unmatched_files.append(fpath)
                continue

            cond, bearing, idx_str = m.groups()
            idx = int(idx_str)

            by_bearing[bearing] += 1
            by_condition[cond] += 1
            by_bearing_condition[(bearing, cond)] += 1
            indices_by_bc[(bearing, cond)].append(idx)

    # 开始生成报告
    lines = []
    lines.append(f"PU 数据集根目录: {ROOT_DIR}")
    lines.append("=" * 80)
    lines.append(f"总 .mat 文件数: {total_mat_files}")
    lines.append("")

    # 轴承维度统计
    lines.append("一、按轴承编号统计 (bearing code)")
    lines.append("-" * 80)
    for bearing in sorted(by_bearing.keys()):
        lines.append(f"{bearing:>6s} : {by_bearing[bearing]} 个 .mat 文件")
    lines.append("")

    # 工况维度统计
    lines.append("二、按工况统计 (operating condition Nxx_Mxx_Fxx)")
    lines.append("-" * 80)
    for cond in sorted(by_condition.keys()):
        lines.append(f"{cond:>12s} : {by_condition[cond]} 个 .mat 文件")
    lines.append("")

    # 轴承 + 工况 维度统计 & 缺失测量编号检查
    lines.append("三、按 (轴承, 工况) 统计，并检查测量编号缺失情况")
    lines.append("-" * 80)
    EXPECTED_INDICES = set(range(1, 21))  # 理论上 1~20
    for (bearing, cond) in sorted(by_bearing_condition.keys()):
        indices = sorted(set(indices_by_bc[(bearing, cond)]))
        missing = sorted(EXPECTED_INDICES - set(indices))
        lines.append(f"{bearing:>6s} @ {cond:>12s} : "
                     f"{len(indices)} 个文件，测量编号 = {indices}")
        if missing:
            lines.append(f"    ⚠ 缺失测量编号: {missing}")
        else:
            lines.append(f"    ✓ 测量编号 1-20 完整")
    lines.append("")

    # 文件名不匹配情况
    lines.append("四、文件名不符合 Nxx_Mxx_Fxx_Kxxx_i.mat 规范的 .mat 文件")
    lines.append("-" * 80)
    if unmatched_files:
        for p in unmatched_files:
            lines.append(p)
    else:
        lines.append("无")
    lines.append("")

    # 非 .mat 文件列表（看有没有奇怪的东西）
    lines.append("五、非 .mat 文件列表（仅供检查，可以忽略）")
    lines.append("-" * 80)
    if non_mat_files:
        for p in non_mat_files:
            lines.append(p)
    else:
        lines.append("无")
    lines.append("")

    report_text = "\n".join(lines)
    print(report_text)

    # 写入报告文件
    report_path = os.path.join(ROOT_DIR, "pu_dataset_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n报告已保存到: {report_path}")


if __name__ == "__main__":
    main()