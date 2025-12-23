"""
批量分析 PU 数据集所有 .mat 文件的内部结构与标签信息，并检查坏文件

功能：
1. 遍历 ROOT_DIR 下所有 .mat 文件；
2. 对每个文件：
   - 尝试用 scipy.io.loadmat 读取；
   - 若读取失败（例如文件损坏），记录为“坏文件”，并跳过后续解析；
   - 若读取成功：
        * 打印顶层变量名与类型；
        * 若是 struct，打印顶层字段；
        * 若存在字段 'Y'，统计/打印每个通道的 Name / Unit / Data 形状；
3. 汇总所有“正常文件”的结构信息，给出：
   - 顶层 struct 字段全局统计；
   - 通道 struct 字段全局统计；
   - 各通道序号上的 Name / Unit / Data 形状集合；
4. 将所有输出内容保存为 ROOT_DIR 下的
   `pu_mat_structure_report.txt` 文本文档。

注意：
- 依赖 scipy： pip install scipy
- 为避免控制台太多输出，只对前 MAX_FILES_DETAILED 个文件打印详细通道信息；
  如果想看所有文件细节，把 MAX_FILES_DETAILED 改为 None。
"""

import os
from collections import defaultdict
import numpy as np
import scipy.io as sio

# ======== 需要你修改的地方 ========
ROOT_DIR = r"D:\user\dataSet\！！工业旋转轴承数据集\德国帕德博恩轴承数据集"

# 为了避免控制台信息过多，这里默认只详细打印前 N 个文件
MAX_FILES_DETAILED = None      # 想看所有文件就改成 None
# ==============================


def to_scalar(x):
    """把 MATLAB 导入的各种 ndarray/cell 变成 Python 标量/字符串（尽量通用）"""
    if isinstance(x, np.ndarray):
        x = x.squeeze()
        if x.shape == ():
            try:
                x = x.item()
            except Exception:
                pass
    return x


def summarize_mat_file(filepath, global_state, detailed, report_lines):
    """
    解析单个 .mat 文件：
    - 尝试 loadmat，失败则记录到 bad_files；
    - 成功则：
      * 打印顶层变量名、类型；
      * 若为 struct，则统计顶层字段；
      * 若存在字段 'Y'，则统计通道字段 / Name / Unit / Data 形状。
    """

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=" * 80)
    log(f"文件: {filepath}")

    # ---------- 尝试读取 .mat ----------
    try:
        # 如果你想尝试 squeeze_me / struct_as_record 也可以在这里调整参数
        mat = sio.loadmat(filepath)
    except Exception as e:
        err_msg = f"❌ 读取该 .mat 文件时发生错误: {repr(e)}"
        log(err_msg)
        global_state["bad_files"].append((filepath, repr(e)))
        return  # 直接返回，不再做后续字段解析

    global_state["good_files"] += 1

    # ---------- 顶层变量 ----------
    top_keys = [k for k in mat.keys() if not k.startswith("__")]
    log(f"顶层变量名: {top_keys}")

    if not top_keys:
        log("⚠ 未找到顶层变量（只有 __header__ 等系统字段）")
        return

    # 通常 PU 每个 .mat 只有一个顶层变量
    top_key = top_keys[0]
    top_obj = mat[top_key]
    log(f"顶层变量类型: {type(top_obj)} shape: {getattr(top_obj, 'shape', None)}")

    # ---------- 解析顶层 struct 字段 ----------
    top_struct = None
    struct_fields = []

    try:
        # 常见情况：顶层变量是 shape=(1,1) 的 ndarray，里面包着一个 struct
        if isinstance(top_obj, np.ndarray) and top_obj.size == 1:
            candidate = top_obj.reshape(-1)[0]
            # candidate 通常是 numpy.void，dtype.names 为字段名
            if isinstance(candidate, np.void) and candidate.dtype.names:
                top_struct = candidate
                struct_fields = list(candidate.dtype.names)
    except Exception as e:
        log(f"解析顶层 struct 时出错: {repr(e)}")

    if top_struct is None:
        log("⚠ 顶层变量不是预期的 struct（或解析失败），跳过字段解析。")
        return

    log(f"顶层 struct 字段: {struct_fields}")
    for fld in struct_fields:
        global_state["top_fields_count"][fld] += 1

    # ---------- 解析 Y 字段（通道信息） ----------
    if "Y" not in struct_fields:
        log("⚠ 顶层 struct 中未发现字段 'Y'")
        return

    Y = top_struct["Y"]
    log(f"Y 类型: {type(Y)} shape: {getattr(Y, 'shape', None)}")

    # Y 一般是 (1, N) 或 (N,) 的 struct 数组，每个元素是一个通道
    if isinstance(Y, np.ndarray):
        if Y.ndim == 2:
            num_channels = Y.shape[1]
        else:
            num_channels = Y.size
    else:
        log("⚠ Y 不是 ndarray，暂不解析通道。")
        return

    log(f"Y 中通道数: {num_channels}")

    for ch_idx in range(num_channels):
        ch = Y[0, ch_idx] if (Y.ndim == 2 and Y.shape[0] == 1) else Y[ch_idx]

        if not isinstance(ch, np.void) or ch.dtype.names is None:
            log(f"  通道 {ch_idx+1}: 类型异常（非 struct），跳过")
            continue

        ch_fields = list(ch.dtype.names)
        for fld in ch_fields:
            global_state["channel_field_count"][fld] += 1

        # Name / Unit / Data
        name = None
        unit = None
        data_len = None
        data_shape = None

        if "Name" in ch_fields:
            name = to_scalar(ch["Name"])
        if "Unit" in ch_fields:
            unit = to_scalar(ch["Unit"])
        if "Data" in ch_fields:
            data_arr = ch["Data"]
            if isinstance(data_arr, np.ndarray):
                data_shape = data_arr.shape
                data_len = data_arr.size

        # 只对前 MAX_FILES_DETAILED 个文件打印通道详细信息
        if detailed:
            log(f"  通道 {ch_idx+1}:")
            log(f"    字段: {ch_fields}")
            log(f"    Name: {name}")
            log(f"    Unit: {unit}")
            log(f"    Data shape: {data_shape}, length={data_len}")

        # 全局统计（按通道序号）
        idx = ch_idx + 1
        if name is not None:
            global_state["channel_names"][idx].add(str(name))
        if unit is not None:
            global_state["channel_units"][idx].add(str(unit))
        if data_shape is not None:
            global_state["channel_shapes"][idx].add(str(data_shape))
        global_state["channel_file_count"][idx] += 1


def main():
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    # 全局统计用容器
    global_state = {
        "top_fields_count": defaultdict(int),     # 顶层 struct 字段 -> 出现次数
        "channel_field_count": defaultdict(int),  # 通道 struct 字段 -> 出现次数
        "channel_names": defaultdict(set),        # 通道序号 -> {Name 集合}
        "channel_units": defaultdict(set),        # 通道序号 -> {Unit 集合}
        "channel_shapes": defaultdict(set),       # 通道序号 -> {Data 形状集合}
        "channel_file_count": defaultdict(int),   # 通道序号 -> 在多少文件中被成功解析
        "bad_files": [],                          # [(filepath, error_msg), ...]
        "good_files": 0,                          # 成功读取的文件数
    }

    # 收集所有 .mat 文件
    mat_files = []
    for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
        for fname in filenames:
            if fname.lower().endswith(".mat"):
                mat_files.append(os.path.join(dirpath, fname))

    mat_files.sort()
    log(f"在目录 '{ROOT_DIR}' 下共找到 {len(mat_files)} 个 .mat 文件。\n")

    # 逐文件解析
    for i, fpath in enumerate(mat_files, start=1):
        detailed = (MAX_FILES_DETAILED is None) or (i <= MAX_FILES_DETAILED)
        summarize_mat_file(fpath, global_state, detailed=detailed, report_lines=report_lines)

        if (MAX_FILES_DETAILED is not None) and (i == MAX_FILES_DETAILED):
            log("=" * 80)
            log(f"以上是前 {MAX_FILES_DETAILED} 个文件的详细信息。")
            log("后续文件将不再逐个打印通道详情，只参与全局统计。\n")

    # ============== 全局总结 ==============
    total_files = len(mat_files)
    bad_files = global_state["bad_files"]
    good_files = global_state["good_files"]

    log("\n" + "#" * 80)
    log("全局总结：文件读取情况")
    log("#" * 80)
    log(f"总 .mat 文件数: {total_files}")
    log(f"成功解析: {good_files}")
    log(f"解析失败: {len(bad_files)}")

    if bad_files:
        log("\n以下 .mat 文件在读取时发生错误（建议重新下载/解压）：")
        for path, err in bad_files:
            log(f"- {path}")
            log(f"    错误: {err}")

    log("\n" + "#" * 80)
    log("全局总结：顶层 struct 字段统计")
    log("#" * 80)
    for fld, cnt in sorted(global_state["top_fields_count"].items(), key=lambda x: x[0]):
        log(f"字段 '{fld}' 出现在 {cnt} 个文件的顶层 struct 中")

    log("\n" + "#" * 80)
    log("全局总结：通道 struct 字段统计")
    log("#" * 80)
    for fld, cnt in sorted(global_state["channel_field_count"].items(), key=lambda x: x[0]):
        log(f"字段 '{fld}' 出现在 {cnt} 个文件的通道 struct 中")

    log("\n" + "#" * 80)
    log("全局总结：按通道序号的 Name / Unit / Data 形状")
    log("#" * 80)
    for idx in sorted(global_state["channel_file_count"].keys()):
        names = ", ".join(sorted(global_state["channel_names"][idx])) or "（无）"
        units = ", ".join(sorted(global_state["channel_units"][idx])) or "（无）"
        shapes = ", ".join(sorted(global_state["channel_shapes"][idx])) or "（未知）"
        files = global_state["channel_file_count"][idx]
        log(f"通道 {idx}: 出现在 {files} 个成功解析的文件中")
        log(f"  Name 集合: {names}")
        log(f"  Unit 集合: {units}")
        log(f"  Data 形状集合: {shapes}")

    # ============== 写入文本报告 ==============
    out_path = os.path.join(ROOT_DIR, "pu_mat_structure_report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    log(f"\n已将详细报告保存为文本文件：{out_path}")


if __name__ == "__main__":
    main()
