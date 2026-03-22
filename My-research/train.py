import subprocess
import sys
import re
from dataclasses import dataclass
from typing import List, Optional, Set
import ctypes
import time
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
print(f"[{time.strftime('%H:%M:%S')}] 本脚本防休眠已开启")

# ====================== 核心配置类（灵活控制开关）======================
@dataclass
class RunConfig:
    """
    运行配置类：通过以下参数灵活筛选需要运行的模型配置文件
    支持模糊匹配（留空则不筛选该维度），多值用逗号分隔
    """
    # 方法名筛选（如：darm,dpjdg,erm），留空则运行所有方法
    methods: str = ""
    # 数据集筛选（如：cwru,phm,pu），留空则运行所有数据集
    datasets: str = ""
    # T编号筛选（如：T1,T2,T5-T8,T9-T12），留空则运行所有T编号
    t_nums: str = ""
    # 后缀筛选（如：300-1,300-5,300-10,300-20），留空则运行所有后缀
    suffixes: str = ""
    # 是否开启调试模式（只打印待运行的配置，不实际执行）
    debug_mode: bool = False


# ====================== 工具函数 ======================
def parse_range_str(range_str: str) -> Set[str]:
    """解析范围字符串（如T1-T4,T9）为具体的T编号集合"""
    result = set()
    if not range_str:
        return result

    parts = range_str.split(",")
    for part in parts:
        part = part.strip()
        if "-" in part:
            # 处理范围（如T1-T4）
            match = re.match(r"T(\d+)-T(\d+)", part)
            if match:
                start = int(match.group(1))
                end = int(match.group(2))
                for num in range(start, end + 1):
                    result.add(f"T{num}")
        else:
            # 处理单个值（如T5）
            if part.startswith("T") and part[1:].isdigit():
                result.add(part)
    return result


def load_config_paths_from_file(file_path: str = "config_paths.txt") -> List[str]:
    config_paths = set()  # 用set去重
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                # 去除首尾空白字符
                line = line.strip()
                # 跳过空行和注释行（#开头）
                if not line or line.startswith("#"):
                    continue
                # 移除可能的引号和逗号（兼容之前生成的格式）
                line = line.replace('"', '').replace(',', '').strip()
                # 验证路径格式（基础校验）
                if not line.endswith(".yaml"):
                    print(f"⚠️  第{line_num}行路径格式异常（非.yaml文件），已跳过：{line}")
                    continue
                config_paths.add(line)
    except FileNotFoundError:
        print(f"❌ 配置路径文件 {file_path} 不存在！")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 读取配置路径文件失败：{str(e)}")
        sys.exit(1)

    # 转为列表并排序（保证顺序稳定）
    return sorted(list(config_paths))


def filter_config_paths(all_paths: List[str], run_config: RunConfig) -> List[str]:
    """根据RunConfig筛选需要运行的配置文件路径"""
    # 解析筛选条件
    target_methods = set([m.strip() for m in run_config.methods.split(",") if m.strip()])
    target_datasets = set([d.strip() for d in run_config.datasets.split(",") if d.strip()])
    target_t_nums = parse_range_str(run_config.t_nums)
    target_suffixes = set([s.strip() for s in run_config.suffixes.split(",") if s.strip()])

    filtered = []
    for path in all_paths:
        # 解析路径中的关键信息（正则匹配）
        # 路径格式示例：darm/darm_cwru_T1_300-1.yaml
        match = re.match(
            r"^(?P<method>\w+)/(?P<method2>\w+)_(?P<dataset>\w+)_(?P<t_num>T\d+)_(?P<suffix>300-\d+)\.yaml$", path)
        if not match:
            print(f"⚠️  路径格式无法解析，已跳过：{path}")
            continue

        info = match.groupdict()
        method = info["method"]
        dataset = info["dataset"]
        t_num = info["t_num"]
        suffix = info["suffix"]

        # 按条件筛选
        if target_methods and method not in target_methods:
            continue
        if target_datasets and dataset not in target_datasets:
            continue
        if target_t_nums and t_num not in target_t_nums:
            continue
        if target_suffixes and suffix not in target_suffixes:
            continue

        filtered.append(path)

    return filtered


# ====================== 执行逻辑 ======================
def run_model_test(config_path: str, debug_mode: bool = False):
    """执行单个模型测试"""
    model_name = config_path.split("/")[-1].replace('.yaml', '')
    print(f"\n测试{model_name}模型")

    if debug_mode:
        print(f"[调试模式] 待执行命令: {sys.executable} -m src.main --configs src/configs/{config_path}")
        return

    cmd = [
        sys.executable,
        "-m", "src.main",
        "--configs", "src/configs/" + config_path
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            encoding="utf-8"
        )
        print(f"✅ {model_name}模型测试完成")
    except subprocess.CalledProcessError as e:
        print(f"❌ {model_name}模型测试失败，错误码: {e.returncode}")


# ====================== 主逻辑 =======================
if __name__ == "__main__":
    # 1. 从文件加载所有配置路径（核心修改）
    CONFIG_PATH_FILE = "model_file_paths.txt"  # 配置路径文件名称
    ALL_CONFIG_PATHS = load_config_paths_from_file(CONFIG_PATH_FILE)

    print(f"📄 从 {CONFIG_PATH_FILE} 加载到 {len(ALL_CONFIG_PATHS)} 个配置路径")

    # 2. 配置运行规则（重点：修改这里即可灵活控制）
    # 示例1：运行darm方法下cwru数据集的所有T编号、所有后缀
    run_config = RunConfig(
        methods=
        # "darm," +
        # "dpjdg," +
        # "erm," +
        # "groupdro," +
        # "irm," +
        # "mixstyle," +
        # "mldg," +
        # "vrex," +
        "mcpdg," +
        "",  # 自动忽略末尾逗号

        # datasets="cwru",
        datasets="phm",
        # datasets="pu",
        debug_mode=False  # 先调试看筛选结果，确认后改为False
    )

    # 示例2：运行dpjdg和erm方法下，phm数据集、T5-T8、后缀300-1和300-5
    # run_config = RunConfig(
    #     methods="dpjdg,erm",
    #     datasets="phm",
    #     t_nums="T5-T8",
    #     suffixes="300-1,300-5",
    #     debug_mode=True
    # )

    # 示例3：运行所有方法的pu数据集、T9-T12、后缀300-20
    # run_config = RunConfig(
    #     datasets="pu",
    #     t_nums="T9-T12",
    #     suffixes="300-20",
    #     debug_mode=False  # 实际执行
    # )

    # 3. 筛选需要运行的配置
    target_paths = filter_config_paths(ALL_CONFIG_PATHS, run_config)
    if not target_paths:
        print("⚠️  没有匹配到任何配置文件，请检查筛选条件！")
        sys.exit(0)

    # 4. 执行测试
    print(f"\n共匹配到 {len(target_paths)} 个配置文件待运行：")
    for i, path in enumerate(target_paths, 1):
        print(f"  {i}. {path}")

    confirm = input("\n是否确认运行？(y/n): ")
    if confirm.lower() != "y":
        print("🚫 取消运行")
        sys.exit(0)

    for path in target_paths:
        run_model_test(path, run_config.debug_mode)
    print("\n🎉 所有匹配的模型测试执行完毕！")