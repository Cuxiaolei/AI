import os
import sys


def print_directory_tree(startpath, prefix=''):
    """
    递归打印目录树结构
    :param startpath: 起始目录路径
    :param prefix: 前缀符号，用于层级展示
    """
    # 过滤掉系统隐藏的临时文件（可选，可注释）
    try:
        entries = [e for e in os.listdir(startpath) if not e.startswith('.DS_Store')]
    except PermissionError:
        print(f'{prefix}└── [权限不足，无法访问]')
        return

    # 排序规则：目录在前，文件在后，按名称字母序排列
    entries.sort(key=lambda x: (not os.path.isdir(os.path.join(startpath, x)), x.lower()))
    entries_count = len(entries)

    for i, entry in enumerate(entries):
        full_path = os.path.join(startpath, entry)
        is_last = i == entries_count - 1  # 是否是当前层级最后一个条目

        # 定义层级符号（树形结构更直观）
        if is_last:
            symbol = '└── '
            next_prefix = prefix + '    '  # 最后一个条目后续无竖线
        else:
            symbol = '├── '
            next_prefix = prefix + '│   '  # 非最后一个条目保留竖线

        # 处理目录
        if os.path.isdir(full_path):
            print(f'{prefix}{symbol}{entry}/')  # 目录末尾加/区分
            try:
                # 递归遍历子目录
                print_directory_tree(full_path, next_prefix)
            except PermissionError:
                print(f'{next_prefix}└── [权限不足，无法访问]')
        # 处理文件
        else:
            print(f'{prefix}{symbol}{entry}')


if __name__ == '__main__':
    # 处理命令行参数：支持指定目标目录，默认当前目录
    if len(sys.argv) > 1:
        start_dir = sys.argv[1]
        if not os.path.isdir(start_dir):
            print(f"❌ 错误：目录 '{start_dir}' 不存在或不是有效目录")
            sys.exit(1)
    else:
        start_dir = os.getcwd()  # 默认当前工作目录

    # 打印标题和根目录
    print(f'\n📂 目录树: {start_dir}')
    # 处理空目录
    if not os.listdir(start_dir):
        print('└── (空目录)')
    else:
        print_directory_tree(start_dir)
    print()