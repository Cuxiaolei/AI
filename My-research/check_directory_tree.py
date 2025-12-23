import os
import sys


def print_directory_tree(startpath, root_dir, prefix=''):
    """
    递归打印目录树结构
    :param startpath: 当前遍历的目录路径
    :param root_dir: 初始根目录（用于精准判断「根目录下的data」）
    :param prefix: 层级前缀符号
    """
    # 过滤系统隐藏文件（如.DS_Store）
    try:
        all_entries = [e for e in os.listdir(startpath) if not e.startswith('.DS_Store')]
    except PermissionError:
        print(f'{prefix}└── [权限不足，无法访问]')
        return

    # 关键：精准定义「根目录下的data文件夹」绝对路径（带路径分隔符，避免误匹配）
    root_data_abs = os.path.join(os.path.abspath(root_dir), 'data') + os.sep
    current_abs = os.path.abspath(startpath) + os.sep

    # 判断：仅「根目录下的data」或其任意子目录 → 过滤所有文件
    is_root_data_tree = current_abs.startswith(root_data_abs)

    if is_root_data_tree:
        # 场景1：根data及其子目录 → 只保留文件夹，过滤所有文件
        entries = [e for e in all_entries if os.path.isdir(os.path.join(startpath, e))]
    else:
        # 场景2：其他目录（包括src/data、根目录其他文件夹）→ 正常显示文件+文件夹
        entries = sorted(all_entries, key=lambda x: (not os.path.isdir(os.path.join(startpath, x)), x.lower()))

    entries_count = len(entries)
    # 空目录提示
    if entries_count == 0:
        print(f'{prefix}└── (空目录)')
        return

    for i, entry in enumerate(entries):
        full_path = os.path.join(startpath, entry)
        is_last = i == entries_count - 1
        # 定义树形符号
        symbol = '└── ' if is_last else '├── '
        next_prefix = prefix + '    ' if is_last else prefix + '│   '

        # 打印目录/文件（仅非根data树的目录会显示文件）
        if os.path.isdir(full_path):
            print(f'{prefix}{symbol}{entry}/')
            # 递归遍历子目录
            print_directory_tree(full_path, root_dir, next_prefix)
        else:
            print(f'{prefix}{symbol}{entry}')


if __name__ == '__main__':
    # 处理命令行参数：支持指定根目录，默认当前目录
    if len(sys.argv) > 1:
        start_dir = sys.argv[1]
        if not os.path.isdir(start_dir):
            print(f"❌ 错误：目录 '{start_dir}' 不存在或不是有效目录")
            sys.exit(1)
    else:
        start_dir = os.getcwd()

    # 打印目录树标题
    print(f'\n📂 目录树: {start_dir}')
    if not os.listdir(start_dir):
        print('└── (空目录)')
    else:
        print_directory_tree(start_dir, start_dir)
    print()