# run.py
import sys
import os

# 将项目根目录添加到Python路径（最优先）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 现在可以安全导入
from main import main

if __name__ == '__main__':
    main()