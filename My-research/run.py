# -*- coding: utf-8 -*-
"""
项目根目录入口：保证你在 PyCharm 直接 Run 也能找到 src 包。
建议运行方式：
- PyCharm 直接运行 run.py
- 或命令行：python run.py --config configs/base.yaml configs/experiments/xxx.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.main import main  # noqa: E402


if __name__ == "__main__":
    main()
