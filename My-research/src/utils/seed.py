# -*- coding: utf-8 -*-
"""
随机种子设置：python / numpy / torch（可选 deterministic）
"""

from __future__ import annotations

import os
import random
from typing import Optional


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
