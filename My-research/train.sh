#!/bin/bash
# 批量执行指令的脚本
set -e  # 关键：任意指令失败则脚本立即退出（可选，增强安全性）

python -m src.main --configs src/configs/mcpdg_ablate/mcpdg_a0_phm_T5_300-5.yaml
python -m src.main --configs src/configs/mcpdg_ablate/mcpdg_a1_phm_T5_300-5.yaml
python -m src.main --configs src/configs/mcpdg_ablate/mcpdg_a2_phm_T5_300-5.yaml
python -m src.main --configs src/configs/mcpdg_ablate/mcpdg_a3_phm_T5_300-5.yaml