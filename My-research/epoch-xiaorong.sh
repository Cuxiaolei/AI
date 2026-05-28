#!/bin/bash
# 批量执行指令的脚本
set -e  # 关键：任意指令失败则脚本立即退出（可选，增强安全性）

echo "phm"
python -m src.main --configs src/configs/mcpdgE1/mcpdgE1_phm_T2_5-1.yaml
python -m src.main --configs src/configs/mcpdgE1/mcpdgE2_phm_T2_5-1.yaml
python -m src.main --configs src/configs/mcpdgE1/mcpdgE3_phm_T2_5-1.yaml
python -m src.main --configs src/configs/mcpdgE1/mcpdgE4_phm_T2_5-1.yaml
python -m src.main --configs src/configs/mcpdgE1/mcpdgE5_phm_T2_5-1.yaml
python -m src.main --configs src/configs/mcpdgE1/mcpdgE6_phm_T2_5-1.yaml
python -m src.main --configs src/configs/mcpdgE1/mcpdgE7_phm_T2_5-1.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_phm_T2_5-1.yaml


echo "pu"
python -m src.main --configs src/configs/mcpdgE1/mcpdgE1_phm_T6_5-1.yaml
python -m src.main --configs src/configs/mcpdgE1/mcpdgE2_phm_T6_5-1.yaml
python -m src.main --configs src/configs/mcpdgE1/mcpdgE3_phm_T6_5-1.yaml
python -m src.main --configs src/configs/mcpdgE1/mcpdgE4_phm_T6_5-1.yaml
python -m src.main --configs src/configs/mcpdgE1/mcpdgE5_phm_T6_5-1.yaml
python -m src.main --configs src/configs/mcpdgE1/mcpdgE6_phm_T6_5-1.yaml
python -m src.main --configs src/configs/mcpdgE1/mcpdgE7_phm_T6_5-1.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_phm_T6_5-1.yaml


