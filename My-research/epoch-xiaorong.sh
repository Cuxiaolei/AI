#!/bin/bash
# 批量执行指令的脚本
set -e  # 关键：任意指令失败则脚本立即退出（可选，增强安全性）

echo "phm"
python -m src.main --configs src/configs/mcpdgE1/mcpdgE1_phm_T2_5-1.yaml
python -m src.main --configs src/configs/mcpdgE2/mcpdgE2_phm_T2_5-1.yaml
python -m src.main --configs src/configs/mcpdgE3/mcpdgE3_phm_T2_5-1.yaml
python -m src.main --configs src/configs/mcpdgE4/mcpdgE4_phm_T2_5-1.yaml
python -m src.main --configs src/configs/mcpdgE5/mcpdgE5_phm_T2_5-1.yaml
python -m src.main --configs src/configs/mcpdgE6/mcpdgE6_phm_T2_5-1.yaml
python -m src.main --configs src/configs/mcpdgE7/mcpdgE7_phm_T2_5-1.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_phm_T2_5-1.yaml


echo "pu"
python -m src.main --configs src/configs/mcpdgE1/mcpdgE1_phm_T6_5-1.yaml
python -m src.main --configs src/configs/mcpdgE2/mcpdgE2_phm_T6_5-1.yaml
python -m src.main --configs src/configs/mcpdgE3/mcpdgE3_phm_T6_5-1.yaml
python -m src.main --configs src/configs/mcpdgE4/mcpdgE4_phm_T6_5-1.yaml
python -m src.main --configs src/configs/mcpdgE5/mcpdgE5_phm_T6_5-1.yaml
python -m src.main --configs src/configs/mcpdgE6/mcpdgE6_phm_T6_5-1.yaml
python -m src.main --configs src/configs/mcpdgE7/mcpdgE7_phm_T6_5-1.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_phm_T6_5-1.yaml


