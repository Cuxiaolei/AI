#!/bin/bash
# 批量执行指令的脚本
set -e  # 关键：任意指令失败则脚本立即退出（可选，增强安全性）

echo "phm"
python -m src.main --configs src/configs/darm/darm_phm_T2_1-1.yaml
python -m src.main --configs src/configs/darm/darm_phm_T2_5-1.yaml
python -m src.main --configs src/configs/dfdn/dfdn_phm_T2_1-1.yaml
python -m src.main --configs src/configs/dfdn/dfdn_phm_T2_5-1.yaml
python -m src.main --configs src/configs/dfdn/dpjdg_phm_T2_1-1.yaml
python -m src.main --configs src/configs/dfdn/dpjdg_phm_T2_5-1.yaml
python -m src.main --configs src/configs/dfdn/erm_phm_T2_1-1.yaml
python -m src.main --configs src/configs/dfdn/erm_phm_T2_5-1.yaml
python -m src.main --configs src/configs/dfdn/masfd_phm_T2_1-1.yaml
python -m src.main --configs src/configs/dfdn/masfd_phm_T2_5-1.yaml
python -m src.main --configs src/configs/dfdn/mldg_phm_T2_1-1.yaml
python -m src.main --configs src/configs/dfdn/mldg_phm_T2_5-1.yaml
python -m src.main --configs src/configs/dfdn/sdagn_phm_T2_1-1.yaml
python -m src.main --configs src/configs/dfdn/sdagn_phm_T2_5-1.yaml
python -m src.main --configs src/configs/dfdn/vrex_phm_T2_1-1.yaml
python -m src.main --configs src/configs/dfdn/vrex_phm_T2_5-1.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_phm_T2_1-1.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_phm_T2_5-1.yaml


echo "pu"
python -m src.main --configs src/configs/darm/darm_pu_T2_1-1.yaml
python -m src.main --configs src/configs/darm/darm_pu_T2_5-1.yaml
python -m src.main --configs src/configs/dfdn/dfdn_pu_T2_1-1.yaml
python -m src.main --configs src/configs/dfdn/dfdn_pu_T2_5-1.yaml
python -m src.main --configs src/configs/dfdn/dpjdg_pu_T2_1-1.yaml
python -m src.main --configs src/configs/dfdn/dpjdg_pu_T2_5-1.yaml
python -m src.main --configs src/configs/dfdn/erm_pu_T2_1-1.yaml
python -m src.main --configs src/configs/dfdn/erm_pu_T2_5-1.yaml
python -m src.main --configs src/configs/dfdn/masfd_pu_T2_1-1.yaml
python -m src.main --configs src/configs/dfdn/masfd_pu_T2_5-1.yaml
python -m src.main --configs src/configs/dfdn/mldg_pu_T2_1-1.yaml
python -m src.main --configs src/configs/dfdn/mldg_pu_T2_5-1.yaml
python -m src.main --configs src/configs/dfdn/sdagn_pu_T2_1-1.yaml
python -m src.main --configs src/configs/dfdn/sdagn_pu_T2_5-1.yaml
python -m src.main --configs src/configs/dfdn/vrex_pu_T2_1-1.yaml
python -m src.main --configs src/configs/dfdn/vrex_pu_T2_5-1.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_pu_T2_1-1.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_pu_T2_5-1.yaml




echo "======== 所有脚本执行完成 ========"
shutdown -h now
