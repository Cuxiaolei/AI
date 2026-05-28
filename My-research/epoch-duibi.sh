#!/bin/bash
# 批量执行指令的脚本
set -e  # 关键：任意指令失败则脚本立即退出（可选，增强安全性）

echo "phm"
#python -m src.main --configs src/configs/darm/darm_phm_T2_1-1.yaml
#python -m src.main --configs src/configs/darm/darm_phm_T2_5-1.yaml
#python -m src.main --configs src/configs/dfdn/dfdn_phm_T2_1-1.yaml
#python -m src.main --configs src/configs/dfdn/dfdn_phm_T2_5-1.yaml
#python -m src.main --configs src/configs/dpjdg/dpjdg_phm_T2_1-1.yaml
#python -m src.main --configs src/configs/dpjdg/dpjdg_phm_T2_5-1.yaml
#python -m src.main --configs src/configs/erm/erm_phm_T2_1-1.yaml
#python -m src.main --configs src/configs/erm/erm_phm_T2_5-1.yaml
#python -m src.main --configs src/configs/masfd/masfd_phm_T2_1-1.yaml
#python -m src.main --configs src/configs/masfd/masfd_phm_T2_5-1.yaml
#python -m src.main --configs src/configs/mldg/mldg_phm_T2_1-1.yaml
#python -m src.main --configs src/configs/mldg/mldg_phm_T2_5-1.yaml
#python -m src.main --configs src/configs/sdagn/sdagn_phm_T2_1-1.yaml
#python -m src.main --configs src/configs/sdagn/sdagn_phm_T2_5-1.yaml
#python -m src.main --configs src/configs/vrex/vrex_phm_T2_1-1.yaml
#python -m src.main --configs src/configs/vrex/vrex_phm_T2_5-1.yaml
#python -m src.main --configs src/configs/mcpdg/mcpdg_phm_T2_1-1.yaml
#python -m src.main --configs src/configs/mcpdg/mcpdg_phm_T2_5-1.yaml


echo "pu"
python -m src.main --configs src/configs/darm/darm_pu_T6_1-1.yaml
python -m src.main --configs src/configs/darm/darm_pu_T6_5-1.yaml
python -m src.main --configs src/configs/dfdn/dfdn_pu_T6_1-1.yaml
python -m src.main --configs src/configs/dfdn/dfdn_pu_T6_5-1.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_pu_T6_1-1.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_pu_T6_5-1.yaml
python -m src.main --configs src/configs/erm/erm_pu_T6_1-1.yaml
python -m src.main --configs src/configs/erm/erm_pu_T6_5-1.yaml
python -m src.main --configs src/configs/masfd/masfd_pu_T6_1-1.yaml
python -m src.main --configs src/configs/masfd/masfd_pu_T6_5-1.yaml
python -m src.main --configs src/configs/mldg/mldg_pu_T6_1-1.yaml
python -m src.main --configs src/configs/mldg/mldg_pu_T6_5-1.yaml
python -m src.main --configs src/configs/sdagn/sdagn_pu_T6_1-1.yaml
python -m src.main --configs src/configs/sdagn/sdagn_pu_T6_5-1.yaml
python -m src.main --configs src/configs/vrex/vrex_pu_T6_1-1.yaml
python -m src.main --configs src/configs/vrex/vrex_pu_T6_5-1.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_pu_T6_1-1.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_pu_T6_5-1.yaml
