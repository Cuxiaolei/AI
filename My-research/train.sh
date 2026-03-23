#!/bin/bash
# 批量执行指令的脚本
set -e  # 关键：任意指令失败则脚本立即退出（可选，增强安全性）

echo darm 模型
python -m src.main --configs src/configs/darm/darm_phm_T1_1.yaml
python -m src.main --configs src/configs/darm/darm_phm_T1_5.yaml
python -m src.main --configs src/configs/darm/darm_phm_T1_10.yaml
python -m src.main --configs src/configs/darm/darm_phm_T1_20.yaml
python -m src.main --configs src/configs/darm/darm_phm_T2_1.yaml
python -m src.main --configs src/configs/darm/darm_phm_T2_5.yaml
python -m src.main --configs src/configs/darm/darm_phm_T2_10.yaml
python -m src.main --configs src/configs/darm/darm_phm_T2_20.yaml
python -m src.main --configs src/configs/darm/darm_phm_T3_1.yaml
python -m src.main --configs src/configs/darm/darm_phm_T3_5.yaml
python -m src.main --configs src/configs/darm/darm_phm_T3_10.yaml
python -m src.main --configs src/configs/darm/darm_phm_T3_20.yaml
python -m src.main --configs src/configs/darm/darm_phm_T4_1.yaml
python -m src.main --configs src/configs/darm/darm_phm_T4_5.yaml
python -m src.main --configs src/configs/darm/darm_phm_T4_10.yaml
python -m src.main --configs src/configs/darm/darm_phm_T4_20.yaml

python -m src.main --configs src/configs/darm/darm_pu_T5_1.yaml
python -m src.main --configs src/configs/darm/darm_pu_T5_5.yaml
python -m src.main --configs src/configs/darm/darm_pu_T5_10.yaml
python -m src.main --configs src/configs/darm/darm_pu_T5_20.yaml
python -m src.main --configs src/configs/darm/darm_pu_T6_1.yaml
python -m src.main --configs src/configs/darm/darm_pu_T6_5.yaml
python -m src.main --configs src/configs/darm/darm_pu_T6_10.yaml
python -m src.main --configs src/configs/darm/darm_pu_T6_20.yaml
python -m src.main --configs src/configs/darm/darm_pu_T7_1.yaml
python -m src.main --configs src/configs/darm/darm_pu_T7_5.yaml
python -m src.main --configs src/configs/darm/darm_pu_T7_10.yaml
python -m src.main --configs src/configs/darm/darm_pu_T7_20.yaml
python -m src.main --configs src/configs/darm/darm_pu_T8_1.yaml
python -m src.main --configs src/configs/darm/darm_pu_T8_5.yaml
python -m src.main --configs src/configs/darm/darm_pu_T8_10.yaml
python -m src.main --configs src/configs/darm/darm_pu_T8_20.yaml

echo dpjdg 模型
python -m src.main --configs src/configs/dpjdg/dpjdg_phm_T1_1.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_phm_T1_5.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_phm_T1_10.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_phm_T1_20.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_phm_T2_1.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_phm_T2_5.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_phm_T2_10.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_phm_T2_20.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_phm_T3_1.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_phm_T3_5.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_phm_T3_10.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_phm_T3_20.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_phm_T4_1.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_phm_T4_5.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_phm_T4_10.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_phm_T4_20.yaml

python -m src.main --configs src/configs/dpjdg/dpjdg_pu_T5_1.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_pu_T5_5.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_pu_T5_10.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_pu_T5_20.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_pu_T6_1.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_pu_T6_5.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_pu_T6_10.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_pu_T6_20.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_pu_T7_1.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_pu_T7_5.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_pu_T7_10.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_pu_T7_20.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_pu_T8_1.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_pu_T8_5.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_pu_T8_10.yaml
python -m src.main --configs src/configs/dpjdg/dpjdg_pu_T8_20.yaml

echo erm 模型
python -m src.main --configs src/configs/erm/erm_phm_T1_1.yaml
python -m src.main --configs src/configs/erm/erm_phm_T1_5.yaml
python -m src.main --configs src/configs/erm/erm_phm_T1_10.yaml
python -m src.main --configs src/configs/erm/erm_phm_T1_20.yaml
python -m src.main --configs src/configs/erm/erm_phm_T2_1.yaml
python -m src.main --configs src/configs/erm/erm_phm_T2_5.yaml
python -m src.main --configs src/configs/erm/erm_phm_T2_10.yaml
python -m src.main --configs src/configs/erm/erm_phm_T2_20.yaml
python -m src.main --configs src/configs/erm/erm_phm_T3_1.yaml
python -m src.main --configs src/configs/erm/erm_phm_T3_5.yaml
python -m src.main --configs src/configs/erm/erm_phm_T3_10.yaml
python -m src.main --configs src/configs/erm/erm_phm_T3_20.yaml
python -m src.main --configs src/configs/erm/erm_phm_T4_1.yaml
python -m src.main --configs src/configs/erm/erm_phm_T4_5.yaml
python -m src.main --configs src/configs/erm/erm_phm_T4_10.yaml
python -m src.main --configs src/configs/erm/erm_phm_T4_20.yaml

python -m src.main --configs src/configs/erm/erm_pu_T5_1.yaml
python -m src.main --configs src/configs/erm/erm_pu_T5_5.yaml
python -m src.main --configs src/configs/erm/erm_pu_T5_10.yaml
python -m src.main --configs src/configs/erm/erm_pu_T5_20.yaml
python -m src.main --configs src/configs/erm/erm_pu_T6_1.yaml
python -m src.main --configs src/configs/erm/erm_pu_T6_5.yaml
python -m src.main --configs src/configs/erm/erm_pu_T6_10.yaml
python -m src.main --configs src/configs/erm/erm_pu_T6_20.yaml
python -m src.main --configs src/configs/erm/erm_pu_T7_1.yaml
python -m src.main --configs src/configs/erm/erm_pu_T7_5.yaml
python -m src.main --configs src/configs/erm/erm_pu_T7_10.yaml
python -m src.main --configs src/configs/erm/erm_pu_T7_20.yaml
python -m src.main --configs src/configs/erm/erm_pu_T8_1.yaml
python -m src.main --configs src/configs/erm/erm_pu_T8_5.yaml
python -m src.main --configs src/configs/erm/erm_pu_T8_10.yaml
python -m src.main --configs src/configs/erm/erm_pu_T8_20.yaml

echo groupdro 模型
python -m src.main --configs src/configs/groupdro/groupdro_phm_T1_1.yaml
python -m src.main --configs src/configs/groupdro/groupdro_phm_T1_5.yaml
python -m src.main --configs src/configs/groupdro/groupdro_phm_T1_10.yaml
python -m src.main --configs src/configs/groupdro/groupdro_phm_T1_20.yaml
python -m src.main --configs src/configs/groupdro/groupdro_phm_T2_1.yaml
python -m src.main --configs src/configs/groupdro/groupdro_phm_T2_5.yaml
python -m src.main --configs src/configs/groupdro/groupdro_phm_T2_10.yaml
python -m src.main --configs src/configs/groupdro/groupdro_phm_T2_20.yaml
python -m src.main --configs src/configs/groupdro/groupdro_phm_T3_1.yaml
python -m src.main --configs src/configs/groupdro/groupdro_phm_T3_5.yaml
python -m src.main --configs src/configs/groupdro/groupdro_phm_T3_10.yaml
python -m src.main --configs src/configs/groupdro/groupdro_phm_T3_20.yaml
python -m src.main --configs src/configs/groupdro/groupdro_phm_T4_1.yaml
python -m src.main --configs src/configs/groupdro/groupdro_phm_T4_5.yaml
python -m src.main --configs src/configs/groupdro/groupdro_phm_T4_10.yaml
python -m src.main --configs src/configs/groupdro/groupdro_phm_T4_20.yaml

python -m src.main --configs src/configs/groupdro/groupdro_pu_T5_1.yaml
python -m src.main --configs src/configs/groupdro/groupdro_pu_T5_5.yaml
python -m src.main --configs src/configs/groupdro/groupdro_pu_T5_10.yaml
python -m src.main --configs src/configs/groupdro/groupdro_pu_T5_20.yaml
python -m src.main --configs src/configs/groupdro/groupdro_pu_T6_1.yaml
python -m src.main --configs src/configs/groupdro/groupdro_pu_T6_5.yaml
python -m src.main --configs src/configs/groupdro/groupdro_pu_T6_10.yaml
python -m src.main --configs src/configs/groupdro/groupdro_pu_T6_20.yaml
python -m src.main --configs src/configs/groupdro/groupdro_pu_T7_1.yaml
python -m src.main --configs src/configs/groupdro/groupdro_pu_T7_5.yaml
python -m src.main --configs src/configs/groupdro/groupdro_pu_T7_10.yaml
python -m src.main --configs src/configs/groupdro/groupdro_pu_T7_20.yaml
python -m src.main --configs src/configs/groupdro/groupdro_pu_T8_1.yaml
python -m src.main --configs src/configs/groupdro/groupdro_pu_T8_5.yaml
python -m src.main --configs src/configs/groupdro/groupdro_pu_T8_10.yaml
python -m src.main --configs src/configs/groupdro/groupdro_pu_T8_20.yaml

echo irm 模型
python -m src.main --configs src/configs/irm/irm_phm_T1_1.yaml
python -m src.main --configs src/configs/irm/irm_phm_T1_5.yaml
python -m src.main --configs src/configs/irm/irm_phm_T1_10.yaml
python -m src.main --configs src/configs/irm/irm_phm_T1_20.yaml
python -m src.main --configs src/configs/irm/irm_phm_T2_1.yaml
python -m src.main --configs src/configs/irm/irm_phm_T2_5.yaml
python -m src.main --configs src/configs/irm/irm_phm_T2_10.yaml
python -m src.main --configs src/configs/irm/irm_phm_T2_20.yaml
python -m src.main --configs src/configs/irm/irm_phm_T3_1.yaml
python -m src.main --configs src/configs/irm/irm_phm_T3_5.yaml
python -m src.main --configs src/configs/irm/irm_phm_T3_10.yaml
python -m src.main --configs src/configs/irm/irm_phm_T3_20.yaml
python -m src.main --configs src/configs/irm/irm_phm_T4_1.yaml
python -m src.main --configs src/configs/irm/irm_phm_T4_5.yaml
python -m src.main --configs src/configs/irm/irm_phm_T4_10.yaml
python -m src.main --configs src/configs/irm/irm_phm_T4_20.yaml

python -m src.main --configs src/configs/irm/irm_pu_T5_1.yaml
python -m src.main --configs src/configs/irm/irm_pu_T5_5.yaml
python -m src.main --configs src/configs/irm/irm_pu_T5_10.yaml
python -m src.main --configs src/configs/irm/irm_pu_T5_20.yaml
python -m src.main --configs src/configs/irm/irm_pu_T6_1.yaml
python -m src.main --configs src/configs/irm/irm_pu_T6_5.yaml
python -m src.main --configs src/configs/irm/irm_pu_T6_10.yaml
python -m src.main --configs src/configs/irm/irm_pu_T6_20.yaml
python -m src.main --configs src/configs/irm/irm_pu_T7_1.yaml
python -m src.main --configs src/configs/irm/irm_pu_T7_5.yaml
python -m src.main --configs src/configs/irm/irm_pu_T7_10.yaml
python -m src.main --configs src/configs/irm/irm_pu_T7_20.yaml
python -m src.main --configs src/configs/irm/irm_pu_T8_1.yaml
python -m src.main --configs src/configs/irm/irm_pu_T8_5.yaml
python -m src.main --configs src/configs/irm/irm_pu_T8_10.yaml
python -m src.main --configs src/configs/irm/irm_pu_T8_20.yaml

echo mixstyle 模型
python -m src.main --configs src/configs/mixstyle/mixstyle_phm_T1_1.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_phm_T1_5.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_phm_T1_10.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_phm_T1_20.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_phm_T2_1.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_phm_T2_5.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_phm_T2_10.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_phm_T2_20.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_phm_T3_1.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_phm_T3_5.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_phm_T3_10.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_phm_T3_20.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_phm_T4_1.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_phm_T4_5.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_phm_T4_10.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_phm_T4_20.yaml

python -m src.main --configs src/configs/mixstyle/mixstyle_pu_T5_1.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_pu_T5_5.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_pu_T5_10.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_pu_T5_20.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_pu_T6_1.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_pu_T6_5.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_pu_T6_10.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_pu_T6_20.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_pu_T7_1.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_pu_T7_5.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_pu_T7_10.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_pu_T7_20.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_pu_T8_1.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_pu_T8_5.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_pu_T8_10.yaml
python -m src.main --configs src/configs/mixstyle/mixstyle_pu_T8_20.yaml

echo mldg 模型
python -m src.main --configs src/configs/mldg/mldg_phm_T1_1.yaml
python -m src.main --configs src/configs/mldg/mldg_phm_T1_5.yaml
python -m src.main --configs src/configs/mldg/mldg_phm_T1_10.yaml
python -m src.main --configs src/configs/mldg/mldg_phm_T1_20.yaml
python -m src.main --configs src/configs/mldg/mldg_phm_T2_1.yaml
python -m src.main --configs src/configs/mldg/mldg_phm_T2_5.yaml
python -m src.main --configs src/configs/mldg/mldg_phm_T2_10.yaml
python -m src.main --configs src/configs/mldg/mldg_phm_T2_20.yaml
python -m src.main --configs src/configs/mldg/mldg_phm_T3_1.yaml
python -m src.main --configs src/configs/mldg/mldg_phm_T3_5.yaml
python -m src.main --configs src/configs/mldg/mldg_phm_T3_10.yaml
python -m src.main --configs src/configs/mldg/mldg_phm_T3_20.yaml
python -m src.main --configs src/configs/mldg/mldg_phm_T4_1.yaml
python -m src.main --configs src/configs/mldg/mldg_phm_T4_5.yaml
python -m src.main --configs src/configs/mldg/mldg_phm_T4_10.yaml
python -m src.main --configs src/configs/mldg/mldg_phm_T4_20.yaml

python -m src.main --configs src/configs/mldg/mldg_pu_T5_1.yaml
python -m src.main --configs src/configs/mldg/mldg_pu_T5_5.yaml
python -m src.main --configs src/configs/mldg/mldg_pu_T5_10.yaml
python -m src.main --configs src/configs/mldg/mldg_pu_T5_20.yaml
python -m src.main --configs src/configs/mldg/mldg_pu_T6_1.yaml
python -m src.main --configs src/configs/mldg/mldg_pu_T6_5.yaml
python -m src.main --configs src/configs/mldg/mldg_pu_T6_10.yaml
python -m src.main --configs src/configs/mldg/mldg_pu_T6_20.yaml
python -m src.main --configs src/configs/mldg/mldg_pu_T7_1.yaml
python -m src.main --configs src/configs/mldg/mldg_pu_T7_5.yaml
python -m src.main --configs src/configs/mldg/mldg_pu_T7_10.yaml
python -m src.main --configs src/configs/mldg/mldg_pu_T7_20.yaml
python -m src.main --configs src/configs/mldg/mldg_pu_T8_1.yaml
python -m src.main --configs src/configs/mldg/mldg_pu_T8_5.yaml
python -m src.main --configs src/configs/mldg/mldg_pu_T8_10.yaml
python -m src.main --configs src/configs/mldg/mldg_pu_T8_20.yaml

echo vrex 模型
python -m src.main --configs src/configs/vrex/vrex_phm_T1_1.yaml
python -m src.main --configs src/configs/vrex/vrex_phm_T1_5.yaml
python -m src.main --configs src/configs/vrex/vrex_phm_T1_10.yaml
python -m src.main --configs src/configs/vrex/vrex_phm_T1_20.yaml
python -m src.main --configs src/configs/vrex/vrex_phm_T2_1.yaml
python -m src.main --configs src/configs/vrex/vrex_phm_T2_5.yaml
python -m src.main --configs src/configs/vrex/vrex_phm_T2_10.yaml
python -m src.main --configs src/configs/vrex/vrex_phm_T2_20.yaml
python -m src.main --configs src/configs/vrex/vrex_phm_T3_1.yaml
python -m src.main --configs src/configs/vrex/vrex_phm_T3_5.yaml
python -m src.main --configs src/configs/vrex/vrex_phm_T3_10.yaml
python -m src.main --configs src/configs/vrex/vrex_phm_T3_20.yaml
python -m src.main --configs src/configs/vrex/vrex_phm_T4_1.yaml
python -m src.main --configs src/configs/vrex/vrex_phm_T4_5.yaml
python -m src.main --configs src/configs/vrex/vrex_phm_T4_10.yaml
python -m src.main --configs src/configs/vrex/vrex_phm_T4_20.yaml

python -m src.main --configs src/configs/vrex/vrex_pu_T5_1.yaml
python -m src.main --configs src/configs/vrex/vrex_pu_T5_5.yaml
python -m src.main --configs src/configs/vrex/vrex_pu_T5_10.yaml
python -m src.main --configs src/configs/vrex/vrex_pu_T5_20.yaml
python -m src.main --configs src/configs/vrex/vrex_pu_T6_1.yaml
python -m src.main --configs src/configs/vrex/vrex_pu_T6_5.yaml
python -m src.main --configs src/configs/vrex/vrex_pu_T6_10.yaml
python -m src.main --configs src/configs/vrex/vrex_pu_T6_20.yaml
python -m src.main --configs src/configs/vrex/vrex_pu_T7_1.yaml
python -m src.main --configs src/configs/vrex/vrex_pu_T7_5.yaml
python -m src.main --configs src/configs/vrex/vrex_pu_T7_10.yaml
python -m src.main --configs src/configs/vrex/vrex_pu_T7_20.yaml
python -m src.main --configs src/configs/vrex/vrex_pu_T8_1.yaml
python -m src.main --configs src/configs/vrex/vrex_pu_T8_5.yaml
python -m src.main --configs src/configs/vrex/vrex_pu_T8_10.yaml
python -m src.main --configs src/configs/vrex/vrex_pu_T8_20.yaml

echo mcpdg 模型
python -m src.main --configs src/configs/mcpdg/mcpdg_phm_T1_1.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_phm_T1_5.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_phm_T1_10.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_phm_T1_20.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_phm_T2_1.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_phm_T2_5.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_phm_T2_10.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_phm_T2_20.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_phm_T3_1.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_phm_T3_5.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_phm_T3_10.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_phm_T3_20.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_phm_T4_1.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_phm_T4_5.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_phm_T4_10.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_phm_T4_20.yaml

python -m src.main --configs src/configs/mcpdg/mcpdg_pu_T5_1.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_pu_T5_5.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_pu_T5_10.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_pu_T5_20.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_pu_T6_1.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_pu_T6_5.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_pu_T6_10.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_pu_T6_20.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_pu_T7_1.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_pu_T7_5.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_pu_T7_10.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_pu_T7_20.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_pu_T8_1.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_pu_T8_5.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_pu_T8_10.yaml
python -m src.main --configs src/configs/mcpdg/mcpdg_pu_T8_20.yaml

shutdown -h now