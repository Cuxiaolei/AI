#!/bin/bash
# 批量执行指令的脚本
set -e  # 关键：任意指令失败则脚本立即退出（可选，增强安全性）

./train-mcpdgE1.sh

./train-mcpdgE2.sh

./train-mcpdgE3.sh

./train-mcpdgE4.sh

./train-mcpdgE5.sh

./train-mcpdgE6.sh

./train-mcpdgE7.sh

./train-mcpdg.sh


echo "======== 所有脚本执行完成 ========"
shutdown -h now