#!/bin/bash
# 批量执行指令的脚本
set -e  # 关键：任意指令失败则脚本立即退出（可选，增强安全性）

echo "duibi"
./epoch-duibi.sh

echo "xiaorong"
./epoch-xiaorong.sh




echo "======== 所有脚本执行完成 ========"
shutdown -h now
