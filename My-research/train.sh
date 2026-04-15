#!/bin/bash
# 批量执行指令的脚本
set -e  # 关键：任意指令失败则脚本立即退出（可选，增强安全性）

#echo "======== 开始执行第一个脚本 erm========"
#./train-erm.sh

echo "======== 开始执行第二个脚本 mldg========"
./train-mldg.sh

#echo "======== 开始执行第三个脚本 vrex========"
#./train-vrex.sh
#
#echo "======== 开始执行第四个脚本 dfdn========"
#./train-dfdn.sh
#
#echo "======== 开始执行第五个脚本 darm========"
#./train-darm.sh
#
#echo "======== 开始执行第六个脚本 sdagn========"
#./train-sdagn.sh
#
#echo "======== 开始执行第七个脚本 dpjdg========"
#./train-dpjdg.sh
#
#echo "======== 开始执行第八个脚本 masfd========"
#./train-masfd.sh
#
echo "======== 开始执行第九个脚本 masfd========"
./train-mcpdg.sh



echo "======== 所有脚本执行完成 ========"
shutdown -h now