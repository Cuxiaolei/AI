#!/bin/bash

# 依次执行训练命令，前一个完成后才会执行下一个
echo "开始执行训练任务..."

# 第一个训练任务
echo "执行第一个训练任务: semseg-oacnns-PFEC-base"
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-PFEC-base -n semseg-oacnns-PFEC-base

# 第二个训练任务
echo "执行第二个训练任务: semseg-oacnns-PFEC-A"
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-PFEC-A -n semseg-oacnns-PFEC-A

# 第三个训练任务
echo "执行第三个训练任务: semseg-oacnns-PFEC-B"
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-PFEC-B -n semseg-oacnns-PFEC-B

# 第四个训练任务
echo "执行第四个训练任务: semseg-oacnns-PFEC-C"
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-PFEC-C -n semseg-oacnns-PFEC-C

# 第五个训练任务
echo "执行第五个训练任务: semseg-oacnns-PFEC-AB"
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-PFEC-AB -n semseg-oacnns-PFEC-AB

# 第六个训练任务
echo "执行第六个训练任务: semseg-oacnns-PFEC-AC"
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-PFEC-AC -n semseg-oacnns-PFEC-AC

# 第七个训练任务
echo "执行第七个训练任务: semseg-oacnns-PFEC-BC"
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-PFEC-BC -n semseg-oacnns-PFEC-BC

echo "所有训练任务执行完毕！"
