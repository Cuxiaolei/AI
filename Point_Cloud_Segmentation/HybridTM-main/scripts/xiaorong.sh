#!/bin/bash

# 依次执行训练命令，前一个完成后才会执行下一个
echo "开始执行训练任务..."

## 第一个训练任务
#echo "执行第一个训练任务: semseg-oacnns-TopoFusion-ABC"
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC -n semseg-oacnns-TopoFusion-ABC

# 第二个训练任务
echo "执行第二个训练任务: semseg-oacnns-TopoFusion-A"
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-A -n semseg-oacnns-TopoFusion-A

# 第三个训练任务
echo "执行第三个训练任务: semseg-oacnns-TopoFusion-B"
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-B -n semseg-oacnns-TopoFusion-B

# 第四个训练任务
echo "执行第四个训练任务: semseg-oacnns-TopoFusion-C"
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-C -n semseg-oacnns-TopoFusion-C

## 第五个训练任务
#echo "执行第五个训练任务: semseg-oacnns-TopoFusion-AB"
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-AB -n semseg-oacnns-TopoFusion-AB
#
## 第六个训练任务
#echo "执行第六个训练任务: semseg-oacnns-TopoFusion-AC"
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-AC -n semseg-oacnns-TopoFusion-AC
#
## 第七个训练任务
#echo "执行第七个训练任务: semseg-oacnns-TopoFusion-BC"
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-BC -n semseg-oacnns-TopoFusion-BC

echo "所有训练任务执行完毕！"

# 所有任务完成后自动关机
shutdown -h now
