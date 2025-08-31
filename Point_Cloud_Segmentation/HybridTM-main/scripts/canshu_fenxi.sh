#!/bin/bash

# 依次执行训练命令，前一个完成后才会执行下一个
echo "开始执行训练任务..."
# k=4
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_k4 -n semseg-oacnns-TopoFusion-ABC_k4

# k=8
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_k8 -n semseg-oacnns-TopoFusion-ABC_k8

# k=16（原配置，作为基准）
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_k16 -n semseg-oacnns-TopoFusion-ABC_k16

# k=24
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_k24 -n semseg-oacnns-TopoFusion-ABC_k24

# k=32
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_k32 -n semseg-oacnns-TopoFusion-ABC_k32

echo "所有训练任务执行完毕！"


# angle_weight=0.4
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_angle0.4 -n semseg-oacnns-TopoFusion-ABC_angle0.4

# angle_weight=0.6
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_angle0.6 -n semseg-oacnns-TopoFusion-ABC_angle0.6

# angle_weight=0.8
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_angle0.8 -n semseg-oacnns-TopoFusion-ABC_angle0.8

# angle_weight=1.0（基准）
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_angle1.0 -n semseg-oacnns-TopoFusion-ABC_angle1.0

# angle_weight=1.2
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_angle1.2 -n semseg-oacnns-TopoFusion-ABC_angle1.2

# angle_weight=1.4
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_angle1.4 -n semseg-oacnns-TopoFusion-ABC_angle1.4

echo "所有训练任务执行完毕！"