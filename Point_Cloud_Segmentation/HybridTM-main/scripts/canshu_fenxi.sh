#!/bin/bash

# 依次执行训练命令，前一个完成后才会执行下一个
echo "开始执行训练任务..."
# angle_weight=0.1
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_angle0.1 -n semseg-oacnns-TopoFusion-ABC_angle0.1-newdata

# angle_weight=0.2
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_angle0.2 -n semseg-oacnns-TopoFusion-ABC_angle0.2-newdata

# angle_weight=0.3
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_angle0.3 -n semseg-oacnns-TopoFusion-ABC_angle0.3-newdata

# angle_weight=0.4
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_angle0.4 -n semseg-oacnns-TopoFusion-ABC_angle0.4-newdata

# angle_weight=0.5
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_angle0.5 -n semseg-oacnns-TopoFusion-ABC_angle0.5-newdata

# angle_weight=0.6
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_angle0.6 -n semseg-oacnns-TopoFusion-ABC_angle0.6-newdata

# angle_weight=0.7
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_angle0.7 -n semseg-oacnns-TopoFusion-ABC_angle0.7-newdata

# angle_weight=0.8
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_angle0.8 -n semseg-oacnns-TopoFusion-ABC_angle0.8-newdata

# angle_weight=0.9
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_angle0.9 -n semseg-oacnns-TopoFusion-ABC_angle0.9-newdata

# angle_weight=1.0（基准）
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_angle1.0 -n semseg-oacnns-TopoFusion-ABC_angle1.0-newdata

# angle_weight=1.1（基准）
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_angle1.1 -n semseg-oacnns-TopoFusion-ABC_angle1.1-newdata

# angle_weight=1.2（基准）
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_angle1.2 -n semseg-oacnns-TopoFusion-ABC_angle1.2-newdata

# angle_weight=1.3（基准）
sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_angle1.3 -n semseg-oacnns-TopoFusion-ABC_angle1.3-newdata

# angle_weight=1.4（基准）
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC_angle1.4 -n semseg-oacnns-TopoFusion-ABC_angle1.4-newdata


echo "所有训练任务执行完毕！"
