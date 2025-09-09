#!/bin/bash

# 依次执行训练命令，前一个完成后才会执行下一个
echo "开始执行训练任务..."

## 第一个训练任务
#echo "执行第一个训练任务: semseg-oacnns-TopoFusion-ABC"
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-ABC -n semseg-oacnns-TopoFusion-ABC

#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-A-angle_weight0.6 -n semseg-oacnns-TopoFusion-A-angle_weight0.6
#
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-A-angle_weight0.8 -n semseg-oacnns-TopoFusion-A-angle_weight0.8
#
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-A-angle_weight1.0 -n semseg-oacnns-TopoFusion-A-angle_weight1.0
#
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-A-angle_weight1.2 -n semseg-oacnns-TopoFusion-A-angle_weight1.2
#
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-A-angle_weight1.4 -n semseg-oacnns-TopoFusion-A-angle_weight1.4


#
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-B-attn_hidden_dim4 -n semseg-oacnns-TopoFusion-B-attn_hidden_dim4
#
##sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-B-attn_hidden_dim8 -n semseg-oacnns-TopoFusion-B-attn_hidden_dim8
#
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-B-attn_hidden_dim10 -n semseg-oacnns-TopoFusion-B-attn_hidden_dim10
#
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-B-attn_hidden_dim12 -n semseg-oacnns-TopoFusion-B-attn_hidden_dim12
#
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-B-attn_hidden_dim14 -n semseg-oacnns-TopoFusion-B-attn_hidden_dim14
#
##sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-B-attn_hidden_dim16 -n semseg-oacnns-TopoFusion-B-attn_hidden_dim16
#
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-B-attn_hidden_dim20 -n semseg-oacnns-TopoFusion-B-attn_hidden_dim20
#
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-B-attn_hidden_dim24 -n semseg-oacnns-TopoFusion-B-attn_hidden_dim24



#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-C-physical_weight0.04 -n semseg-oacnns-TopoFusion-C-physical_weight0.04
#
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-C-physical_weight0.08 -n semseg-oacnns-TopoFusion-C-physical_weight0.08
#
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-C-physical_weight0.12 -n semseg-oacnns-TopoFusion-C-physical_weight0.12
#
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-C-physical_weight0.16 -n semseg-oacnns-TopoFusion-C-physical_weight0.16
#
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-C-physical_weight0.20 -n semseg-oacnns-TopoFusion-C-physical_weight0.20
#
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-C-physical_weight0.24 -n semseg-oacnns-TopoFusion-C-physical_weight0.24
#
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-C-physical_weight0.28 -n semseg-oacnns-TopoFusion-C-physical_weight0.28
#
#sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-C-physical_weight0.32 -n semseg-oacnns-TopoFusion-C-physical_weight0.32



sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-C-physical_weight0.36 -n semseg-oacnns-TopoFusion-C-physical_weight0.36

sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-C-physical_weight0.40 -n semseg-oacnns-TopoFusion-C-physical_weight0.40

sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-C-physical_weight0.44 -n semseg-oacnns-TopoFusion-C-physical_weight0.44

sh scripts/train.sh -g 1 -d scannet -c semseg-oacnns-TopoFusion-C-physical_weight0.48 -n semseg-oacnns-TopoFusion-C-physical_weight0.48

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
