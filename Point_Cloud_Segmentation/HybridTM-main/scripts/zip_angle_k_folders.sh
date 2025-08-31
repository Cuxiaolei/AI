#!/bin/bash

# 压缩带角度和k参数的文件夹
echo "压缩 semseg-oacnns-TopoFusion-ABC_angle0.4..."
zip -r semseg-oacnns-TopoFusion-ABC_angle0.4.zip semseg-oacnns-TopoFusion-ABC_angle0.4 -x semseg-oacnns-TopoFusion-ABC_angle0.4/model/*

echo "压缩 semseg-oacnns-TopoFusion-ABC_angle0.6..."
zip -r semseg-oacnns-TopoFusion-ABC_angle0.6.zip semseg-oacnns-TopoFusion-ABC_angle0.6 -x semseg-oacnns-TopoFusion-ABC_angle0.6/model/*

echo "压缩 semseg-oacnns-TopoFusion-ABC_angle0.8..."
zip -r semseg-oacnns-TopoFusion-ABC_angle0.8.zip semseg-oacnns-TopoFusion-ABC_angle0.8 -x semseg-oacnns-TopoFusion-ABC_angle0.8/model/*

echo "压缩 semseg-oacnns-TopoFusion-ABC_angle1.0..."
zip -r semseg-oacnns-TopoFusion-ABC_angle1.0.zip semseg-oacnns-TopoFusion-ABC_angle1.0 -x semseg-oacnns-TopoFusion-ABC_angle1.0/model/*

echo "压缩 semseg-oacnns-TopoFusion-ABC_angle1.2..."
zip -r semseg-oacnns-TopoFusion-ABC_angle1.2.zip semseg-oacnns-TopoFusion-ABC_angle1.2 -x semseg-oacnns-TopoFusion-ABC_angle1.2/model/*

echo "压缩 semseg-oacnns-TopoFusion-ABC_angle1.4..."
zip -r semseg-oacnns-TopoFusion-ABC_angle1.4.zip semseg-oacnns-TopoFusion-ABC_angle1.4 -x semseg-oacnns-TopoFusion-ABC_angle1.4/model/*

echo "压缩 semseg-oacnns-TopoFusion-ABC_k4..."
zip -r semseg-oacnns-TopoFusion-ABC_k4.zip semseg-oacnns-TopoFusion-ABC_k4 -x semseg-oacnns-TopoFusion-ABC_k4/model/*

echo "压缩 semseg-oacnns-TopoFusion-ABC_k8..."
zip -r semseg-oacnns-TopoFusion-ABC_k8.zip semseg-oacnns-TopoFusion-ABC_k8 -x semseg-oacnns-TopoFusion-ABC_k8/model/*

echo "压缩 semseg-oacnns-TopoFusion-ABC_k24..."
zip -r semseg-oacnns-TopoFusion-ABC_k24.zip semseg-oacnns-TopoFusion-ABC_k24 -x semseg-oacnns-TopoFusion-ABC_k24/model/*

echo "压缩 semseg-oacnns-TopoFusion-ABC_k32..."
zip -r semseg-oacnns-TopoFusion-ABC_k32.zip semseg-oacnns-TopoFusion-ABC_k32 -x semseg-oacnns-TopoFusion-ABC_k32/model/*

echo "所有带角度和k参数的文件夹压缩完成！"
