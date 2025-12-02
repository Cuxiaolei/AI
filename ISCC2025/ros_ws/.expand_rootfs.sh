#!/bin/bash

root_part_dev="/dev/mmcblk0p2"
root_disk_dev="/dev/mmcblk0"

echo $root_disk_dev
echo $root_part_dev

# 扩容根分区
sudo fdisk $root_disk_dev <<EOF
d
2
n
p
2


w
EOF

sudo resize2fs $root_part_dev

echo "文件系统已成功扩展。"

sleep 1

sudo systemctl disable expand_rootfs.service

# 重新启动
sudo reboot
