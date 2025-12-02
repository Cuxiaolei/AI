#!/usr/bin/env python3
# encoding: utf-8
# Date:2023/07/10
import rospy
from ainex_kinematics.gait_manager import GaitManager

rospy.init_node('simple_gait_control_demo')
gait_manager = GaitManager()
rospy.sleep(0.2)

# 控制机器人行走
gait_manager.move(1, 0.03, 0, 0)
rospy.sleep(3)  # 以速度1前进3秒然后停下
gait_manager.stop()

gait_manager.move(2, -0.01, 0, 0, arm_swap=0)  # 关闭手臂摆动
rospy.sleep(3)  # 后退3秒然后停下
gait_manager.stop()

gait_manager.move(2, 0, 0, 5)
rospy.sleep(3)  # 右转3秒

gait_manager.move(3, 0.01, 0, 5)
rospy.sleep(3)  # 前进同时右转3秒

gait_manager.move(2, 0, 0.01, 0, step_num=3)  # 控制行走步数

gait_manager.move(2, -0.01, 0, 0)
rospy.sleep(3)  # 后退3秒然后停下
gait_manager.stop()
