#!/usr/bin/python3
# coding=utf8
import rospy
from ainex_interfaces.msg import RGB
from ainex_interfaces.srv import SetRGB

rospy.init_node('rgb_node')
# 设置rgb颜色
msg = RGB()
msg.r = 255
rospy.ServiceProxy('/sensor/rgb/set_rgb_state', SetRGB)(msg)
rospy.sleep(1)
msg = RGB()
msg.g = 255
rospy.ServiceProxy('/sensor/rgb/set_rgb_state', SetRGB)(msg)
rospy.sleep(1)
msg = RGB()
msg.b = 255
rospy.ServiceProxy('/sensor//rgb/set_rgb_state', SetRGB)(msg)
rospy.sleep(1)
msg = RGB()
rospy.ServiceProxy('/sensor/rgb/set_rgb_state', SetRGB)(msg)
