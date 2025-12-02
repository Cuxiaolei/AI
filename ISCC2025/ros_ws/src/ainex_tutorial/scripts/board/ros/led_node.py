#!/usr/bin/python3
# coding=utf8
import rospy
from std_srvs.srv import SetBool

rospy.init_node('led_node')
# 打开led
rospy.ServiceProxy('/sensor/led/set_led_state', SetBool)(True)
# 延时0.5s关闭
rospy.sleep(0.5)
rospy.ServiceProxy('/sensor/led/set_led_state', SetBool)(False)
