#!/usr/bin/python3
# coding=utf8
import rospy
from std_srvs.srv import SetBool
from sensor_msgs.msg import MagneticField, Imu

def imu_callback(msg):
    rospy.loginfo(msg)

def imu_mag_callback(msg):
    rospy.loginfo(msg)

rospy.init_node('imu_node')
rospy.sleep(0.2)

# 开启imu发布
rospy.ServiceProxy('/sensor/imu/enable', SetBool)(True)

# 订阅imu数据
rospy.Subscriber('/sensor/imu/imu_raw', Imu, imu_callback)
# rospy.Subscriber('/sensor/imu/imu_mag', MagneticField, imu_mag_callback)
try:
    rospy.spin()
except keyboardinterrupt:
    print("shutting down")
