#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2023/06/12
# 板载资源
import math
import rospy
from icm20948 import ICM20948
from std_msgs.msg import Int32
from std_srvs.srv import SetBool
from ainex_interfaces.srv import SetFloat, SetRGB
from sensor_msgs.msg import MagneticField, Imu
from ainex_sdk import button, buzzer, led, rgb, imu

class Sensor:
    gravity = 9.80665
    def __init__(self, name):
        self.name = name
        rospy.init_node(self.name)
        self.enable_button = True
        self.enable_imu = True
        
        rospy.Service('/sensor/button/enable', SetBool, self.set_button_enable)
        rospy.Service('/sensor/imu/enable', SetBool, self.set_imu_enable)
        rospy.Service('/sensor/rgb/set_rgb_state', SetRGB, self.set_rgb_state_srv)
        rospy.Service('/sensor/led/set_led_state', SetBool, self.set_led_state_srv)
        rospy.Service('/sensor/buzzer/set_buzzer_state', SetBool, self.set_buzzer_state_srv)
        rospy.Service('/sensor/buzzer/set_buzzer_frequency', SetFloat, self.set_buzzer_frequency)

        self.imu = ICM20948()
        self.IMU_FRAME = rospy.get_param('~imu_frame', 'imu_link')
        imu_raw = rospy.get_param('~imu_raw', '/sensor/imu/imu_raw')
        freq = rospy.get_param('~freq', 50)
    
        self.raw_pub = rospy.Publisher(imu_raw, Imu, queue_size=1)
        self.mag_pub = rospy.Publisher('/sensor/imu/imu_mag', MagneticField, queue_size=1)

        button_pub = rospy.Publisher('/sensor/button/get_button_state', Int32, queue_size=1)
        rospy.sleep(0.2)
        
        buzzer.on()
        led.off()
        rgb.set_color(0, 0, 0)
        rospy.sleep(0.1)
        buzzer.off()
        
        rate = rospy.Rate(freq)
        try:
            while not rospy.is_shutdown():
                if self.enable_imu:
                    self.get_imu_callback()
                if self.enable_button:
                    button_pub.publish(button.get_button_status())
                rate.sleep()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down")

    def set_button_enable(self, msg):
        self.enable_button = msg.data
        
        return [True, 'set_button_enable']

    def set_imu_enable(self, msg):
        self.enable_imu = msg.data

        return [True, 'set_imu_enable']

    def set_buzzer_frequency(self, msg):
        buzzer.on()
        rospy.sleep(1/msg.data)
        buzzer.off()

        return [True, 'set_buzzer_frequency']

    def set_buzzer_state_srv(self, msg):
        if msg.data: 
            buzzer.on()
        else:
            buzzer.off()

        return [True, 'set_buzzer_state']

    def set_rgb_state_srv(self, msg):
        rgb.set_color(msg.data.r, msg.data.g, msg.data.b)
        
        return [True, 'set_rgb']

    def set_led_state_srv(self, msg):
        if msg.data:
            led.on()
        else:
            led.off()

        return [True, 'set_led']

    def get_imu_callback(self):
        try:
            ax, ay, az, gx, gy, gz = self.imu.read_accelerometer_gyro_data()    
            mx, my, mz = self.imu.read_magnetometer_data()
            raw_msg = Imu()
            raw_msg.header.frame_id = self.IMU_FRAME
            raw_msg.header.stamp = rospy.Time.now()
                
            raw_msg.orientation.w = 0
            raw_msg.orientation.x = 0
            raw_msg.orientation.y = 0
            raw_msg.orientation.z = 0
                
            raw_msg.linear_acceleration.x = ax * self.gravity
            raw_msg.linear_acceleration.y = ay * self.gravity
            raw_msg.linear_acceleration.z = az * self.gravity

            raw_msg.angular_velocity.x = math.radians(gx)
            raw_msg.angular_velocity.y = math.radians(gy)
            raw_msg.angular_velocity.z = math.radians(gz)

            raw_msg.orientation_covariance = [1e6, 0, 0, 0, 1e6, 0, 0, 0, 1e-6]
            raw_msg.angular_velocity_covariance = [1e6, 0, 0, 0, 1e6, 0, 0, 0, 1e-6]
            raw_msg.linear_acceleration_covariance = [-1,0,0,0,0,0,0,0,0]
                
            mag_msg = MagneticField()
            mag_msg.header.stamp = raw_msg.header.stamp
            mag_msg.magnetic_field.x = mx
            mag_msg.magnetic_field.y = my
            mag_msg.magnetic_field.z = mz
            mag_msg.magnetic_field_covariance[0] = -1
            
            self.raw_pub.publish(raw_msg)
            self.mag_pub.publish(mag_msg)
        except:
            pass

if __name__ == '__main__':
    Sensor('sensor') 
