#!/usr/bin/env python3
# encoding: utf-8
# Date:2023/07/20
import time, os
import pigpio
import RPi.GPIO as GPIO
from ainex_kinematics.motion_manager import MotionManager

os.system('sudo pigpiod')
time.sleep(1)

# 调用上位机生成的动作, 参数为动作组存储的路径
motion_manager = MotionManager('/home/ubuntu/software/ainex_controller/ActionGroups')

mode = GPIO.getmode()
if mode == 1 or mode is None:  # 是否已经设置引脚编码
    GPIO.setmode(GPIO.BCM)  # 设为BCM编码
GPIO.setwarnings(False)

touch_sensor_pin = 20

pi = pigpio.pi()
pi.set_mode(touch_sensor_pin, pigpio.INPUT)
pi.set_pull_up_down(touch_sensor_pin, pigpio.PUD_UP)

while True:
    if pi.read(touch_sensor_pin) == 0:
        time.sleep(0.05)
        if pi.read(touch_sensor_pin) == 1:
            motion_manager.run_action('twist')

