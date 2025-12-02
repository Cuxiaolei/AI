#!/usr/bin/env python3
# encoding: utf-8
# Date:2023/07/20
import rospy
import time, os
import pigpio
import signal
import threading
import RPi.GPIO as GPIO
from ainex_kinematics.gait_manager import GaitManager

rospy.init_node('gait_control_demo')
# 步态控制库
gait_manager = GaitManager()
rospy.sleep(0.2)

os.system('sudo pigpiod')
time.sleep(1)

mode = GPIO.getmode()
if mode == 1 or mode is None:  # 是否已经设置引脚编码
    GPIO.setmode(GPIO.BCM)  # 设为BCM编码
GPIO.setwarnings(False)

left_sensor_pin = 20
right_sensor_pin = 5

pi = pigpio.pi()
pi.set_mode(left_sensor_pin, pigpio.INPUT)
pi.set_mode(right_sensor_pin, pigpio.INPUT)
pi.set_pull_up_down(left_sensor_pin, pigpio.PUD_UP)
pi.set_pull_up_down(right_sensor_pin, pigpio.PUD_UP)

turn = False
count = 0
miss_count = 0
is_running = True
def shutdown(signum, frame):
    global is_running
    is_running = False

signal.signal(signal.SIGINT, shutdown)

def move():
    while is_running:
        if turn:
            gait_manager.move(2, 0, 0, 10, step_num=6) # 右转
        else:
            gait_manager.move(2, 0.01, 0, 0) # 直走
        time.sleep(0.01)

threading.Thread(target=move, daemon=True).start()

while is_running:
    if pi.read(left_sensor_pin) == 0 or pi.read(right_sensor_pin) == 0:
        count += 1
        miss_count = 0
    else:
        miss_count += 1
        count = 0
    if count > 5:
        count = 0
        turn = True
    if miss_count > 5:
        miss_count = 0
        turn = False

    time.sleep(0.01)
gait_manager.stop()
