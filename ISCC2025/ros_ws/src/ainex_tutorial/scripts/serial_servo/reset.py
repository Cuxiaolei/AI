#!/usr/bin/env python3
# encoding: utf-8
from ainex_sdk import hiwonder_servo_controller

servo_control = hiwonder_servo_controller.HiwonderServoController('/dev/ttyAMA0', 115200)
for i in range(1, 25):
    print(i)
    servo_control.set_servo_range(i, 0, 1000)        
