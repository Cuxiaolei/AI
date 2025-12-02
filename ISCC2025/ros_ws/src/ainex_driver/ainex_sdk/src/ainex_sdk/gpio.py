#!/usr/bin/env python3
# encoding: utf-8
# Date:2023/04/24
#gpio引脚对应, 采用BCM编码
import sys
import platform

if 'raspi' in platform.release():
    import RPi.GPIO as GPIO
else:
    sys.path.append('/opt/nvidia/jetson-gpio/lib/python/Jetson/GPIO')
    import Jetson.GPIO as GPIO

mode = GPIO.getmode()
if mode == 1 or mode is None:  # 是否已经设置引脚编码
    GPIO.setmode(GPIO.BCM)  # 设为BCM编码
GPIO.setwarnings(False)

# 扩展板rx，tx
adapter_board_rx_con = 19
adapter_board_tx_con = 26

adapter_board_buzzer = 23  # 蜂鸣器
adapter_board_key = 13  # 按键
adapter_board_led = 16  # 蓝色led
adapter_board_rgb = 12  # rgb灯

