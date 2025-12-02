#!/usr/bin/python3
# coding=utf8
from ainex_sdk.gpio import *

GPIO.setup(adapter_board_led, GPIO.OUT)  # 设置引脚为输出模式(set pin as output mode)

def on():
    GPIO.output(adapter_board_led, 0)

def off():
    GPIO.output(adapter_board_led, 1)

def set(new_state):
    GPIO.output(adapter_board_led, new_state)

if __name__ == '__main__':
    import time
    while True:
        on()
        time.sleep(0.3)
        off()
        time.sleep(0.3)
