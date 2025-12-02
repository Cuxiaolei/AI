#!/usr/bin/python3
# coding=utf8
from ainex_sdk.gpio import *

GPIO.setup(adapter_board_buzzer, GPIO.OUT)

def on():
    GPIO.output(adapter_board_buzzer, 1)

def off():
    GPIO.output(adapter_board_buzzer, 0)

def set(new_state):
    GPIO.output(adapter_board_buzzer, new_state)

if __name__ == '__main__':
    import time
    on()
    time.sleep(0.5)
    off()
