#!/usr/bin/python3
# coding=utf8
import os
import time
import pigpio
from ainex_sdk.gpio import *

os.system('pigpiod')
time.sleep(1)
pi = pigpio.pi()
pi.set_mode(adapter_board_key, pigpio.INPUT)
pi.set_pull_up_down(adapter_board_key, pigpio.PUD_UP)
key_dict = {"key": adapter_board_key}

def get_button_status():
    return pi.read(adapter_board_key)

if __name__ == '__main__':
    while True:
        print(get_button_status())
        time.sleep(0.01)
