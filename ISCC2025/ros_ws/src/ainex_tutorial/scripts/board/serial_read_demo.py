#!/usr/bin/env python3
# encoding: utf-8
import time
import serial
import platform
from ainex_sdk.gpio import *

if 'raspi' in platform.release():
    serialHandle = serial.Serial("/dev/ttyAMA0", 115200)  # 初始化串口， 波特率为115200
else:
    serialHandle = serial.Serial("/dev/ttyTHS1", 115200)  # 初始化串口， 波特率为11520O

rx_pin = adapter_board_rx_con #  19
tx_pin = adapter_board_tx_con #  26

def port_as_write():
    GPIO.output(tx_pin, 1)  # 拉高TX_CON
    GPIO.output(rx_pin, 0)  # 拉低RX_CON

def port_as_read():
    GPIO.output(rx_pin, 1)  # 拉高RX_CON
    GPIO.output(tx_pin, 0)  # 拉低TX_CON

def port_init():
    GPIO.setwarnings(False)
    mode = GPIO.getmode()
    if mode == 1 or mode is None:
        GPIO.setmode(GPIO.BCM)
    GPIO.setup(rx_pin, GPIO.OUT)  # 配置RX_CON 即 rx_pin 为输出
    GPIO.output(rx_pin, 0)
    GPIO.setup(tx_pin, GPIO.OUT)  # 配置TX_CON 即 tx_pin 为输出
    GPIO.output(tx_pin, 1)

port_init()
port_as_read()

data = []

print('serial read data at 100hz')

while True:
    print(serialHandle.read())
    time.sleep(0.01)
