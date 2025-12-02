#!/usr/bin/python3
# coding=utf8
import time
from rpi_ws281x import PixelStrip
from rpi_ws281x import Color as PixelColor

__RGB_COUNT = 2
__RGB_PIN = 12#adapter_board_rgb
__RGB_FREQ_HZ = 800000
__RGB_DMA = 10
__RGB_BRIGHTNESS = 120
__RGB_CHANNEL = 0
__RGB_INVERT = False

RGB = PixelStrip(__RGB_COUNT, __RGB_PIN, __RGB_FREQ_HZ, __RGB_DMA, __RGB_INVERT, __RGB_BRIGHTNESS, __RGB_CHANNEL)
RGB.begin()
RGB.setPixelColor(0, PixelColor(0, 0, 0))
RGB.show()

def set_color(r, g, b):
    RGB.setPixelColor(0, PixelColor(r, g, b))
    RGB.show()

if __name__ == '__main__':
    while True:
        try:
            set_color(255, 0, 0)
            time.sleep(0.3)
            set_color(0, 255, 0)
            time.sleep(0.3)
            set_color(0, 0, 255)
            time.sleep(0.3)
        except KeyboardInterrupt:
            set_color(0, 0, 0)
            break
