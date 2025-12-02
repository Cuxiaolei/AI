#!/usr/bin/python3
# coding=utf8
import time
from ainex_sdk import imu

while True:
    (x, y, z), (ax, ay, az, gx, gy, gz) = imu.get_data()
    print("""Accel: {:05.2f} {:05.2f} {:05.2f}
Gyro:  {:05.2f} {:05.2f} {:05.2f}
Mag:   {:05.2f} {:05.2f} {:05.2f}""".format(
        ax, ay, az, gx, gy, gz, x, y, z
        ))
    time.sleep(0.1)
