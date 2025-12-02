#!/usr/bin/env python3
# encoding: utf-8
import threading
from bus_servo_control import *

def getServoPulse(servo_id):
    return getBusServoPulse(servo_id)

def getServoID(servo_id):
    return getBusServoID(servo_id)

def getServoDeviation(servo_id):
    return getBusServoDeviation(servo_id)

def setServoPulse(servo_id, pulse, use_time):
    setBusServoPulse(servo_id, pulse, use_time)

def setServoDeviation(servo_id ,dev):
    setBusServoDeviation(servo_id, dev)
    
def saveServoDeviation(servo_id):
    saveBusServoDeviation(servo_id)

def unloadServo(servo_id):
    unloadBusServo(servo_id)
