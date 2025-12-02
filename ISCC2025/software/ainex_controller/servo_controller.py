#!/usr/bin/env python3
# encoding: utf-8
import time
import threading
from bus_servo_control import *
import action_group_controller as controller

def getServoPulse(servo_id):
    return getBusServoPulse(servo_id)

def getServoID(servo_id):
    return getBusServoID(servo_id)

def getServoDeviation(servo_id):
    return getBusServoDeviation(servo_id)

def setServoPulse(servo_id, pulse, use_time):
    setBusServoPulse(servo_id, pulse, use_time)

def setServosPulse(args):
    setBusServosPulse(args)

def setServoDeviation(servo_id ,dev):
    setBusServoDeviation(servo_id, dev)
    
def saveServoDeviation(servo_id):
    saveBusServoDeviation(servo_id)

def unloadServo(servo_id):
    unloadBusServo(servo_id)

def runActionGroup(num):
    threading.Thread(target=controller.runAction, args=(num, )).start()    

def stopActionGroup():    
    controller.stop_action_group() 

def wait_for_finish():
    while True:
        if controller.action_finish():
            time.sleep(0.01)
        else:
            break

if __name__ == '__main__':
    unloadServo(23)
    unloadServo(24)
    pass
    # for i in range(14):
        # t = time.time()
        # getServoDeviation(i + 1)
        # print(time.time() - t)
