#!/usr/bin/env python3
# encoding: utf-8
import os
import time
import sqlite3 as sql
from bus_servo_control import *

runningAction = False
stopRunning = False

action_path = os.path.split(os.path.realpath(__file__))[0]

def stop_servo():
    for i in range(22):
        stopBusServo(i+1) 

def stop_action_group():
    global stopRunning, runningAction
    
    runningAction = False
    stopRunning = True
    time.sleep(0.1)

def action_finish():
    return runningAction

def runAction(actNum):
    '''
    运行动作组，无法发送stop停止信号
    :param actNum: 动作组名字 ， 字符串类型
    :param times:  运行次数
    :return:
    '''
    global runningAction
    global stopRunning

    if actNum is None:
        return
    actNum = os.path.join(action_path, 'ActionGroups', actNum + ".d6a")
    stopRunning = False
    if os.path.exists(actNum) is True:
        if runningAction is False:
            runningAction = True
            ag = sql.connect(actNum)
            cu = ag.cursor()
            cu.execute("select * from ActionGroup")
            while True:
                act = cu.fetchone()
                if stopRunning is True:
                    stopRunning = False                   
                    break
                if act is not None:
                    for i in range(0, len(act)-2, 1):
                        setBusServoPulse(i+1, act[2 + i], act[1])
                    time.sleep(float(act[1])/1000.0)
                else:   # 运行完才退出
                    break
            runningAction = False
            cu.close()
            ag.close()
    else:
        runningAction = False
        print("未能找到动作组文件")

if __name__ == "__main__":
    runAction('hurdling')
