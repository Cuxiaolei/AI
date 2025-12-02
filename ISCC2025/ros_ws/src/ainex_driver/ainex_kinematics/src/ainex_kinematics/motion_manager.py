#!/usr/bin/env python3
# encoding: utf-8
# Date:2023/07/10
# 串口舵机控制库
import os
import time
import sqlite3 as sql
from ainex_sdk import hiwonder_servo_controller

class MotionManager:
    runningAction = False
    stopRunning = False
    
    def __init__(self, action_path='/home/ubuntu/software/ainex_controller/ActionGroups', serial_port='/dev/ttyAMA0', baudrate=115200):
        self.servo_control = hiwonder_servo_controller.HiwonderServoController(serial_port, baudrate)
        self.action_path = action_path

    def set_servos_position(self, duration, *args):
        '''
        控制多个舵机转动
        :param duration: 时间ms
        :param args: 舵机id和位置, [[1, 500], [2, 500], ...]
        '''
        self.servo_control.set_servos_position(duration, args)

    def get_servos_position(self, *args):
        '''
        获取多个舵机位置
        :param args: 舵机id
        '''
        return self.servo_control.get_servos_position(args)

    def stop_action_group(self):
        self.stopRunning = True

    def run_action(self, actNum):
        '''
        运行动作组，无法发送stop停止信号
        :param actNum: 动作组名字 ， 字符串类型
        :param times:  运行次数
        :return:
        '''
        if actNum is None and self.action_path is not None:
            return
        actNum = os.path.join(self.action_path, actNum + ".d6a")
        self.stopRunning = False
        if os.path.exists(actNum):
            if not self.runningAction:
                self.runningAction = True
                ag = sql.connect(actNum)
                cu = ag.cursor()
                cu.execute("select * from ActionGroup")
                while True:
                    act = cu.fetchone()
                    if self.stopRunning:
                        self.stopRunning = False                   
                        break
                    if act is not None:
                        data = []
                        for i in range(0, len(act) - 2, 1):
                            data.append([i + 1, act[2 + i]])
                        self.set_servos_position(act[1], data) 
                        time.sleep(float(act[1])/1000.0)
                    else:   # 运行完才退出
                        break
                self.runningAction = False
                
                cu.close()
                ag.close()
        else:
            self.runningAction = False
            print('can not find aciton file')

if __name__ == '__main__':
    motion_manager = MotionManager(action_path='/home/ubuntu/software/ainex_controller/ActionGroups')
    
    # 单个舵机运行
    motion_manager.set_servos_position(500, [[23, 300]])
    time.sleep(0.5) 
    
    # 多个舵机运行
    motion_manager.set_servos_position(500, [[23, 500], [24, 500]])
    time.sleep(0.5)
    
    # 执行动作组
    motion_manager.run_action('left_shot')
    motion_manager.run_action('right_shot')
