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
        运行动作组，输出数据库中的详细数据日志
        :param actNum: 动作组名字 ， 字符串类型
        :return:
        '''
        if actNum is None and self.action_path is not None:
            return
        actNum = os.path.join(self.action_path, actNum + ".d6a")
        self.stopRunning = False
        if os.path.exists(actNum):
            # print(f"===== 开始执行动作组: {actNum} =====")
            if not self.runningAction:
                self.runningAction = True
                ag = sql.connect(actNum)
                cu = ag.cursor()
                # 查询总帧数（用于日志显示进度）
                cu.execute("SELECT COUNT(*) FROM ActionGroup")
                total_frames = cu.fetchone()[0]
                print(f"动作组总帧数: {total_frames}\n")

                # 重新执行查询，获取所有帧数据
                cu.execute("select * from ActionGroup")
                frame_count = 0  # 帧计数器

                while True:
                    act = cu.fetchone()
                    if self.stopRunning:
                        self.stopRunning = False
                        print("\n===== 动作被强制停止 =====")
                        break
                    if act is not None:
                        frame_count += 1
                        # 打印当前帧的基础信息
                        # print(f"----- 帧序号: {frame_count}/{total_frames} -----")
                        # print(f"帧ID: {act[0]}")  # 数据库中的帧序号
                        # print(f"执行时间: {act[1]}ms")  # 该帧舵机转动时间

                        # 解析并打印所有舵机的ID和位置
                        data = []
                        servo_info = []
                        for i in range(0, len(act) - 2, 1):
                            servo_id = i + 1  # 舵机ID（从1开始）
                            servo_pos = act[2 + i]  # 舵机位置
                            data.append([servo_id, servo_pos])
                            servo_info.append(f"舵机{servo_id}: {servo_pos}")

                        # 打印当前帧的所有舵机信息（用逗号分隔）
                        # print("舵机位置: " + ", ".join(servo_info) + "\n")

                        # 执行当前帧动作
                        self.set_servos_position(act[1], data)
                        time.sleep(float(act[1]) / 1000.0)
                    else:  # 所有帧执行完毕
                        # print(f"===== 动作组 {actNum} 执行完成（共{frame_count}帧） =====")
                        break
                self.runningAction = False
                cu.close()
                ag.close()
        else:
            self.runningAction = False
            print(f'错误: 未找到动作组文件 {actNum}')

    def _calc_joint_pos(self, block_data, frame_idx):
        """
        第4、5帧中20关节与18关节联动调整，避免爪子抬高
        :param block_data: 物块数据（x/y/size/img_width/img_height）
        :param frame_idx: 帧序号（1~8）
        :return: 关节位置列表
        """
        # 1. 固定帧（1~3、6~8）保持原动作组
        if frame_idx == 1:
            return [
                [1, 500], [2, 500], [3, 820], [4, 180], [5, 770], [6, 230], [7, 200], [8, 800],
                [9, 500], [10, 500], [11, 500], [12, 500], [13, 835], [14, 165], [15, 804], [16, 170],
                [17, 500], [18, 500], [19, 70], [20, 930], [21, 400], [22, 500]
            ]

        elif frame_idx == 2:
            return [
                [1, 500], [2, 500], [3, 820], [4, 180], [5, 770], [6, 230], [7, 200], [8, 800],
                [9, 500], [10, 500], [11, 500], [12, 500], [13, 835], [14, 420], [15, 798], [16, 150],
                [17, 500], [18, 480], [19, 70], [20, 740], [21, 400], [22, 650]
            ]

        elif frame_idx == 3:
            return [
                [1, 500], [2, 500], [3, 820], [4, 180], [5, 770], [6, 230], [7, 200], [8, 800],
                [9, 500], [10, 500], [11, 500], [12, 500], [13, 835], [14, 520], [15, 791], [16, 197],
                [17, 500], [18, 140], [19, 70], [20, 710], [21, 400], [22, 820]
            ]

        elif frame_idx == 6:
            return [
                [1, 500], [2, 500], [3, 820], [4, 180], [5, 770], [6, 230], [7, 200], [8, 800],
                [9, 500], [10, 500], [11, 500], [12, 500], [13, 835], [14, 500], [15, 800], [16, 180],
                [17, 500], [18, 360], [19, 70], [20, 730], [21, 400], [22, 540]
            ]

        elif frame_idx == 7:
            return [
                [1, 500], [2, 500], [3, 820], [4, 180], [5, 770], [6, 230], [7, 200], [8, 800],
                [9, 500], [10, 500], [11, 500], [12, 500], [13, 835], [14, 500], [15, 790], [16, 180],
                [17, 500], [18, 467], [19, 70], [20, 730], [21, 400], [22, 540]
            ]

        elif frame_idx == 8:
            return [
                [1, 500], [2, 500], [3, 640], [4, 360], [5, 507], [6, 493], [7, 290], [8, 710],
                [9, 500], [10, 500], [11, 500], [12, 499], [13, 875], [14, 125], [15, 803], [16, 193],
                [17, 500], [18, 511], [19, 100], [20, 900], [21, 400], [22, 540]
            ]

        # 2. 第4帧：20关节与18关节联动调整
        elif frame_idx == 4:
            # 提取物块参数
            x_ratio = block_data["x"] / block_data["img_width"]  # 0（左）~1（右）

            # 固定关节（移除18关节，改为动态计算）
            fixed_joints = [
                [1, 500], [2, 500], [3, 820], [4, 180], [5, 770], [6, 230], [7, 200], [8, 800],
                [9, 500], [10, 500], [11, 500], [12, 500], [13, 835], [14, 515], [15, 782], [16, 100],
                [17, 500], [19, 70], [21, 400], [22, 800]
            ]

            # 动态关节联动计算：
            # 20关节（弯曲）：x_ratio越大（物块偏右），位置越小（向内弯曲越多）
            elbow2_20 = int(700 - x_ratio * 100)  # 左→700，右→600（调整范围±100）
            elbow2_20 = max(600, min(700, elbow2_20))

            # 18关节（旋转）：与20关节联动，20关节位置越小（弯曲越多），18关节位置也越小（旋转越多）
            # 计算逻辑：基于20关节的偏移量，同步调整18关节
            elbow1_18 = int(80 - (700 - elbow2_20) / 100 * 40)  # 20关节每减小100，18关节减小40
            elbow1_18 = max(40, min(80, elbow1_18))  # 限制范围：40（多旋转）~80（少旋转）

            dynamic_joints = [
                [18, elbow1_18],  # 旋转关节
                [20, elbow2_20]  # 弯曲关节
            ]

            return fixed_joints + dynamic_joints

        # 3. 第5帧：与第4帧保持联动逻辑一致
        elif frame_idx == 5:
            # 提取物块参数
            x_ratio = block_data["x"] / block_data["img_width"]  # 0（左）~1（右）

            # 固定关节（移除18关节，改为动态计算）
            fixed_joints = [
                [1, 500], [2, 500], [3, 820], [4, 180], [5, 770], [6, 230], [7, 200], [8, 800],
                [9, 500], [10, 500], [11, 500], [12, 500], [13, 835], [14, 515], [15, 782], [16, 100],
                [17, 500], [19, 70], [21, 400], [22, 540]
            ]

            # 动态关节联动计算（同第4帧逻辑，保持动作连贯）
            elbow2_20 = int(700 - x_ratio * 100)
            elbow2_20 = max(600, min(700, elbow2_20))

            elbow1_18 = int(80 - (700 - elbow2_20) / 100 * 40)
            elbow1_18 = max(40, min(80, elbow1_18))

            dynamic_joints = [
                [18, elbow1_18],
                [20, elbow2_20]
            ]

            return fixed_joints + dynamic_joints

        return []  # 兜底返回空列表（避免异常）

    def run_dynamic_crawl_right(self, block_data):
        """
        动态生成右手夹取的8帧动作（无需读取.d6a文件）
        :param block_data: 物块位置数据（字典，含x/y/size/img_width/img_height）
        :return: True（执行成功）/False（执行失败）
        """
        # 验证传入数据的完整性
        required_keys = ["x", "y", "size", "img_width", "img_height"]
        if not all(key in block_data for key in required_keys):
            print("错误：传入的block_data缺少必要字段（需包含x/y/size/img_width/img_height）")
            return False

        # 8帧动作的执行时间（与原动作组一致，确保流畅）
        frame_durations = [1000, 1000, 1000, 1000, 800, 600, 500, 800]  # 单位：ms

        self.stopRunning = False
        if not self.runningAction:
            self.runningAction = True
            print("===== 开始动态生成右手夹取动作 =====")

            for frame_idx in range(1, 9):
                if self.stopRunning:
                    self.stopRunning = False
                    print("===== 动态动作被强制停止 =====")
                    self.runningAction = False
                    return False

                # 1. 计算当前帧的关节位置
                joint_positions = self._calc_joint_pos(block_data, frame_idx)
                print("八个动作组的具体数据")
                print(joint_positions)
                # 2. 获取当前帧的执行时间
                duration = frame_durations[frame_idx - 1]
                # 3. 执行当前帧动作（控制舵机）
                self.set_servos_position(duration, joint_positions)
                # 4. 等待动作完成
                time.sleep(float(duration) / 1000.0)
                # 5. 打印日志（可选，用于调试）
                print(f"动态帧 {frame_idx}/8 执行完成：duration={duration}ms，关节数={len(joint_positions)}")

            self.runningAction = False
            print("===== 动态右手夹取动作执行完成 =====")
            return True
        else:
            print("错误：已有动作在运行中，无法启动动态夹取")
            return False

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
