#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/09/27
# @author:aiden

import rospy
import signal
import time
import math
from ainex_sdk import misc, common
from ainex_example.color_common import Common
from ainex_example.visual_patrol import VisualPatrol
from ainex_example.approach_object import ApproachObject
from ainex_interfaces.srv import SetString
from ainex_interfaces.msg import ObjectsInfo, ColorDetect, ROI
import sys
import os
from datetime import datetime
import re  # 用于过滤颜色控制符



# 自定义日志重定向类
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout

        # 1. 强制以UTF-8编码创建文件
        self.log = open(filename, "w", encoding="utf-8")
        # 2. 正则表达式：匹配终端颜色控制符（如\x1B[1;32m）
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, message):
        # 输出到控制台（保留颜色）
        self.terminal.write(message)
        # 过滤颜色控制符后写入文件（纯文本）
        cleaned_message = self.ansi_escape.sub('', message)
        self.log.write(cleaned_message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


class CompetitionNode(Common):


    # 按顺序检测三个roi，如果检测到黑线立刻跳出
    # y_min, y_max, x_min, x_max分别表示占图像的比例, 即实际大小为y_min*height
    line_roi = [(5 / 12, 6 / 12, 1 / 4, 3 / 4),
                (6 / 12, 7 / 12, 1 / 4, 3 / 4),
                (7 / 12, 8 / 12, 1 / 4, 3 / 4)
                ]

    stairs_roi = [1 / 5, 1, 0, 1]
    hurdles_roi = [1 / 5, 1, 0.15, 0.85]
    block_roi = [0, 350 / 480, 0, 1]
    intersection_roi = [100 / 480, 350 / 480, 0, 1]

    # 所需动作的名称
    hurdles_action_name = 'hurdles_fast_2'
    climb_stairs_action_name = 'climb_stairs_ours'
    descend_stairs_action_name = 'descend_stairs_our'
    crawl_left_action_name = 'crawl_left'  # 左行走动作名称
    crawl_right_action_name = 'crawl_right'  # 右行走动作名称
    place_block_action_name = 'place_block_our'  # 放置方块动作名称
    forward_step = 'forward_one_step_1'  # 前进一步
    back_step = 'move_back_our'  # 后退一步
    move_left = 'move_left_our'  # 左移
    move_right = 'move_right_1'  # 右移
    # self.motion_manager.run_action(self.back_step)
    # 图像处理时缩放到这个分辨率， 不建议修改
    image_process_size = [160, 120]

    # 跨栏状态下的目标位置阈值
    enter_hurdles_y = 280 / 480
    # hurdles_x_stop = 0.5
    hurdles_y_stop = 275

    # 上台阶状态下的目标位置阈值
    enter_climb_stairs_y = 185 / 480  # 当检测到的标识像素坐标y值占图像的比例大于此值时进入此阶段
    climb_stairs_x_stop = 0.5  # 当检测到的标识像素坐标x值占图像的比例在此值附近(范围可在ApproachObject里设置)时停止前后移动
    climb_stairs_y_stop = 210 / 480  # 当检测到的标识像素坐标y值占图像的比例在此值附近(范围可在ApproachObject里设置)时停止横向移动

    # 下台阶状态下的目标位置阈值
    enter_descend_stairs_y = 200 / 480  # 150
    descend_stairs_x_stop = 0.5
    descend_stairs_y_stop = 265 / 480


    # 向左抓取状态下的目标位置阈值
    enter_crawl_left_y = 180 / 480  # 当检测到的标识像素坐标y值占图像的比例大于此值时进入此阶段
    crawl_left_x_stop = 0.4  # 左行走的x轴目标位置范围可在ApproachObject里设置)时停止前后移动
    crawl_left_y_stop = 0.45  # 左行走的y轴目标位置(范围可在ApproachObject里设置)时停止横向移动

    # 向右抓取状态下的目标位置阈值，原理同上
    enter_crawl_right_y = 180 / 480  # 120;500
    crawl_right_x_stop = 0.6
    crawl_right_y_stop = 0.45

    # 放块状态下的目标位置阈值
    enter_place_block_y = 100 / 480
    place_block_x_stop = 320 / 640
    place_block_y_stop = 240 / 480


    def __init__(self, name):
        script_path = os.path.abspath(__file__)
        # 2. 提取代码文件所在的目录（比如 /home/ubuntu/ros_ws/src/ainex_example/scripts）
        script_dir = os.path.dirname(script_path)
        # 3. 在代码目录下创建日志文件夹（绝对路径，100%确定位置）
        log_dir = os.path.join(script_dir, "competition_logs")

        # 4. 强制创建文件夹，同时处理Linux下的权限/路径错误
        try:
            # exist_ok=True：即使文件夹已存在，也不会报错（避免重复创建的异常）
            # mode=0o755：设置Linux文件夹权限为“所有者可读写执行，其他用户可读可执行”，避免权限不足
            os.makedirs(log_dir, exist_ok=True, mode=0o755)
            # 验证是否真的创建成功（防止极端情况，比如路径被占用）
            if os.path.exists(log_dir):
                print(f"✅ 日志文件夹已创建在：{log_dir}")  # 终端会打印具体路径，方便你去查看
            else:
                raise Exception("文件夹创建后未检测到，可能路径被占用")
        except Exception as e:
            # 若创建失败（比如权限不足），直接降级到Linux的“用户主目录”（/home/ubuntu）创建，确保不报错
            log_dir = os.path.join(os.path.expanduser("~"), "competition_logs")
            os.makedirs(log_dir, exist_ok=True, mode=0o755)
            print(f"⚠️  原路径创建失败（{e}），已在用户主目录创建：{log_dir}")

        # 生成带时间戳的日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(log_dir, f"competition_log_{timestamp}.txt")

        # 重定向标准输出
        self.logger = Logger(log_filename)
        sys.stdout = self.logger
        sys.stderr = self.logger  # 同时捕获错误输出


        rospy.init_node(name)
        common.loginfo(f"初始化CompetitionNode节点: {name}")
        self.calib_config = common.get_yaml_data('/home/ubuntu/ros_ws/src/ainex_example/config/calib.yaml')
        self.name = name
        self.count = 0
        self.running = True
        self.slow = True
        self.objects_info = []
        self.delay_time = 0
        self.current_state = "visual_patrol"  # 当前状态
        self.next_state = "hurdles"  # 下一状态
        # self.next_state = "climb_stairs"  # 下一状态
        # self.next_state = "descend_stairs"  # 下一状态
        # self.next_state = "crawl_left"  # 下一状态
        # self.next_state = "crawl_right"  # 下一状态
        # self.next_state = "place_block"  # 下一状态

        self.state = {'visual_patrol': [[500, 260],
                                        ['black', self.line_roi, self.image_process_size, self.set_visual_patrol_color],
                                        False],  # 巡线
                      'climb_stairs': [[500, 260],
                                       ['red', self.stairs_roi, self.image_process_size, self.set_stairs_color], False],
                      # 上台阶
                      'descend_stairs': [[500, 260],
                                         ['red', self.stairs_roi, self.image_process_size, self.set_stairs_color],
                                         False],  # 下台阶
                      'hurdles': [[500, 260],
                                  ['blue', self.hurdles_roi, self.image_process_size, self.set_hurdles_color], False],
                      # 跨栏
                      'crawl_left': [[500, 260],
                                     ['green', self.block_roi, self.image_process_size, self.set_block_color], False],
                      # 左抓取状态
                      'crawl_right': [[500, 260],
                                      ['green', self.block_roi, self.image_process_size, self.set_block_color], False],
                      # 右抓取
                      'place_block': [[500, 260], ['black', self.intersection_roi, self.image_process_size,
                                                   self.set_intersection_color], False]}  # 放置

        self.head_pan_init = self.state[self.current_state][0][0]  # 左右舵机的初始值
        self.head_tilt_init = self.state[self.current_state][0][1]  # 上下舵机的初始值
        super().__init__(name, self.head_pan_init, self.head_tilt_init)
        common.loginfo(f"初始化父类Common，头部初始位置: 水平={self.head_pan_init}, 垂直={self.head_tilt_init}")

        self.approach_object = ApproachObject(self.gait_manager)
        self.visual_patrol = VisualPatrol(self.gait_manager)

        # 减小巡线步幅，提高其他标志检测稳定性
        self.visual_patrol.update_go_gait(x_max=0.028)
        self.visual_patrol.update_turn_gait(x_max=0.022)
        common.loginfo("初始化ApproachObject和VisualPatrol模块")

        signal.signal(signal.SIGINT, self.shutdown)

        # 订阅颜色识别结果
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)
        rospy.Service('~set_color', SetString, self.set_color_srv_callback)  # 设置颜色
        common.loginfo("订阅颜色识别结果话题并创建颜色设置服务")

        self.motion_manager.run_action('walk_ready_2')
        common.loginfo(f"调用动作组: walk_ready_2")

        if rospy.get_param('~start', True):
            # 通知颜色识别准备，此时只显示摄像头原画
            self.enter_func(None)
            self.start_srv_callback(None)
            common.loginfo('开始执行组合任务流程')







    def shutdown(self, signum, frame):
        self.running = False
        common.loginfo(f'{self.name} 收到关闭信号，正在停止...')

        # 恢复标准输出
        sys.stdout = self.logger.terminal
        sys.stderr = self.logger.terminal
        self.logger.close()

        common.loginfo('日志已保存到文件')

    def set_visual_patrol_color(self, color, roi, image_process_size):
        common.loginfo(f"设置巡线颜色参数: 颜色={color}, ROI={roi}")
        # 设置巡线颜色
        line_param = ColorDetect()
        line_param.color_name = color
        line_param.use_name = True
        line_param.detect_type = 'line'
        line_param.image_process_size = image_process_size
        line_param.line_roi.up.y_min = int(roi[0][0] * image_process_size[1])
        line_param.line_roi.up.y_max = int(roi[0][1] * image_process_size[1])
        line_param.line_roi.up.x_min = int(roi[0][2] * image_process_size[0])
        line_param.line_roi.up.x_max = int(roi[0][3] * image_process_size[0])

        line_param.line_roi.center.y_min = int(roi[1][0] * image_process_size[1])
        line_param.line_roi.center.y_max = int(roi[1][1] * image_process_size[1])
        line_param.line_roi.center.x_min = int(roi[1][2] * image_process_size[0])
        line_param.line_roi.center.x_max = int(roi[1][3] * image_process_size[0])

        line_param.line_roi.down.y_min = int(roi[2][0] * image_process_size[1])
        line_param.line_roi.down.y_max = int(roi[2][1] * image_process_size[1])
        line_param.line_roi.down.x_min = int(roi[2][2] * image_process_size[0])
        line_param.line_roi.down.x_max = int(roi[2][3] * image_process_size[0])

        line_param.min_area = 1
        line_param.max_area = image_process_size[0] * image_process_size[1]

        return line_param

    # 生成方块颜色识别参数
    def set_block_color(self, color, roi, image_process_size):
        common.loginfo(f"设置方块颜色参数: 颜色={color}, ROI={roi}")
        block_param = ColorDetect()
        block_param.color_name = color
        block_param.detect_type = 'circle'
        block_param.use_name = True
        block_param.image_process_size = self.image_process_size
        block_param.roi.y_min = int(roi[0] * image_process_size[1])
        block_param.roi.y_max = int(roi[1] * image_process_size[1])
        block_param.roi.x_min = int(roi[2] * image_process_size[0])
        block_param.roi.x_max = int(roi[3] * image_process_size[0])
        block_param.min_area = 10
        block_param.max_area = image_process_size[0] * image_process_size[1]

        return block_param

    # 生成交叉点颜色识别参数
    def set_intersection_color(self, color, roi, image_process_size):
        common.loginfo(f"设置交叉点颜色参数: 颜色={color}, ROI={roi}")
        intersection_param = ColorDetect()
        intersection_param.color_name = color
        intersection_param.detect_type = 'intersection'
        intersection_param.use_name = True
        intersection_param.image_process_size = image_process_size
        intersection_param.roi.y_min = int(roi[0] * image_process_size[1])
        intersection_param.roi.y_max = int(roi[1] * image_process_size[1])
        intersection_param.roi.x_min = int(roi[2] * image_process_size[0])
        intersection_param.roi.x_max = int(roi[3] * image_process_size[0])
        intersection_param.min_area = 10
        intersection_param.max_area = image_process_size[0] * image_process_size[1]

        return intersection_param

    def set_stairs_color(self, color, roi, image_process_size):
        common.loginfo(f"设置台阶颜色参数: 颜色={color}, ROI={roi}")
        # 设置台阶标志颜色
        stairs_param = ColorDetect()
        stairs_param.color_name = color
        stairs_param.detect_type = 'side'
        stairs_param.use_name = True
        stairs_param.image_process_size = image_process_size
        stairs_param.roi.y_min = int(roi[0] * image_process_size[1])
        stairs_param.roi.y_max = int(roi[1] * image_process_size[1])
        stairs_param.roi.x_min = int(roi[2] * image_process_size[0])
        stairs_param.roi.x_max = int(roi[3] * image_process_size[0])
        stairs_param.min_area = 10 * 20
        stairs_param.max_area = image_process_size[0] * image_process_size[1]

        return stairs_param

    def set_hurdles_color(self, color, roi, image_process_size):
        common.loginfo(f"设置跨栏颜色参数: 颜色={color}, ROI={roi}")
        # 设置台阶标志颜色
        hurdles_param = ColorDetect()
        hurdles_param.color_name = color
        hurdles_param.detect_type = 'side'
        hurdles_param.use_name = True
        hurdles_param.image_process_size = image_process_size
        hurdles_param.roi.y_min = int(roi[0] * image_process_size[1])
        hurdles_param.roi.y_max = int(roi[1] * image_process_size[1])
        hurdles_param.roi.x_min = int(roi[2] * image_process_size[0])
        hurdles_param.roi.x_max = int(roi[3] * image_process_size[0])
        hurdles_param.min_area = 10 * 20
        hurdles_param.max_area = image_process_size[0] * image_process_size[1]

        return hurdles_param

    def set_color_srv_callback(self, msg):
        common.loginfo("收到颜色设置服务请求")
        # 设置颜色
        block_param = self.set_block_color(self.state['crawl_left'][1][0], self.state['crawl_left'][1][1],
                                           self.state['crawl_left'][1][2])
        stairs_param = self.set_stairs_color(self.state['climb_stairs'][1][0], self.state['climb_stairs'][1][1],
                                             self.state['climb_stairs'][1][2])
        line_param = self.set_visual_patrol_color(self.state['visual_patrol'][1][0], self.state['visual_patrol'][1][1],
                                                  self.state['visual_patrol'][1][2])
        hurdles_param = self.set_hurdles_color(self.state['hurdles'][1][0], self.state['hurdles'][1][1],
                                               self.state['hurdles'][1][2])
        intersection_param = self.set_intersection_color(self.state['place_block'][1][0],
                                                         self.state['place_block'][1][1],
                                                         self.state['place_block'][1][2])

        self.detect_pub.publish([line_param, stairs_param, hurdles_param, intersection_param, block_param])
        common.loginfo(f'{self.name} 完成颜色参数设置')

        return [True, 'set_color']

    def get_color_callback(self, msg):
        # 获取颜色识别结果
        self.objects_info = msg.data
        # 仅在调试时输出，避免日志过多
        # common.loginfo(f"收到颜色识别结果，包含{len(msg.data)}个目标")

    def state_init(self, current_state, next_state):
        # 不同阶段的初始化
        if self.state[current_state][2] == False:
            common.loginfo(f"初始化状态: {current_state}, 下一状态: {next_state}")
            self.state[current_state][2] = True
            self.init_action(self.state[current_state][0][0], self.state[current_state][0][1])  # 头部姿态
            param1 = self.state[current_state][1][3](self.state[current_state][1][0], self.state[current_state][1][1],
                                                     self.state[current_state][1][2])
            param2 = self.state[next_state][1][3](self.state[next_state][1][0], self.state[next_state][1][1],
                                                  self.state[next_state][1][2])
            self.detect_pub.publish([param1, param2])  # 颜色检测设置
            common.loginfo(f"{current_state} 初始化完成")

    # 进入跨栏
    def enter_hurdles(self, hurdles_data):
        common.loginfo("进入enter_hurdles方法")
        if hurdles_data is not None:
            print("state:enter_hurdles ", "hurdles_data.y:", hurdles_data.y)
            if hurdles_data.y > self.hurdles_roi[0] * hurdles_data.height and self.slow:
                self.visual_patrol.update_go_gait(x_max=0.015)
                self.visual_patrol.update_turn_gait(x_max=0.01)
                self.slow = False
                common.loginfo("调整步态参数为跨栏模式")
            if hurdles_data.y > self.enter_hurdles_y * hurdles_data.height:
                self.count += 1
                common.loginfo(f"满足跨栏条件，累计计数: {self.count}")
                if self.count > 0:
                    self.count = 0
                    self.gait_manager.disable()
                    self.approach_object.update_approach_stop_value(y_approach_value=15, x_approach_value=15,
                                                                    yaw_approach_value=5)
                    # common.loginfo(f"调用动作组: hand_back")
                    self.motion_manager.run_action('hand_back_4')  # 手往后，防止遮挡
                    return True
            else:
                self.count = 0
                common.loginfo("未满足跨栏条件，重置计数")
        return False

    # 退出跨栏
    def exit_hurdles(self, hurdles_data):
        common.loginfo("进入exit_hurdles方法")
        # 跨栏处理
        if hurdles_data is not None:
            self.missed_hurdles_flag_count = 0


            offset_y = self.hurdles_y_stop - hurdles_data.y
            print("当前状态:exit_hurdles ", "hurdles_data.y:", hurdles_data.y,"self.hurdles_y_stop:", self.hurdles_y_stop,"offset_y:", offset_y)


            # 如果未到达目标高度，缓慢向前推进
            if offset_y < -15:
                common.loginfo(f"向前推进，调用动作组: {self.forward_step}")
                self.motion_manager.run_action(self.forward_step)
                time.sleep(0.15)  # 缩短间隔防止过冲

            # 无论是否校准成功，超过3帧后都执行后续跨栏动作
            self.gait_manager.disable()
            common.loginfo('准备执行跨栏动作')
            self.gait_manager.set_step([420, 0.22, 0.02], 0.01, 0, 0, None, 0, 1)
            common.loginfo(f"调用动作组: {self.hurdles_action_name}")
            self.motion_manager.run_action(self.hurdles_action_name)
            rospy.sleep(0.5)
            self.visual_patrol.update_go_gait(x_max=0.028)
            self.visual_patrol.update_turn_gait(x_max=0.022)
            self.slow = True
            common.loginfo(f"调用动作组: walk_ready")
            self.motion_manager.run_action('walk_ready')
            rospy.sleep(0.5)
            return True

        else:
            common.loginfo(f"未检测到栏杆，向后微调，调用动作组: {self.back_step}")
            self.motion_manager.run_action(self.back_step)
            time.sleep(0.01)



    # 进入上台阶
    def enter_climb_stairs(self, stairs_data):
        common.loginfo("进入enter_climb_stairs方法")
        if stairs_data is not None:
            print("state:enter_climb_stairs ", "stairs_data.y:", stairs_data.y)
            if stairs_data.y > self.stairs_roi[0] * stairs_data.height and self.slow:
                self.visual_patrol.update_go_gait(x_max=0.02)
                self.visual_patrol.update_turn_gait(x_max=0.018)
                self.slow = False
                common.loginfo("调整步态参数为慢速模式")
            if stairs_data.y > self.enter_climb_stairs_y * stairs_data.height:
                self.count += 1
                common.loginfo(f"满足上台阶条件，累计计数: {self.count}")
                if self.count > 1:  # 主线程比较快，颜色检测回调慢一点，需要连续检测来排除滞后干扰
                    self.count = 0
                    self.gait_manager.disable()
                    self.approach_object.update_approach_stop_value(20, 15, 4)  # 设置靠近目标停止的条件，分别为y, x, angle误差
                    # common.loginfo(f"调用动作组: hand_back")
                    self.motion_manager.run_action('hand_back_4')  # 手往后，防止遮挡
                    rospy.sleep(0.5)
                    return True
            else:
                self.count = 0
                common.loginfo("未满足上台阶条件，重置计数")
        return False





    # 退出上台阶
    def exit_climb_stairs(self, stairs_data):
        common.loginfo("进入exit_climb_stairs方法")
        # 上阶梯处理
        if stairs_data is not None:
            print("state:exit_climb_stairs ", "stairs_data.y:", stairs_data.y)

            # 超过3帧或校准成功后执行上台阶动作
            self.gait_manager.disable()  # 关闭步态控制
            common.loginfo('准备执行上台阶动作')
            # self.motion_manager.run_action(self.move_left)
            self.gait_manager.set_step([320, 0.25, 0.02], 0.015, 0, 0, None, 0, 2)
            common.loginfo(f"调用动作组: {self.climb_stairs_action_name}")
            self.motion_manager.run_action(self.climb_stairs_action_name)  # 执行上台阶动作
            rospy.sleep(0.5)
            common.loginfo(f"调用动作组: walk_ready")
            self.motion_manager.run_action('walk_ready')
            rospy.sleep(0.5)
            return True

    # 进入下台阶
    def enter_descend_stairs(self, stairs_data):
        common.loginfo("进入enter_descend_stairs方法")
        if stairs_data is not None:
            print("state:enter_descend_stairs ", "stairs_data.y:", stairs_data.y)
            if stairs_data.y > self.enter_descend_stairs_y * stairs_data.height:
                self.visual_patrol.update_go_gait(x_max=0.012)
                self.visual_patrol.update_turn_gait(x_max=0.01)
                self.count += 1
                common.loginfo(f"满足下台阶条件，累计计数: {self.count}")
                if self.count > 1:
                    self.count = 0
                    self.gait_manager.disable()
                    self.approach_object.update_approach_stop_value(15, 10, 3)
                    # common.loginfo(f"调用动作组: hand_back")
                    self.motion_manager.run_action('hand_back_4')  # 手往后，防止遮挡
                    rospy.sleep(0.5)
                    return True
            else:
                self.count = 0
                common.loginfo("未满足下台阶条件，重置计数")
        return False


    # 退出下台阶
    def exit_descend_stairs(self, stairs_data):
        common.loginfo("进入exit_descend_stairs方法")
        # 下阶梯处理
        if stairs_data is not None:
            self.missed_red_flag_count = 0
            print("state:exit_descend_stairs ", "stairs_data.y:", stairs_data.y)

            # 超过3帧或校准成功后执行下台阶动作
            self.gait_manager.disable()  # 关闭步态控制
            common.loginfo('准备执行上台阶动作')
            # self.motion_manager.run_action(self.move_right)
            self.gait_manager.set_step([320, 0.25, 0.02], 0.015, 0, 0, None, 0, 2)
            common.loginfo(f"调用动作组: {self.descend_stairs_action_name}")
            self.motion_manager.run_action(self.descend_stairs_action_name)  # 执行下台阶动作
            rospy.sleep(0.5)
            self.visual_patrol.update_go_gait(x_max=0.028)
            self.visual_patrol.update_turn_gait(x_max=0.022)
            self.slow = True
            common.loginfo(f"调用动作组: walk_ready")
            self.motion_manager.run_action('walk_ready')
            rospy.sleep(0.5)
            return True


    # 进入左抓取判断函数
    def enter_crawl_left(self, block_data):
        common.loginfo("进入enter_crawl_left方法")
        if block_data is not None:
            print("state:enter_crawl_left", " block_data.x:", block_data.x, " block_data.y:", block_data.y)
            if block_data.y > self.block_roi[0] * block_data.height and self.slow:
                self.visual_patrol.update_go_gait(x_max=0.02)
                self.visual_patrol.update_turn_gait(x_max=0.015)
                self.slow = False
                common.loginfo("调整步态参数为左抓取模式")

            if block_data.y > block_data.height * self.enter_crawl_left_y:
                self.count += 1
                common.loginfo(f"满足左抓取条件，累计计数: {self.count}")
                if self.count > 1:  # 连续检测到满足条件,切换到左行走状态
                    self.count = 0
                    self.gait_manager.disable()
                    return True
            else:
                self.count = 0
                common.loginfo("未满足左抓取条件，重置计数")
        else:
            self.count = 0
            common.loginfo("未检测到方块，重置左抓取计数")
        return False


    # 退出左抓取判断函数
    def exit_crawl_left(self, block_data):
        common.loginfo("进入exit_crawl_left方法")
        """
        退出左抓取状态逻辑：
        保证机器人在靠近目标时进行精确的左右调整和前进控制，使停靠位置更加精确。
        """
        if block_data is not None:
            common.loginfo(f"state:exit_crawl_left, block_data.x: {block_data.x}, block_data.y: {block_data.y}")

            centerx_left = self.crawl_left_x_stop * block_data.width
            centery_left = self.crawl_left_y_stop * block_data.height
            offset_x = block_data.x - centerx_left
            offset_y = block_data.y - centery_left
            common.loginfo(f"左抓取位置x偏差: {offset_x}，目标中心: {centerx_left}")
            common.loginfo(f"左抓取位置y偏差: {offset_y}，目标中心: {centery_left}")

            # 根据左右偏差进行微调
            if offset_x > 15:  # 阈值从 20 改为 15 提高精度
                common.loginfo(f"向右微调，调用动作组: {self.move_right}")
                self.motion_manager.run_action(self.move_right)
            elif offset_x < -15:
                common.loginfo(f"向左微调，调用动作组: {self.move_left}")
                self.motion_manager.run_action(self.move_left)

            # 如果未到达目标高度，缓慢向前推进
            elif offset_y < -15:
                common.loginfo(f"向前推进，调用动作组: {self.forward_step}")
                self.motion_manager.run_action(self.forward_step)
                time.sleep(0.15)  # 缩短间隔防止过冲

            elif offset_y > 15:
                common.loginfo(f"向后推进，调用动作组: {self.back_step}")
                self.motion_manager.run_action(self.back_step)
                time.sleep(0.15)  # 缩短间隔防止过冲

            # 到达目标位置，执行抓取动作
            else:
                time.sleep(0.3)  # 减少延迟，加快响应
                common.loginfo(f"到达左抓取位置，调用动作组: {self.crawl_left_action_name}")
                self.motion_manager.run_action(self.crawl_left_action_name)
                self.visual_patrol.update_go_gait(arm_swap=0)
                self.visual_patrol.update_turn_gait(arm_swap=0)
                self.visual_patrol.update_go_gait(x_max=0.028)
                self.visual_patrol.update_turn_gait(x_max=0.022)
                self.slow = True
                return True

        return False

    # 进入右抓取判断函数
    def enter_crawl_right(self, block_data):
        common.loginfo("进入enter_crawl_right方法")
        if block_data is not None:
            print("state:enter_crawl_right", " block_data.x:", block_data.x, " block_data.y:", block_data.y)
            if block_data.y > self.block_roi[0] * block_data.height and self.slow:
                self.visual_patrol.update_go_gait(x_max=0.015)
                self.visual_patrol.update_turn_gait(x_max=0.015)
                self.slow = False
                common.loginfo("调整步态参数为右抓取模式")
            if block_data.y > block_data.height * self.enter_crawl_right_y:
                self.count += 1
                common.loginfo(f"满足右抓取条件，累计计数: {self.count}")
                if self.count > 1:
                    self.count = 0
                    self.gait_manager.disable()
                    return True
            else:
                self.count = 0
                common.loginfo("未满足右抓取条件，重置计数")
        else:
            self.count = 0
            common.loginfo("未检测到方块，重置右抓取计数")
        return False



    # 退出右抓取判断函数
    def exit_crawl_right(self, block_data):
        common.loginfo("进入exit_crawl_right方法")
        """
        退出右抓取状态逻辑：
        保证机器人在靠近目标时进行精确的左右调整和前进控制，使停靠位置更加精确。
        """

        if block_data is not None:
            common.loginfo(f"state:exit_crawl_right, block_data.x: {block_data.x}, block_data.y: {block_data.y}")

            centerx_right = self.crawl_right_x_stop * block_data.width
            centery_right = self.crawl_right_y_stop * block_data.height
            offset_x = block_data.x - centerx_right
            offset_y = block_data.y - centery_right
            common.loginfo(f"右抓取位置偏差: {offset_x}，目标中心: {centerx_right}")

            # 根据左右偏差进行微调
            if offset_x > 15:  # 阈值由 20 降至 15 提高精度
                common.loginfo(f"向右微调，调用动作组: {self.move_right}")
                self.motion_manager.run_action(self.move_right)
            elif offset_x < -15:
                common.loginfo(f"向左微调，调用动作组: {self.move_left}")
                self.motion_manager.run_action(self.move_left)

            # 如果未到达目标高度，缓慢向前推进
            elif offset_y < -15:
                common.loginfo(f"向前推进，调用动作组: {self.forward_step}")
                self.motion_manager.run_action(self.forward_step)
                time.sleep(0.15)  # 缩短时间防止过冲

            elif offset_y > 15:
                common.loginfo(f"向后推进，调用动作组: {self.back_step}")
                self.motion_manager.run_action(self.back_step)
                time.sleep(0.15)  # 缩短时间防止过冲

            # 到达目标位置，执行抓取动作
            else:
                time.sleep(0.3)  # 减少延迟，加快响应
                common.loginfo(f"到达右抓取位置，调用动作组: {self.crawl_right_action_name}")
                self.motion_manager.run_action(self.crawl_right_action_name)
                self.visual_patrol.update_go_gait(arm_swap=30)
                self.visual_patrol.update_turn_gait(arm_swap=30)
                self.visual_patrol.update_go_gait(x_max=0.02)
                self.visual_patrol.update_turn_gait(x_max=0.018)
                self.slow = True
                self.delay_time = time.time() + 13.8  # 抓取完成延时再识别
                return True


        return False



    def enter_place_block(self, line_data):
        common.loginfo("进入enter_place_block方法")
        """
        判断是否进入放置点（放置块/横条）：
        - 只有当连续 positive_frames_threshold 帧满足目标条件时返回 True；
        - 不再维护或使用 no_line_count / force_exit_place；当 line_data 为 None 时仅重置正向计数并返回 False；
        - 保留横条判定逻辑（宽度 + 近似水平）。
        """

        # --- 初始化计数器（稳健） ---
        if not hasattr(self, 'count'):
            self.count = 0

        # 参数（可根据实际场景微调）
        positive_frames_threshold = 1  # 连续满足放置点条件才触发
        span_threshold = 120  # 横向跨度最小像素阈值（或用相对宽度）
        min_vertical_span = 5  # 垂直方向跨度下限（避免噪点）
        angle_threshold_deg = 15  # 认为“横向”允许的最大角度偏差（度）
        self.visual_patrol.update_go_gait(x_max=0.015)
        self.visual_patrol.update_turn_gait(x_max=0.012)
        common.loginfo("调整步态参数为放置方块模式")

        # --- 若本帧无检测到任何 line_data：仅重置正向计数并返回 False ---
        if line_data is None:
            self.count = 0
            common.loginfo("未检测到放置点标志，重置计数")
            return False

        # --- 有 line_data：安全提取字段（防止属性缺失崩溃） ---
        try:
            lx, ly = line_data.left_point[0], line_data.left_point[1]
            rx, ry = line_data.right_point[0], line_data.right_point[1]
            # 有些检测器会提供参考 y、width、height；若无则使用默认 None / 计算值
            y_val = getattr(line_data, 'y', max(ly, ry))
            width = getattr(line_data, 'width', None)
            height = getattr(line_data, 'height', None)
        except Exception as e:
            # 数据结构异常：打印调试信息，重置正向计数，不视为无线
            try:
                common.loginfo(f"enter_place_block: 无效的数据结构: {e}")
                common.loginfo(f"line_data: {line_data}")
            except Exception:
                pass
            self.count = 0
            return False

        # --- 计算几何特征：横向跨度、垂直跨度、与水平线的角度（度） ---
        horiz_span = abs(rx - lx)
        vertical_span = abs(ry - ly)
        dx = rx - lx
        dy = ry - ly

        # 若线段长度为0，给一个大角度值以避免误判
        if dx == 0 and dy == 0:
            angle_deg = 90.0
        else:
            angle_deg = abs(math.degrees(math.atan2(dy, dx)))
            if angle_deg > 90:
                angle_deg = 180 - angle_deg

        # 调试输出，便于定位哪个条件没满足
        try:
            common.loginfo(
                f"放置点检测: y_val={y_val}, 横向跨度={horiz_span}/{span_threshold}, 垂直跨度={vertical_span}/{min_vertical_span}, 角度={angle_deg:.1f}/{angle_threshold_deg}度")
        except Exception:
            pass

        # --- 判定条件：考虑横条特征（足够宽且近似水平） ---
        is_horizontal_enough = (horiz_span >= span_threshold) and (angle_deg <= angle_threshold_deg) and (vertical_span >= min_vertical_span) and (vertical_span <= 50)

        if is_horizontal_enough:
            # 满足目标条件：累加正向计数并在达到阈值后停止并返回 True
            self.count += 1
            try:
                common.loginfo(f"检测到疑似放置点，计数={self.count}/{positive_frames_threshold}")
            except Exception:
                pass

            if self.count >= positive_frames_threshold:
                # 达到连续帧阈值 -> 认为真实进入放置点
                self.count = 0
                try:
                    if hasattr(self, 'gait_manager') and self.gait_manager is not None:
                        self.gait_manager.stop()
                except Exception as e:
                    try:
                        common.loginfo(f"gait_manager.stop() 异常: {e}")
                    except Exception:
                        pass
                common.loginfo('确认检测到交叉点，准备进入放置状态')
                return True
            else:
                return False
        else:
            # 有数据但不满足“放置点”条件：只重置正向计数，继续等待下一帧
            self.count = 0
            try:
                reasons = []
                if horiz_span < span_threshold:
                    reasons.append(f"横向跨度太小 ({horiz_span}<{span_threshold})")
                if angle_deg > angle_threshold_deg:
                    reasons.append(f"角度太大 ({angle_deg:.1f}deg>{angle_threshold_deg}deg)")
                if vertical_span < min_vertical_span:
                    reasons.append(f"垂直跨度太小 ({vertical_span}<{min_vertical_span})")
                common.loginfo(f"未满足放置点条件: {'; '.join(reasons)}")
            except Exception:
                pass
            return False




    # 退出放块判断函数
    def exit_place_block(self, line_data):
        common.loginfo("进入exit_place_block方法")
        # 放置方块处理：先微调到位，再前进4步并放置
        if line_data is not None:
            self.missed_black_flag_count = 0
            common.loginfo(f"state:exit_place_block, line_data.x: {line_data.x}, line_data.y: {line_data.y}")

            # 1. 计算目标位置（基于放块阈值参数）
            target_x = self.place_block_x_stop * line_data.width  # x轴目标位置
            target_y = self.place_block_y_stop * line_data.height  # y轴目标位置
            offset_x = line_data.x - target_x  # x轴偏差（当前x - 目标x）
            offset_y = line_data.y - target_y  # y轴偏差（当前y - 目标y）
            common.loginfo(
                f"放块位置偏差: x={offset_x:.1f}, y={offset_y:.1f}，目标x={target_x:.1f}, 目标y={target_y:.1f}")

            # 2. 横向微调（左右调整，基于x偏差）
            if abs(offset_x) > 10:  # 偏差超过10像素时微调
                if offset_x < 0:  # 当前x > 目标x：需向左移
                    common.loginfo(f"向左微调（x偏差{offset_x:.1f}），调用动作组: {self.move_left}")
                    self.motion_manager.run_action(self.move_left)
                    rospy.sleep(0.15)  # 等待动作完成
                    return False  # 本轮只做微调，不继续后续步骤
                else:  # 当前x < 目标x：需向右移
                    common.loginfo(f"向右微调（x偏差{offset_x:.1f}），调用动作组: {self.move_right}")
                    self.motion_manager.run_action(self.move_right)
                    rospy.sleep(0.15)
                    return False

            # 3. 纵向微调（前后调整，基于y偏差）
            if abs(offset_y) > 10:  # 偏差超过10像素时微调
                if offset_y < 0:  # 当前y < 目标y：需向前移（靠近目标）
                    common.loginfo(f"向前微调（y偏差{offset_y:.1f}），调用动作组: {self.forward_step}")
                    self.motion_manager.run_action(self.forward_step)
                    rospy.sleep(0.15)
                    return False
                else:  # 当前y > 目标y：需向后移（远离目标）
                    common.loginfo(f"向后微调（y偏差{offset_y:.1f}），调用动作组: {self.back_step}")
                    self.motion_manager.run_action(self.back_step)
                    rospy.sleep(0.15)
                    return False

            # # 4. 角度微调（基于偏航角，可选）
            # if abs(line_data.angle - self.place_block_yaw_stop) > 3:  # 角度偏差>3度时调整
            #     common.loginfo(f"旋转微调（角度偏差{line_data.angle - self.place_block_yaw_stop:.1f}度）")
            #     # 此处可根据实际旋转动作名修改，例如：
            #     # self.motion_manager.run_action('turn_small')  # 假设存在小角度旋转动作
            #     rospy.sleep(0.15)
            #     return False

            # 5. 所有微调完成，执行前进4步
            common.loginfo(f"已到达放块目标位置，开始前进4步")
            self.gait_manager.set_step([320, 0.25, 0.02], 0.018, 0, 0, None, 0, 5)  # 前进4步
            rospy.sleep(1.0)  # 等待前进动作完成

            # 6. 执行放置物块动作
            common.loginfo(f"前进完成，调用放置动作组: {self.place_block_action_name}")
            self.motion_manager.run_action(self.place_block_action_name)
            rospy.sleep(0.5)

            # 7. 重置步态参数，完成放块流程
            self.visual_patrol.update_go_gait(x_max=0.028)
            self.visual_patrol.update_turn_gait(x_max=0.022)
            self.slow = True
            return True  # 放块完成

        else:
            # 未检测到交叉点时的处理
            self.count = 0
            if hasattr(self, 'missed_black_flag_count'):
                self.missed_black_flag_count += 1
            else:
                self.missed_black_flag_count = 1
            common.loginfo(f"未检测到交叉点，累计计数: {self.missed_black_flag_count}")

            # 超过阈值时强制执行（避免卡壳）
            if self.missed_black_flag_count > 3:
                common.loginfo(f"超过阈值，强制前进4步并放置")
                self.gait_manager.set_step([320, 0.25, 0.02], 0.018, 0, 0, None, 0, 3)
                rospy.sleep(1.0)
                self.motion_manager.run_action(self.place_block_action_name)
                self.missed_black_flag_count = 0
                return True

        return False

    def run(self):
        common.loginfo("开始执行主循环")
        while self.running:
            if self.start:
                # common.loginfo(f"调用动作组: hand_back_4")
                self.motion_manager.run_action('hand_back_4')
                # 获取识别结果
                line_data = None
                side_data = None
                block_data = None
                intersection_data = None
                for object_info in self.objects_info:
                    if object_info.type == 'line':
                        line_data = object_info
                    if object_info.type == 'side':
                        side_data = object_info
                    if object_info.type == 'circle':
                        block_data = object_info
                    if object_info.type == 'intersection':
                        intersection_data = object_info

                # common.loginfo(f"当前状态: {self.current_state}, 下一状态: {self.next_state}")

                if self.current_state == 'visual_patrol':
                    if line_data is not None:
                        self.visual_patrol.process(line_data.x, line_data.width)
                        # common.loginfo(f"调用动作组: hand_back_4")
                        self.motion_manager.run_action('hand_back_4')
                    else:
                        common.loginfo("visual_patrol状态: 未检测到线数据")
                elif self.current_state == 'hurdles':
                    if self.exit_hurdles(side_data):
                        self.current_state = 'visual_patrol'
                        self.next_state = 'climb_stairs'
                        self.state[self.current_state][2] = False  # 重新初始化当前阶段
                        common.loginfo('状态切换: exit hurdles ---> enter visual_patrol')
                    else:
                        rospy.sleep(0.8)

                if self.current_state == 'visual_patrol':
                    if line_data is not None:
                        self.visual_patrol.process(line_data.x, line_data.width)
                    else:
                        common.loginfo("visual_patrol状态: 未检测到线数据")
                elif self.current_state == 'climb_stairs':
                    if self.exit_climb_stairs(side_data):
                        self.current_state = 'visual_patrol'
                        self.next_state = 'descend_stairs'
                        self.state[self.current_state][2] = False  # 重新初始化当前阶段
                        common.loginfo('状态切换: exit climb_stairs ---> enter visual_patrol')
                    else:
                        rospy.sleep(0.8)

                elif self.current_state == 'descend_stairs':
                    if self.exit_descend_stairs(side_data):
                        self.current_state = 'visual_patrol'
                        self.next_state = 'crawl_left'
                        self.state[self.current_state][2] = False
                        common.loginfo('状态切换: exit descend_stairs ---> enter visual_patrol')
                    else:
                        rospy.sleep(0.8)

                elif self.current_state == 'crawl_left':
                    # 如果当前状态是左抓取
                    if self.exit_crawl_left(block_data):
                        # 如果完成左抓取
                        self.current_state = 'visual_patrol'
                        self.next_state = 'crawl_right'
                        self.state[self.current_state][2] = False
                        common.loginfo('状态切换: exit crawl_left ---> enter visual_patrol')
                    else:
                        rospy.sleep(0.8)  # 等机体平稳下来

                elif self.current_state == 'crawl_right':
                    # 右抓取状态
                    if self.exit_crawl_right(block_data):
                        # 如果完成右抓取
                        self.current_state = 'visual_patrol'
                        self.next_state = 'place_block'
                        self.state[self.current_state][2] = False
                        common.loginfo('状态切换: exit crawl_right ---> enter visual_patrol')
                    else:
                        rospy.sleep(0.8)

                elif self.current_state == 'place_block':
                    if self.exit_place_block(intersection_data):
                        self.running = False
                        common.loginfo('状态切换: exit place_block ---> 任务完成')
                    else:
                        rospy.sleep(0.8)

                # 是否退出巡线，进入下一阶段
                if self.next_state == 'hurdles':
                    if self.enter_hurdles(side_data):
                        self.current_state = 'hurdles'
                        self.next_state = 'visual_patrol'
                        common.loginfo('状态切换: exit visual_patrol ---> enter hurdles')
                elif self.next_state == 'climb_stairs':
                    if self.enter_climb_stairs(side_data):
                        self.current_state = 'climb_stairs'
                        self.next_state = 'visual_patrol'
                        common.loginfo('状态切换: exit visual_patrol ---> enter climb_stairs')
                elif self.next_state == 'descend_stairs':
                    if self.enter_descend_stairs(side_data):
                        self.current_state = 'descend_stairs'
                        self.next_state = 'visual_patrol'
                        common.loginfo('状态切换: exit visual_patrol ---> enter descend_stairs')

                elif self.next_state == 'crawl_left':
                    if self.enter_crawl_left(block_data):
                        self.current_state = 'crawl_left'
                        self.next_state = 'visual_patrol'
                        common.loginfo('状态切换: exit visual_patrol ---> enter crawl_left')
                elif self.next_state == 'crawl_right':
                    if self.enter_crawl_right(block_data):
                        self.current_state = 'crawl_right'
                        self.next_state = 'visual_patrol'
                        common.loginfo('状态切换: exit visual_patrol ---> enter crawl_right')
                elif self.next_state == 'place_block':
                    self.visual_patrol.update_go_gait(x_max=0.02)

                    if time.time() > self.delay_time:
                        if self.enter_place_block(intersection_data):
                            self.current_state = 'place_block'
                            self.next_state = 'visual_patrol'
                            common.loginfo('状态切换: exit visual_patrol ---> enter place_block')
                    else:
                        common.loginfo(f"等待抓取完成，剩余时间: {self.delay_time - time.time():.2f}秒")

                self.state_init(self.current_state, self.next_state)

                rospy.sleep(0.01)  # 防止空载
            else:
                rospy.sleep(0.01)





        common.loginfo(f'日志已保存到 {self.logger.log.name}')
        common.loginfo("主循环结束，节点关闭")

        self.init_action(self.head_pan_init, self.head_tilt_init)
        self.stop_srv_callback(None)
        rospy.signal_shutdown('shutdown')


if __name__ == "__main__":
    CompetitionNode('competition').run()