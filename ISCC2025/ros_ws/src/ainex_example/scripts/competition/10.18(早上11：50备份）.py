#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/09/27
# @author:aiden

import rospy
import signal
import time
from ainex_sdk import misc, common
from ainex_example.color_common import Common
from ainex_example.visual_patrol import VisualPatrol
from ainex_example.approach_object import ApproachObject
from ainex_interfaces.srv import SetString
from ainex_interfaces.msg import ObjectsInfo, ColorDetect, ROI


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
    hurdles_action_name = 'hurdles_fast1'
    climb_stairs_action_name = 'climb_stairs_competition_fast2'
    descend_stairs_action_name = 'descend_stairs_fast1'
    crawl_left_action_name = 'crawl_left_3'  # 左行走动作名称
    crawl_right_action_name = 'crawl_right_4'  # 右行走动作名称
    place_block_action_name = 'place_block_1'  # 放置方块动作名称
    forward_step = 'forward_one_step_1'  # 前进一步
    back_step = 'back_step_1'  # 后退一步
    move_left = 'move_left_1'  # 左移
    move_right = 'move_right_1'  # 右移
    # self.motion_manager.run_action(self.back_step)
    # 图像处理时缩放到这个分辨率， 不建议修改
    image_process_size = [160, 120]
    # 上台阶状态下的目标位置阈值
    enter_climb_stairs_y = 185 / 480  # 当检测到的标识像素坐标y值占图像的比例大于此值时进入此阶段
    climb_stairs_x_stop = 0.5  # 当检测到的标识像素坐标x值占图像的比例在此值附近(范围可在ApproachObject里设置)时停止前后移动
    climb_stairs_y_stop = 286 / 480  # 当检测到的标识像素坐标y值占图像的比例在此值附近(范围可在ApproachObject里设置)时停止横向移动
    climb_stairs_yaw_stop = 0  # 当检测到的标识角度在此值附近(范围可在ApproachObject里设置)时停止旋转移动
    # 下台阶状态下的目标位置阈值
    enter_descend_stairs_y = 180 / 480  # 150
    descend_stairs_x_stop = 0.5
    descend_stairs_y_stop = 265 / 480
    descend_stairs_yaw_stop = 0
    # 跨栏状态下的目标位置阈值
    enter_hurdles_y = 60 / 480
    hurdles_x_stop = 0.5
    hurdles_y_stop = 88 / 480
    hurdles_yaw_stop = 0
    # 向左抓取状态下的目标位置阈值
    enter_crawl_left_y = 202 / 480  # 当检测到的标识像素坐标y值占图像的比例大于此值时进入此阶段
    crawl_left_x_stop = 225 / 640  # 左行走的x轴目标位置范围可在ApproachObject里设置)时停止前后移动
    crawl_left_y_stop = 235 / 480  # 左行走的y轴目标位置(范围可在ApproachObject里设置)时停止横向移动
    crawl_left_yaw_stop = 0  # 当检测到的标识角度在此值附近(范围可在ApproachObject里设置)时停止旋转移动
    # 向右抓取状态下的目标位置阈值，原理同上
    enter_crawl_right_y = 85 / 640  # 120;500
    crawl_right_x_stop = 390 / 640
    crawl_right_y_stop = 80 / 480
    crawl_right_yaw_stop = 0
    # 放块状态下的目标位置阈值
    enter_place_block_y = 270 / 480
    place_block_x_stop = 300 / 640
    place_block_y_stop = 320 / 480
    place_block_yaw_stop = 0

    def __init__(self, name):
        rospy.init_node(name)
        self.calib_config = common.get_yaml_data('/home/ubuntu/ros_ws/src/ainex_example/config/calib.yaml')
        self.name = name
        self.count = 0
        self.running = True
        self.slow = True
        self.objects_info = []
        self.delay_time = 0
        self.current_state = "visual_patrol"  # 当前状态
        self.next_state = "hurdles"  # 下一状态

        self.state = {'visual_patrol': [[500, 260],
                                        ['black', self.line_roi, self.image_process_size, self.set_visual_patrol_color],
                                        False],  # 巡线
                      'climb_stairs_competition_fast2': [[500, 260],
                                                         ['red', self.stairs_roi, self.image_process_size,
                                                          self.set_stairs_color], False],
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

        self.approach_object = ApproachObject(self.gait_manager)
        self.visual_patrol = VisualPatrol(self.gait_manager)
        # 减小巡线步幅，提高其他标志检测稳定性
        self.visual_patrol.update_go_gait(x_max=0.028)
        self.visual_patrol.update_turn_gait(x_max=0.022)

        signal.signal(signal.SIGINT, self.shutdown)

        # 订阅颜色识别结果
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)
        rospy.Service('~set_color', SetString, self.set_color_srv_callback)  # 设置颜色
        self.motion_manager.run_action('walk_ready_2')

        if rospy.get_param('~start', True):
            # 通知颜色识别准备，此时只显示摄像头原画
            self.enter_func(None)
            self.start_srv_callback(None)
            common.loginfo('start ombination')

    def shutdown(self, signum, frame):
        self.running = False
        common.loginfo('%s shutdown' % self.name)

    def set_visual_patrol_color(self, color, roi, image_process_size):
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
        # 设置颜色
        block_param = self.set_block_color(self.state['crawl_left'][1][0])
        stairs_param = self.set_stairs_color(self.state['climb_stairs_competition_fast2'][1][0])
        line_param = self.set_visual_patrol_color(self.state['visual_patrol'][1][0])
        hurdles_param = self.set_hurdles_color(self.state['hurdles'][1][0])
        intersection_param = self.set_intersection_color(self.state['place_block'][1][0])

        self.detect_pub.publish([line_param, stairs_param, hurdles_param, intersection_param, block_param])
        common.loginfo('%s set_color' % self.name)

        return [True, 'set_color']

    def get_color_callback(self, msg):
        # 获取颜色识别结果
        self.objects_info = msg.data

    def state_init(self, current_state, next_state):
        # 不同阶段的初始化
        if self.state[current_state][2] == False:
            self.state[current_state][2] = True
            self.init_action(self.state[current_state][0][0], self.state[current_state][0][1])  # 头部姿态
            param1 = self.state[current_state][1][3](self.state[current_state][1][0], self.state[current_state][1][1],
                                                     self.state[current_state][1][2])
            param2 = self.state[next_state][1][3](self.state[next_state][1][0], self.state[next_state][1][1],
                                                  self.state[next_state][1][2])
            self.detect_pub.publish([param1, param2])  # 颜色检测设置
            common.loginfo(current_state + ' init')

    # 退出上台阶

    def exit_climb_stairs(self, stairs_data):
        # 上阶梯处理
        if stairs_data is not None:
            print("state:exit_climb_stairs ", "stairs_data.y:", stairs_data.y)

            # 只校准最大 3 帧
            frame_count = 0
            calibrated = False

            while frame_count < 0:
                if self.approach_object.process(
                        max(stairs_data.y, stairs_data.left_point[1], stairs_data.right_point[1]),
                        stairs_data.x,
                        stairs_data.angle,
                        self.climb_stairs_y_stop * stairs_data.height,
                        self.climb_stairs_x_stop * stairs_data.width,
                        self.climb_stairs_yaw_stop,
                        stairs_data.width,
                        stairs_data.height
                ):
                    calibrated = True
                    break
                frame_count += 1

            # 超过3帧或校准成功后执行上台阶动作
            self.gait_manager.disable()  # 关闭步态控制
            common.loginfo('climb_stairs_competition_fast2')
            # self.gait_manager.set_step([420, 0.22, 0.02], 0.015, 0, 0, None, 0, 3)
            self.motion_manager.run_action(self.move_right)
            self.motion_manager.run_action(self.move_right)
            self.motion_manager.run_action(self.move_right)
            self.motion_manager.run_action(self.move_right)
            # self.motion_manager.run_action(self.forward_step)
            self.gait_manager.set_step([320, 0.25, 0.02], 0.015, 0, 0, None, 0, 2)
            self.motion_manager.run_action(self.climb_stairs_action_name)  # 执行上台阶动作
            rospy.sleep(0.5)
            self.motion_manager.run_action('walk_ready')
            rospy.sleep(0.5)
            return True

        else:
            self.count = 0
            # 未识别到红色标志，累计计数
            if hasattr(self, 'missed_red_flag_count'):
                self.missed_red_flag_count += 1
            else:
                self.missed_red_flag_count = 1

            # 超过阈值未识别红色标志，直接执行上台阶动作
            if self.missed_red_flag_count > 2:
                self.motion_manager.run_action(self.move_right)
                self.motion_manager.run_action(self.move_right)
                self.motion_manager.run_action(self.move_right)
                # self.motion_manager.run_action(self.forward_step)
                self.gait_manager.set_step([320, 0.25, 0.02], 0.018, 0, 0, None, 0, 2)
                self.motion_manager.run_action(self.climb_stairs_action_name)
                self.missed_red_flag_count = 0
                return True

    # 退出下台阶

    def exit_descend_stairs(self, stairs_data):
        # 下阶梯处理
        if stairs_data is not None:
            self.missed_red_flag_count = 0
            print("state:exit_descend_stairs ", "stairs_data.y:", stairs_data.y)

            # 只校准最大 3 帧
            frame_count = 0
            calibrated = False
            self.motion_manager.run_action(self.move_right)
            self.motion_manager.run_action(self.move_right)

            while frame_count < 0:
                if self.approach_object.process(
                        max(stairs_data.y, stairs_data.left_point[1], stairs_data.right_point[1]),
                        stairs_data.x + self.calib_config['center_x_offset'],
                        stairs_data.angle,
                        self.descend_stairs_y_stop * stairs_data.height,
                        self.descend_stairs_x_stop * stairs_data.width,
                        self.descend_stairs_yaw_stop,
                        stairs_data.width,
                        stairs_data.height
                ):
                    calibrated = True
                    break
                frame_count += 1

            # 超过3帧或校准成功后执行下台阶动作
            self.gait_manager.disable()  # 关闭步态控制
            common.loginfo('descend_stairs')
            # self.motion_manager.run_action(self.forward_step)
            # self.motion_manager.run_action(self.forward_step)
            # self.motion_manager.run_action(self.forward_step)
            # self.motion_manager.run_action(self.forward_step)
            self.gait_manager.set_step([320, 0.22, 0.02], 0.012, 0, 0, None, 0, 2)
            self.motion_manager.run_action(self.move_right)
            self.motion_manager.run_action(self.move_right)
            self.motion_manager.run_action(self.move_right)
            self.motion_manager.run_action(self.descend_stairs_action_name)  # 执行下台阶动作
            rospy.sleep(0.5)
            self.visual_patrol.update_go_gait(x_max=0.028)
            self.visual_patrol.update_turn_gait(x_max=0.022)
            self.slow = True
            self.motion_manager.run_action('walk_ready')
            rospy.sleep(0.5)
            return True

        else:
            self.count = 0
            # 未识别到红色标志，累计计数
            if hasattr(self, 'missed_red_flag_count'):
                self.missed_red_flag_count += 1
            else:
                self.missed_red_flag_count = 1

            # 超过阈值未识别红色标志，直接执行下台阶动作
            if self.missed_red_flag_count > 2:
                self.motion_manager.run_action(self.move_right)
                self.motion_manager.run_action(self.move_right)
                self.motion_manager.run_action(self.move_right)
                self.motion_manager.run_action(self.forward_step)
                self.motion_manager.run_action(self.descend_stairs_action_name)
                self.missed_red_flag_count = 0
                return True

    # 退出跨栏

    def exit_hurdles(self, hurdles_data):
        # 跨栏处理
        if hurdles_data is not None:
            self.missed_hurdles_flag_count = 0
            print("state:exit_hurdles ", "hurdles_data.y:", hurdles_data.y)

            # 只校准最大 3 帧
            frame_count = 0
            calibrated = False

            while frame_count < 0:
                if self.approach_object.process(
                        max(hurdles_data.y, hurdles_data.left_point[1], hurdles_data.right_point[1]),
                        hurdles_data.x + self.calib_config['center_x_offset'],
                        hurdles_data.angle,
                        self.hurdles_y_stop * hurdles_data.height,
                        self.hurdles_x_stop * hurdles_data.width,
                        self.hurdles_yaw_stop,
                        hurdles_data.width,
                        hurdles_data.height
                ):
                    calibrated = True
                    break
                frame_count += 1

            # 无论是否校准成功，超过3帧后都执行后续跨栏动作
            self.gait_manager.disable()
            common.loginfo('hurdles')
            self.gait_manager.set_step([420, 0.22, 0.02], 0.01, 0, 0, None, 0, 3)
            self.motion_manager.run_action(self.hurdles_action_name)
            rospy.sleep(0.5)
            self.visual_patrol.update_go_gait(x_max=0.028)
            self.visual_patrol.update_turn_gait(x_max=0.022)
            self.slow = True
            self.motion_manager.run_action('walk_ready')
            rospy.sleep(0.5)
            return True

        else:
            self.count = 0
            # 未识别到跨栏的累计计数
            if hasattr(self, 'missed_hurdles_flag_count'):
                self.missed_hurdles_flag_count += 1
            else:
                self.missed_hurdles_flag_count = 1

            # 超过阈值直接执行跨栏动作
            if self.missed_hurdles_flag_count > 1:
                self.motion_manager.run_action(self.move_right)
                self.motion_manager.run_action(self.hurdles_action_name)
                self.missed_hurdles_flag_count = 0
                return True

    # 进入上台阶
    def enter_climb_stairs(self, stairs_data):
        if stairs_data is not None:
            # self.missed_red_flag_count = 0
            print("state:enter_climb_stairs ", "stairs_data.y:", stairs_data.y)
            if stairs_data.y > self.stairs_roi[0] * stairs_data.height and self.slow:
                self.visual_patrol.update_go_gait(x_max=0.02)
                self.visual_patrol.update_turn_gait(x_max=0.018)
                self.slow = False
            if max(stairs_data.y, stairs_data.left_point[1],
                   stairs_data.right_point[1]) > self.enter_climb_stairs_y * stairs_data.height:
                self.count += 1
                if self.count > 3:  # 主线程比较快，颜色检测回调慢一点，需要连续检测来排除滞后干扰
                    self.count = 0
                    self.gait_manager.disable()
                    self.approach_object.update_approach_stop_value(20, 15, 4)  # 设置靠近目标停止的条件，分别为y, x, angle误差
                    self.motion_manager.run_action('hand_back')  # 手往后，防止遮挡
                    rospy.sleep(0.5)
                    return True
            else:
                self.count = 0
        return False

    # 进入下台阶
    def enter_descend_stairs(self, stairs_data):
        if stairs_data is not None:
            print("state:enter_descend_stairs ", "stairs_data.y:", stairs_data.y)
            if max(stairs_data.y, stairs_data.left_point[1],
                   stairs_data.right_point[1]) > self.enter_descend_stairs_y * stairs_data.height:
                self.visual_patrol.update_go_gait(x_max=0.012)
                self.visual_patrol.update_turn_gait(x_max=0.01)
                self.count += 1
                if self.count > 1:
                    self.count = 0
                    self.gait_manager.disable()
                    self.approach_object.update_approach_stop_value(20, 10, 3)
                    self.motion_manager.run_action('hand_back')  # 手往后，防止遮挡
                    rospy.sleep(0.5)
                    return True
            else:
                self.count = 0
        return False

    # 进入跨栏
    def enter_hurdles(self, hurdles_data):
        if hurdles_data is not None:
            print("state:enter_hurdles ", "hurdles_data.y:", hurdles_data.y)
            if hurdles_data.y > self.hurdles_roi[0] * hurdles_data.height and self.slow:
                self.visual_patrol.update_go_gait(x_max=0.015)
                self.visual_patrol.update_turn_gait(x_max=0.01)
                self.slow = False
            if max(hurdles_data.y, hurdles_data.left_point[1],
                   hurdles_data.right_point[1]) > self.enter_hurdles_y * hurdles_data.height:
                self.count += 1
                if self.count > 3:
                    self.count = 0
                    self.gait_manager.disable()
                    self.approach_object.update_approach_stop_value(y_approach_value=20, x_approach_value=15,
                                                                    yaw_approach_value=5)
                    self.motion_manager.run_action('hand_back')  # 手往后，防止遮挡
                    return True
            else:
                self.count = 0
        return False

    # 进入左抓取判断函数
    """
    def enter_crawl_left(self, block_data):

        if block_data is not None:
            print("state:enter_crawl_left",
                  " block_data.x:", block_data.x,
                  " block_data.y:", block_data.y)

            # 当目标在 ROI 区域且处于慢行模式时调整运动步幅
            if block_data.y > self.block_roi[0] * block_data.height and self.slow:
                self.visual_patrol.update_go_gait(x_max=0.015)  # 调整步伐更精准
                self.visual_patrol.update_turn_gait(x_max=0.012)  # 调整转向精度
                self.slow = False

            # 如果目标进入屏幕高度的前 1/4 区域，进行计数以确认进入状态
            if block_data.y > block_data.height / 4:
                self.count += 1
                if self.count > 5:  # 连续检测超过 5 次切换至左抓取状态
                    self.count = 0
                    self.gait_manager.disable()  # 禁用巡逻步态
                    return True
            else:
                self.count = 0
        else:
            self.count = 0
        return False
    """

    def enter_crawl_left(self, block_data):
        if block_data is not None:
            print("state:enter_crawl_left", " block_data.x:", block_data.x, " block_data.y:", block_data.y)
            if block_data.y > self.block_roi[0] * block_data.height and self.slow:
                self.visual_patrol.update_go_gait(x_max=0.02)
                self.visual_patrol.update_turn_gait(x_max=0.015)
                self.slow = False
            # 如果色块的y坐标大于高度的1/4,说明有一部分进入ROI区域
            if block_data.y > block_data.height / 4:
                self.count += 1
                if self.count > 5:  # 连续检测到满足条件,切换到左行走状态
                    self.count = 0
                    self.gait_manager.disable()
                    return True
            else:
                self.count = 0
        else:
            self.count = 0
        return False

    # 进入右抓取判断函数
    """
    def enter_crawl_right(self, block_data):

        if block_data is not None:
            print("state:enter_crawl_right",
                  " block_data.x:", block_data.x,
                  " block_data.y:", block_data.y)

            # 当目标在 ROI 区域且处于慢行模式时调整运动步幅
            if block_data.y > self.block_roi[0] * block_data.height and self.slow:
                self.visual_patrol.update_go_gait(x_max=0.015)  # 步伐更精准
                self.visual_patrol.update_turn_gait(x_max=0.012)  # 转向更平稳
                self.slow = False

            # 如果目标进入屏幕高度的前 1/4 区域，进行计数确认状态
            if block_data.y > block_data.height / 4:
                self.count += 1
                if self.count > 5:  # 连续检测超过 5 次切换至右抓取状态
                    self.count = 0
                    self.gait_manager.disable()
                    return True
            else:
                self.count = 0
        else:
            self.count = 0
        return False

    # 进入放置判断函数
    """

    def enter_crawl_right(self, block_data):
        if block_data is not None:
            print("state:enter_crawl_right", " block_data.x:", block_data.x, " block_data.y:", block_data.y)
            if block_data.y > self.block_roi[0] * block_data.height and self.slow:
                self.visual_patrol.update_go_gait(x_max=0.015)
                self.visual_patrol.update_turn_gait(x_max=0.015)
                self.slow = False
            if block_data.y > block_data.height / 4:
                self.count += 1
                if self.count > 5:
                    self.count = 0
                    self.gait_manager.disable()
                    return True
            else:
                self.count = 0
        else:
            self.count = 0
        return False

    def enter_place_block(self, line_data):
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
        min_vertical_span = 8  # 垂直方向跨度下限（避免噪点）
        angle_threshold_deg = 30  # 认为“横向”允许的最大角度偏差（度）
        self.visual_patrol.update_go_gait(x_max=0.015)
        self.visual_patrol.update_turn_gait(x_max=0.012)
        # --- 若本帧无检测到任何 line_data：仅重置正向计数并返回 False ---
        if line_data is None:
            self.count = 0
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
                print("DEBUG: enter_place_block: invalid line_data structure:", e)
                print("DEBUG: line_data:", line_data)
            except Exception:
                pass
            self.count = 0
            return False

        # --- 计算几何特征：横向跨度、垂直跨度、与水平线的角度（度） ---
        horiz_span = abs(rx - lx)
        vertical_span = abs(ry - ly)
        dx = rx - lx
        dy = ry - ly
        import math
        # 若线段长度为0，给一个大角度值以避免误判
        if dx == 0 and dy == 0:
            angle_deg = 90.0
        else:
            angle_deg = abs(math.degrees(math.atan2(dy, dx)))
            if angle_deg > 90:
                angle_deg = 180 - angle_deg

        # 调试输出，便于定位哪个条件没满足
        try:
            print("DEBUG: enter_place_block: y_val:", y_val,
                  "horiz_span:", horiz_span, "span_threshold:", span_threshold,
                  "vertical_span:", vertical_span, "min_vertical_span:", min_vertical_span,
                  "angle_deg:", angle_deg, "angle_threshold_deg:", angle_threshold_deg)
        except Exception:
            pass

        # --- 判定条件：考虑横条特征（足够宽且近似水平） ---
        is_horizontal_enough = (horiz_span >= span_threshold) and (angle_deg <= angle_threshold_deg)

        if is_horizontal_enough:
            # 满足目标条件：累加正向计数并在达到阈值后停止并返回 True
            self.count += 1
            try:
                print(f"DEBUG: enter_place_block: target-like detected, count={self.count}/{positive_frames_threshold}")
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
                        print("DEBUG: gait_manager.stop() exception:", e)
                    except Exception:
                        pass
                try:
                    common.loginfo('intersection detect')
                except Exception:
                    pass
                return True
            else:
                return False
        else:
            # 有数据但不满足“放置点”条件：只重置正向计数，继续等待下一帧
            self.count = 0
            try:
                reasons = []
                if horiz_span < span_threshold:
                    reasons.append(f"horiz_span too small ({horiz_span}<{span_threshold})")
                if angle_deg > angle_threshold_deg:
                    reasons.append(f"angle too large ({angle_deg:.1f}deg>{angle_threshold_deg}deg)")
                if vertical_span < min_vertical_span:
                    reasons.append(f"vertical_span too small ({vertical_span}<{min_vertical_span})")
                print("DEBUG: enter_place_block: not target ->", "; ".join(reasons))
            except Exception:
                pass
            return False

    #####

    # 退出左抓取判断函数
    def exit_crawl_left(self, block_data):
        """
        退出左抓取状态逻辑：
        保证机器人在靠近目标时进行精确的左右调整和前进控制，使停靠位置更加精确。
        """
        if block_data is not None:
            print("state:exit_crawl_left",
                  " block_data.x:", block_data.x,
                  " block_data.y:", block_data.y)

            centerx_left = self.crawl_left_x_stop * block_data.width
            offset_x = block_data.x - centerx_left

            # 根据左右偏差进行微调
            if offset_x > 15:  # 阈值从 20 改为 15 提高精度
                self.motion_manager.run_action(self.move_right)
            elif offset_x < -15:
                self.motion_manager.run_action(self.move_left)

            # 如果未到达目标高度，缓慢向前推进
            elif block_data.y < self.crawl_left_y_stop * block_data.height:
                self.motion_manager.run_action(self.forward_step)
                time.sleep(0.15)  # 缩短间隔防止过冲

            # 到达目标位置，执行抓取动作
            elif block_data.y >= self.crawl_left_y_stop * block_data.height:
                time.sleep(0.3)  # 减少延迟，加快响应
                self.motion_manager.run_action(self.crawl_left_action_name)
                self.visual_patrol.update_go_gait(arm_swap=0)
                self.visual_patrol.update_turn_gait(arm_swap=0)
                self.visual_patrol.update_go_gait(x_max=0.028)
                self.visual_patrol.update_turn_gait(x_max=0.022)
                self.slow = True
                return True
        else:
            self.motion_manager.run_action(self.back_step)
            time.sleep(0.01)

        return False

    # 退出右抓取判断函数

    def exit_crawl_right(self, block_data):
        """
        退出右抓取状态逻辑：
        保证机器人在靠近目标时进行精确的左右调整和前进控制，使停靠位置更加精确。
        """

        if block_data is not None:
            print("state:exit_crawl_right",
                  " block_data.x:", block_data.x,
                  "block_data.y:", block_data.y)

            centerx_right = self.crawl_right_x_stop * block_data.width
            offset_x = block_data.x - centerx_right
            # 根据左右偏差进行微调
            if offset_x > 15:  # 阈值由 20 降至 15 提高精度
                self.motion_manager.run_action(self.move_right)
            elif offset_x < -15:
                self.motion_manager.run_action(self.move_left)

            # 如果未到达目标高度，缓慢向前推进
            elif block_data.y < self.crawl_right_y_stop * block_data.height:
                self.motion_manager.run_action(self.forward_step)
                time.sleep(0.15)  # 缩短时间防止过冲

            # 到达目标位置，执行抓取动作
            elif block_data.y >= self.crawl_right_y_stop * block_data.height:
                time.sleep(0.3)  # 减少延迟，加快响应

                self.motion_manager.run_action(self.crawl_right_action_name)
                self.visual_patrol.update_go_gait(arm_swap=30)
                self.visual_patrol.update_turn_gait(arm_swap=30)
                self.visual_patrol.update_go_gait(x_max=0.02)
                self.visual_patrol.update_turn_gait(x_max=0.018)
                self.slow = True
                self.delay_time = time.time() + 13.8  # 抓取完成延时再识别
                return True
        else:
            self.motion_manager.run_action(self.back_step)
            time.sleep(0.01)

        return False

    # 退出放块判断函数

    def exit_place_block(self, line_data):
        # 放置方块处理
        if line_data is not None:
            self.missed_black_flag_count = 0
            print("state:exit_place_block ", "line_data.y:", line_data.y)

            # 只校准最大 3 帧
            frame_count = 0
            calibrated = False

            while frame_count < 1:
                if self.approach_object.process(
                        max(line_data.y, line_data.left_point[1], line_data.right_point[1]),
                        line_data.x,
                        line_data.angle,
                        self.place_block_y_stop * line_data.height,
                        self.place_block_x_stop * line_data.width,
                        self.place_block_yaw_stop,
                        line_data.width,
                        line_data.height
                ):
                    calibrated = True
                    break
                frame_count += 1

            # 超过3帧或校准成功后执行放置方块动作
            self.gait_manager.disable()  # 关闭步态控制
            walking_param = self.gait_manager.get_gait_param()  # 获取当前的步态参数
            walking_param['body_height'] = 0.015  # 设置身体高度,单位米
            walking_param['pelvis_offset'] = 0  # 设置骨盆位置的前后偏移量,单位度
            walking_param['step_height'] = 0.01  # 设置步高,单位米
            walking_param['hip_pitch_offset'] = 20  # 设置髋关节角度偏移量,单位度
            walking_param['z_swap_amplitude'] = 0.006  # 左右足高交替振幅,单位米
            self.motion_manager.run_action(self.move_right)
            self.motion_manager.run_action(self.move_right)
            # self.motion_manager.run_action(self.move_right)
            # self.motion_manager.run_action(self.move_right)
            # self.motion_manager.run_action(self.move_right)

            self.gait_manager.set_step([320, 0.25, 0.02], 0.02, 0, 0, None, 0, 3)
            common.loginfo('place_block')
            self.motion_manager.run_action(self.place_block_action_name)
            return True

        else:
            self.count = 0
            # 未识别到黑色标志，累计计数
            if hasattr(self, 'missed_black_flag_count'):
                self.missed_black_flag_count += 1
            else:
                self.missed_black_flag_count = 1

            # 超过阈值直接执行放置方块动作
            if self.missed_black_flag_count > 2:
                walking_param = self.gait_manager.get_gait_param()
                walking_param['body_height'] = 0.015
                walking_param['pelvis_offset'] = 0
                walking_param['step_height'] = 0.01
                walking_param['hip_pitch_offset'] = 20
                walking_param['z_swap_amplitude'] = 0.006
                # self.motion_manager.run_action(self.move_left)
                # self.motion_manager.run_action(self.move_left)
                # self.motion_manager.run_action(self.move_left)
                self.motion_manager.run_action(self.move_left)

                self.motion_manager.run_action(self.move_right)
                self.gait_manager.set_step([320, 0.25, 0.02], 0.018, 0, 0, None, 0, 3)
                common.loginfo('place_block')
                self.motion_manager.run_action(self.place_block_action_name)
                self.missed_black_flag_count = 0
                return True

    def run(self):
        while self.running:
            if self.start:
                self.motion_manager.run_action('hand_back')
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
                    # print(object_info)

                if self.current_state == 'visual_patrol':
                    if line_data is not None:
                        self.visual_patrol.process(line_data.x, line_data.width)
                        self.motion_manager.run_action('hand_back')
                elif self.current_state == 'hurdles':
                    if self.exit_hurdles(side_data):
                        self.current_state = 'visual_patrol'
                        self.next_state = 'climb_stairs_competition_fast2'
                        self.state[self.current_state][2] = False  # 重新初始化当前阶段
                        common.loginfo('exit hurdles ---> enter visual_patrol')
                    else:
                        rospy.sleep(0.8)

                if self.current_state == 'visual_patrol':
                    if line_data is not None:
                        self.visual_patrol.process(line_data.x, line_data.width)
                elif self.current_state == 'climb_stairs_competition_fast2':
                    if self.exit_climb_stairs(side_data):
                        self.current_state = 'visual_patrol'
                        self.next_state = 'descend_stairs'
                        self.state[self.current_state][2] = False  # 重新初始化当前阶段
                        common.loginfo('exit climb_stairs ---> enter visual_patrol')
                    else:
                        rospy.sleep(0.8)

                elif self.current_state == 'descend_stairs':
                    if self.exit_descend_stairs(side_data):
                        self.current_state = 'visual_patrol'
                        self.next_state = 'crawl_left'
                        self.state[self.current_state][2] = False
                        common.loginfo('exit descend_stairs ---> enter visual_patrol')
                    else:
                        rospy.sleep(0.8)

                elif self.current_state == 'crawl_left':
                    # 如果当前状态是左抓取
                    if self.exit_crawl_left(block_data):
                        # 如果完成左抓取
                        self.current_state = 'visual_patrol'
                        self.next_state = 'crawl_right'
                        self.state[self.current_state][2] = False
                    else:
                        rospy.sleep(0.8)  # 等机体平稳下来

                elif self.current_state == 'crawl_right':
                    # 右抓取状态
                    if self.exit_crawl_right(block_data):
                        # 如果完成右抓取
                        self.current_state = 'visual_patrol'
                        self.next_state = 'place_block'
                        self.state[self.current_state][2] = False
                    else:
                        rospy.sleep(0.8)

                elif self.current_state == 'place_block':
                    if self.exit_place_block(intersection_data):
                        self.running = False
                        common.loginfo('exit place_block ---> enter visual_patrol')
                    else:
                        rospy.sleep(0.8)

                # 是否退出巡线，进入下一阶段
                if self.next_state == 'hurdles':
                    if self.enter_hurdles(side_data):
                        self.current_state = 'hurdles'
                        self.next_state = 'visual_patrol'
                        common.loginfo('exit visual_patrol ---> enter hurdles')
                elif self.next_state == 'climb_stairs_competition_fast2':
                    if self.enter_climb_stairs(side_data):
                        self.current_state = 'climb_stairs_competition_fast2'
                        self.next_state = 'visual_patrol'
                        common.loginfo('exit visual_patrol ---> enter climb_stairs')
                elif self.next_state == 'descend_stairs':
                    if self.enter_descend_stairs(side_data):
                        self.current_state = 'descend_stairs'
                        self.next_state = 'visual_patrol'
                        common.loginfo('exit visual_patrol ---> enter descend_stairs')

                elif self.next_state == 'crawl_left':
                    if self.enter_crawl_left(block_data):
                        self.current_state = 'crawl_left'
                        self.next_state = 'visual_patrol'
                        common.loginfo('exit visual_patrol ---> enter crawl_left')
                elif self.next_state == 'crawl_right':
                    if self.enter_crawl_right(block_data):
                        self.current_state = 'crawl_right'
                        self.next_state = 'visual_patrol'
                        common.loginfo('exit visual_patrol ---> enter crawl_right')
                elif self.next_state == 'place_block':
                    self.visual_patrol.update_go_gait(x_max=0.02)
                    # print(time.time(),self.delay_time)

                    if time.time() > self.delay_time:
                        if self.enter_place_block(intersection_data):
                            self.current_state = 'place_block'
                            self.next_state = 'visual_patrol'

                self.state_init(self.current_state, self.next_state)

                rospy.sleep(0.01)  # 防止空载
            else:
                rospy.sleep(0.01)

        self.init_action(self.head_pan_init, self.head_tilt_init)
        self.stop_srv_callback(None)
        rospy.signal_shutdown('shutdown')


if __name__ == "__main__":
    CompetitionNode('competition').run()
