#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/07/04
# @author:aiden
# 巡线抓取
import rospy
import signal
from std_msgs.msg import Float64, String
from ainex_sdk import misc, common
from ainex_example.color_common import Common
from ainex_example.visual_patrol import VisualPatrol
from ainex_example.approach_object import ApproachObject
from ainex_interfaces.srv import SetString
from ainex_interfaces.msg import ObjectsInfo, ColorDetect, ROI

class VisualPatrolPickUp(Common):
    # 按顺序检测三个roi，如果检测到黑线立刻跳出
    line_roi = [(5 / 12, 6 / 12, 1 / 4, 3 / 4),
                (6 / 12, 7 / 12, 1 / 4, 3 / 4),
                (7 / 12, 8 / 12, 1 / 4, 3 / 4)
                ]

    block_roi = [0, 2 / 3, 0, 1]
    intersection_roi = [100 / 480, 340 / 480, 0, 1]
    # 向左左行走状态下的目标位置阈值
    enter_crawl_left_y = 290 / 480 # 当检测到的标识像素坐标y值占图像的比例大于此值时进入此阶段
    crawl_left_x_stop = 265 / 640  # 左行走的x轴目标位置范围可在ApproachObject里设置)时停止前后移动
    crawl_left_y_stop = 290 / 480  # 左行走的y轴目标位置(范围可在ApproachObject里设置)时停止横向移动
    crawl_left_yaw_stop = 0        # 当检测到的标识角度在此值附近(范围可在ApproachObject里设置)时停止旋转移动
    # 向左左行走状态下的目标位置阈值，原理同上
    enter_crawl_right_y = 280 / 480
    crawl_right_x_stop = 390 / 640
    crawl_right_y_stop = 272 / 480
    crawl_right_yaw_stop = 0
    # 放块状态下的目标位置阈值
    enter_place_block_y = 320 / 480
    place_block_x_stop = 320 / 640
    place_block_y_stop = 350 / 480
    place_block_yaw_stop = 0

    crawl_left_action_name = 'crawl_left'   # 左行走动作名称
    crawl_right_action_name = 'crawl_right' # 右行走动作名称
    place_block_action_name = 'place_block' # 放置方块动作名称
    
    image_process_size = [160, 120]

    def __init__(self, name):
        rospy.init_node(name)       # 初始化ROS节点
        self.name = name
        self.running = True         # 运行标志
        self.count = 0              # 计数器
        self.objects_info = []      # 存储识别结果
        self.current_state = "visual_patrol"    # 当前状态
        self.next_state = "crawl_left"          # 下一状态

        # 定义状态机
        self.state = {'visual_patrol':  [[500, 260], ['black', self.line_roi, self.image_process_size, self.set_visual_patrol_color], False],           # 巡线状态
                      'crawl_left':   [[500, 260], ['green', self.block_roi, self.image_process_size, self.set_block_color], False],                    # 左爬行状态
                      'crawl_right': [[500, 260], ['green', self.block_roi, self.image_process_size, self.set_block_color], False],                     # 右爬行状态
                      'place_block':   [[500, 260], ['black', self.intersection_roi, self.image_process_size, self.set_intersection_color], False]}     # 放块状态
        
        self.head_pan_init = self.state[self.current_state][0][0]   # 左右舵机的初始值
        self.head_tilt_init = self.state[self.current_state][0][1]  # 上下舵机的初始值
        super().__init__(name, self.head_pan_init, self.head_tilt_init) # 初始化父类

        self.visual_patrol = VisualPatrol(self.gait_manager)        # 巡线节点
        self.visual_patrol.update_go_gait(x_max=0.01)
        self.visual_patrol.update_turn_gait(x_max=0.01)
        self.approach_object = ApproachObject(self.gait_manager)    # 逼近节点
        signal.signal(signal.SIGINT, self.shutdown)

        # 订阅颜色识别结果
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)
        rospy.Service('~set_color', SetString, self.set_color_srv_callback)  # 设置颜色
        self.motion_manager.run_action('walk_ready')

        if rospy.get_param('~start', True):
            # 通知颜色识别准备，此时只显示摄像头原画
            self.enter_func(None)
            self.start_srv_callback(None)  # 开启颜色识别
            common.loginfo('start crawl')

    # 关闭节点
    def shutdown(self, signum, frame):
        self.running = False
        common.loginfo('%s shutdown' % self.name)

    # 生成巡线颜色识别参数 
    def set_visual_patrol_color(self, color, roi, image_process_size):
        # 设置巡线颜色
        line_param = ColorDetect()
        line_param.color_name = color
        line_param.use_name = True
        line_param.detect_type = 'line'     # 检测类型为线条
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
        block_param.max_area = image_process_size[0]*image_process_size[1]
        
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
    # 设置颜色服务回调函数
    def set_color_srv_callback(self, msg):
        # 设置颜色
        block_param = self.set_block_color(msg.data)
        line_param = self.set_visual_patrol_color('black')
        intersection_param = self.set_intersection_color('black')
        
        self.detect_pub.publish([line_param, block_param, intersection_param])
        common.loginfo('%s set_color' % self.name)
        
        return [True, 'set_color']
    # 获取颜色识别结果回调函数
    def get_color_callback(self, msg):
        # 获取颜色识别结果
        self.objects_info = msg.data
    # 状态初始化函数
    def state_init(self, current_state, next_state):
        # 不同阶段的初始化
        if self.state[current_state][2] == False:
            self.state[current_state][2] = True
            self.init_action(self.state[current_state][0][0], self.state[current_state][0][1])  # 头部姿态
            param1 = self.state[current_state][1][3](self.state[current_state][1][0], self.state[current_state][1][1], self.state[current_state][1][2])
            param2 = self.state[next_state][1][3](self.state[next_state][1][0], self.state[next_state][1][1], self.state[next_state][1][2])
            self.detect_pub.publish([param1, param2])  # 颜色检测设置
            common.loginfo('%s init' % current_state)
    # 进入左行走判断函数
    def enter_crawl_left(self, block_data):
        if block_data is not None:
            # 如果色块的y坐标大于高度的1/4,说明有一部分进入ROI区域
            if block_data.y > block_data.height / 4:
                self.count += 1
                if self.count > 5:  # 连续检测到满足条件,切换到左行走状态
                    self.count = 0
                    self.gait_manager.disable()
                    self.motion_manager.run_action('hand_back')  # 手往后，防止遮挡
                    return True
            else:
                self.count = 0
        else:
            self.count = 0
        return False
    # 进入右行走判断函数
    def enter_crawl_right(self, block_data):
        if block_data is not None:
            if block_data.y > block_data.height / 4:
                self.count += 1
                if self.count > 5:
                    self.count = 0
                    self.gait_manager.disable()
                    self.motion_manager.run_action('hand_back')  # 手往后，防止遮挡
                    return True
            else:
                self.count = 0
        else:
            self.count = 0
        return False
    # 进入右爬行判断函数
    def enter_place_block(self, line_data):
        if line_data is not None:
            if max(line_data.y, line_data.left_point[1], line_data.right_point[1]) > self.enter_place_block_y * line_data.height:
                self.count += 1
                if self.count > 2:
                    self.count = 0
                    self.gait_manager.stop()
                    common.loginfo('intersection detect')
                    return True
            else:
                self.count = 0
        else:
            self.count = 0
        return False
    # 退出左行走判断函数
    def exit_crawl_left(self, block_data):
        if block_data is not None:
            if self.approach_object.process(block_data.y, block_data.x, block_data.angle, 
                                            self.crawl_left_y_stop*block_data.height, self.crawl_left_x_stop*block_data.width, self.crawl_left_yaw_stop, block_data.width, block_data.height):
                self.gait_manager.disable()
                common.loginfo('crawl_left')
                self.motion_manager.run_action(self.crawl_left_action_name)
                self.visual_patrol.update_go_gait(arm_swap=0)
                self.visual_patrol.update_turn_gait(arm_swap=0)
                return True
        return False
    # 退出右行走判断函数
    def exit_crawl_right(self, block_data):
        if block_data is not None:
            if self.approach_object.process(block_data.y, block_data.x, block_data.angle, 
                                            self.crawl_right_y_stop*block_data.height, self.crawl_right_x_stop*block_data.width, self.crawl_right_yaw_stop, block_data.width, block_data.height):
                self.gait_manager.disable()
                common.loginfo('crawl_right')
                self.motion_manager.run_action(self.crawl_right_action_name)
                self.visual_patrol.update_go_gait(arm_swap=0)
                self.visual_patrol.update_turn_gait(arm_swap=0)
                return True
        return False
    # 退出放块判断函数
    def exit_place_block(self, line_data):
        if line_data is not None:
            if self.approach_object.process(max(line_data.y, line_data.left_point[1], line_data.right_point[1]), line_data.x, line_data.angle, 
                                            self.place_block_y_stop*line_data.height, self.place_block_x_stop*line_data.width, self.place_block_yaw_stop, line_data.width, line_data.height):
                self.gait_manager.disable()  # 关闭步态控制
                walking_param = self.gait_manager.get_gait_param()  # 获取当前的步态参数
                walking_param['body_height'] = 0.015                # 设置身体高度,单位米
                walking_param['pelvis_offset'] = 7                  # 设置骨盆位置的前后偏移量,单位度
                walking_param['step_height'] = 0.02                 # 设置步高,单位米
                walking_param['hip_pitch_offset'] = 20              # 设置髋关节角度偏移量,单位度
                walking_param['z_swap_amplitude'] = 0.006           # 设置左右足高度交替时的振幅,单位米
                # 调用set_step 设置步态参数,包括步长、步频等
                self.gait_manager.set_step([500, 0.2, 0.035], 0.02, 0, 0, walking_param, 0, 3)           
                common.loginfo('place_block')
                self.motion_manager.run_action(self.place_block_action_name)
                return True
        return False
    # 程序主循环
    def run(self):
        while self.running:
            if self.start:  # 如果start为True,表示程序开始运行
                line_data = None
                block_data = None
                intersection_data = None   
                # 从识别结果中提取出线、方块、交叉点的数据
                for object_info in self.objects_info:
                    if object_info.type == 'line':
                        line_data = object_info
                    if object_info.type == 'circle':
                        block_data = object_info
                    if object_info.type == 'intersection':
                        intersection_data = object_info

                # 当前阶段处理完成，回到巡线
                if self.current_state == 'visual_patrol':
                     # 如果当前状态是巡线
                    if line_data is not None:
                        # 如果检测到线条
                        # 执行巡线过程
                        self.visual_patrol.process(line_data.x, line_data.width)
                elif self.current_state == 'crawl_left':
                    # 如果当前状态是左行走
                    if self.exit_crawl_left(block_data):
                        # 如果完成左行走
                        self.current_state = 'visual_patrol'
                        self.next_state = 'crawl_right'
                        self.state[self.current_state][2] = False
                    else:
                        rospy.sleep(0.8)  # 等机体平稳下来
                elif self.current_state == 'crawl_right':
                    # 右行走状态
                    if self.exit_crawl_right(block_data):
                        # 如果完成右行走
                        self.current_state = 'visual_patrol'
                        self.next_state = 'place_block'
                        self.state[self.current_state][2] = False
                    else:
                        rospy.sleep(0.8)
                elif self.current_state == 'place_block':
                    # 放块状态
                    if self.exit_place_block(intersection_data):
                        self.running = False
                    else:
                        rospy.sleep(0.8)

                # # 状态切换判断，是否退出巡线，进入下一阶段
                if self.next_state == 'crawl_left':
                    if self.enter_crawl_left(block_data):
                        self.current_state = 'crawl_left'
                        self.next_state = 'visual_patrol'
                elif self.next_state == 'crawl_right':
                    if self.enter_crawl_right(block_data):
                        self.current_state = 'crawl_right'
                        self.next_state = 'visual_patrol'
                elif self.next_state == 'place_block':
                    if self.enter_place_block(intersection_data):
                        self.current_state = 'place_block'
                        self.next_state = 'visual_patrol'

                self.state_init(self.current_state, self.next_state)
                
                rospy.sleep(0.01)  # 防止空载
            else:
                rospy.sleep(0.01)
        # 程序结束,关闭节点
        self.init_action(self.head_pan_init, self.head_tilt_init)
        self.stop_srv_callback(None)
        rospy.signal_shutdown('shutdown')

if __name__ == "__main__":
    VisualPatrolPickUp('visual_patrol_pick_up').run()
