#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/07/10
# @author:aiden
# 颜色分拣
import copy
import rospy
import signal
from ainex_sdk import common
from ainex_example.color_common import Common
from ainex_interfaces.srv import SetString
from ainex_interfaces.msg import ObjectsInfo, ColorDetect

class ColorSortNode(Common):
    # y_min, y_max, x_min, x_max分别表示占图像的比例
    # 定义颜色块区域
    block_roi = [5/8, 7/8, 3/8, 5/8]
    # 定义放块动作名字
    right_hand_put_block_action_name = 'right_hand_put_block'   # 右手放块动作
    left_hand_put_block_action_name = 'left_hand_put_block'     # 左手放块动作

    # 图像处理时缩放到这个分辨率
    image_process_size = [160, 120]
   
    def __init__(self, name):
        # 定义放块动作名字
        rospy.init_node(name)
        self.name = name
        self.count = 0          # 计数器
        self.running = True
        self.objects_info = []  # 存储识别结果
        self.current_state = 'color_sort'   # 定义状态机

        # [['red', 'green', 'blue'], self.block_roi, self.image_process_size, self.set_blocks_color]
        # 颜色识别的参数设置,包括要识别的颜色、ROI区域、图像处理分辨率、生成颜色识别参数的方法。
        self.state = {'color_sort': [[500, 330], [['red', 'green', 'blue'], self.block_roi, self.image_process_size, self.set_blocks_color], False]}
        
        # 获取状态机初始头部姿态
        self.head_pan_init = self.state[self.current_state][0][0]   # 左右舵机的初始值
        self.head_tilt_init = self.state[self.current_state][0][1]  # 上下舵机的初始值
        super().__init__(name, self.head_pan_init, self.head_tilt_init)
        self.motion_manager.run_action('stand')
        signal.signal(signal.SIGINT, self.shutdown)

        # 订阅颜色识别结果
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)
        rospy.Service('~set_color', SetString, self.set_color_srv_callback)  # 设置颜色
        
        if rospy.get_param('~start', True):
            # 通知颜色识别准备，此时只显示摄像头原画
            self.enter_func(None)
            self.start_srv_callback(None)  # 开启颜色识别
            common.loginfo('start color_sort')
    # 关闭节点回调函数
    def shutdown(self, signum, frame):
        self.running = False
        common.loginfo('%s shutdown' % self.name)
    # 生成颜色块识别参数
    def set_blocks_color(self, colors, roi, image_process_size):
        # 根据颜色生成参数
        red_block_param = ColorDetect()
        red_block_param.color_name = colors[0]
        red_block_param.use_name = True
        red_block_param.detect_type = 'circle'
        red_block_param.roi.y_min = int(roi[0] * image_process_size[1])
        red_block_param.roi.y_max = int(roi[1] * image_process_size[1])
        red_block_param.roi.x_min = int(roi[2] * image_process_size[0])
        red_block_param.roi.x_max = int(roi[3] * image_process_size[0])
        red_block_param.image_process_size = image_process_size
        red_block_param.min_area = 10
        red_block_param.max_area = image_process_size[0]*image_process_size[1]

        green_block_param = copy.deepcopy(red_block_param)
        green_block_param.color_name = colors[1]
        
        blue_block_param = copy.deepcopy(red_block_param)
        blue_block_param.color_name = colors[2]

        return [red_block_param, green_block_param, blue_block_param]
    # 设置颜色服务回调函数
    def set_color_srv_callback(self, msg):
        # 设置颜色
        blocks_param = self.set_blocks_color(msg.data)
        self.detect_pub.publish([stairs_param])
        common.loginfo('%s set_color' % self.name)
        return [True, 'set_color']
    
    # 获取颜色识别结果回调函数
    def get_color_callback(self, msg):
        # 获取颜色识别结果
        self.objects_info = msg.data

    # 颜色块处理函数
    def color_sort_process(self, blocks_data):
        # 根据颜色播放动作
        if blocks_data is not None:
            if blocks_data.label == 'red':      # 处理红色色块
                self.count += 1
                if self.count > 10:
                    self.count = 0
                    common.loginfo('red')
                    # 执行放块动作组
                    self.motion_manager.run_action(self.right_hand_put_block_action_name)
            elif blocks_data.label == 'green':  # 处理绿色色块
                self.count += 1
                if self.count > 10:
                    self.count = 0
                    common.loginfo('green')
                    # 执行放块动作
                    self.motion_manager.set_servos_position(200, [[23, 400]])
                    rospy.sleep(0.2)
                    self.motion_manager.set_servos_position(300, [[23, 600]])
                    rospy.sleep(0.3)
                    self.motion_manager.set_servos_position(300, [[23, 400]])
                    rospy.sleep(0.3)
                    self.motion_manager.set_servos_position(200, [[23, 500]])
                    rospy.sleep(1.5)
            elif blocks_data.label == 'blue':     # 处理蓝色色块
                self.count += 1
                if self.count > 10:
                    self.count = 0
                    common.loginfo('blue')
                    self.motion_manager.run_action(self.left_hand_put_block_action_name)
        else:
            self.count = 0       # 识别丢失时计数清零

    def run(self):
        while self.running:
            if self.start:      # 根据状态机执行
                if self.state[self.current_state][2] == False:
                    self.state[self.current_state][2] = True
                    # 设置头部姿态
                    self.init_action(self.state[self.current_state][0][0], self.state[self.current_state][0][1])  # 头部姿态
                    param = self.state[self.current_state][1][3](self.state[self.current_state][1][0],
                                                                 self.state[self.current_state][1][1],
                                                                 self.state[self.current_state][1][2])
                    self.detect_pub.publish(param)  # 颜色检测设置

                # 获取识别结果
                blocks_data = None
                for object_info in self.objects_info:
                    if object_info.type == 'circle':
                        blocks_data = object_info
                # 调用处理函数
                if self.current_state == 'color_sort':
                    self.color_sort_process(blocks_data)
                
                rospy.sleep(0.01)
            else:
                rospy.sleep(0.01)
        # 当前状态完成后关闭节点
        self.init_action(self.head_pan_init, self.head_tilt_init)
        self.stop_srv_callback(None)
        rospy.signal_shutdown('shutdown')

if __name__ == "__main__":
    ColorSortNode('color_sort').run()
