#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/07/20
# @author:aiden
# 自主搬运
import math
import copy
import rospy
import signal
from std_srvs.srv import Empty, EmptyResponse   # 导入ROS Empty服务定义
from std_msgs.msg import Float64, String
from ainex_sdk import pid, misc, common
from ainex_example.color_common import Common
from ainex_example.pid_track import PIDTrack
from apriltag_ros.msg import AprilTagDetectionArray # 导入ROS Empty服务定义
from ainex_example.approach_object import ApproachObject    # 导入ApproachObject类
from ainex_interfaces.srv import SetString
from ainex_interfaces.msg import ObjectsInfo, ColorDetect, ColorsDetect  # 导入自定义消息类型

class AutonomousTransportNode(Common):
    block_x_stop = 320 / 640  # 当检测到的标识像素坐标x值占图像的比例在此值附近时停止前后移动(范围可在ApproachObject里设置)时停止前后移动
    block_y_stop = 220 / 480  # 当检测到的标识像素坐标y值占图像的比例在此值附近时停止横向移动(范围可在ApproachObject里设置)时停止横向移动
    block_yaw_stop = 0        # 当检测到的标识角度在此值附近时停止旋转移动(范围可在ApproachObject里设置)时停止旋转移动

    tag_y_stop = 240 / 480    # AprilTag在图像中的停止位置比例

    move_up_action_name = 'move_up'     # 定义的抬起动作的名称
    put_down_action_name = 'put_down'   # 定义的放下动作的名称

    image_process_size = [160, 120]     # 图像处理的分辨率大小
    
    # 搬运颜色对应的tag
    transport_dict = {'red': 1, 'green': 2, 'blue': 3}
    
    # 搬运顺序
    transport_index = ['red', 'green', 'blue']
    
    def __init__(self, name):
        rospy.init_node(name)       # ROS节点初始化
        self.name = name            # 节点名称
        self.running = True         # 运行状态
        self.count_miss = 0         # 丢失检测次数
        self.object_info = None     # 颜色识别结果
        self.start_index = 0        # 搬运顺序索引
        self.y_stop = self.block_y_stop # tag停止像素坐标y值占比
        self.tag_data = {'1': False, '2': False, '3': False}            # 检测到的tag数据
        self.remain_block_list = copy.deepcopy(self.transport_index)    # 剩余搬运块顺序 
        self.current_color = self.transport_index[0]                    # 当前搬运颜色
        self.body_track_state = 'approach'      # 躯体追踪状态
        self.head_track_state = 'track_block'   # 头部追踪状态
        self.transport_state = 'pick_block'     # 搬运状态
        self.start_find_block = False           # 开始找方块标志
        # 初始化头部位置
        self.head_pan_init = 500   # 左右舵机的初始值
        self.head_tilt_init = 280  # 上下舵机的初始值
        self.head_time_stamp = rospy.get_time() # 上一次头部运动时间
        super().__init__(name, self.head_pan_init, self.head_tilt_init)  # 初始化Common类
        self.calib_config = common.get_yaml_data('/home/ubuntu/ros_ws/src/ainex_example/config/calib.yaml')
        self.approach_object = ApproachObject(self.gait_manager)         # 初始化ApproachObject类
        
        # 更新ApproachObject的停止判断阈值
        self.approach_object.update_stop_count(1)                       # 停止判断的平稳次数阈值
        self.approach_object.update_approach_stop_value(10, 40, 7)      # 设置判断停止的误差阈值,分别对应x/y/yaw
        self.approach_object.update_gait_range(x_range=[-0.015, 0.015]) # 设置步态移动范围

        # 获取步态参数
        self.walking_param = self.gait_manager.get_gait_param()
        self.walking_param['step_height'] = 0.025    # 步高，单位：米
        self.walking_param['pelvis_offset'] = 3      # 盆骨偏移
        self.walking_param['hip_pitch_offset'] = 15  # 髋关节偏移
        self.walking_param['body_height'] = 0.035    # 身体高度
        # 更新ApproachObject的步态参数
        self.approach_object.update_gait(walking_param=self.walking_param)  

        
        self.motion_manager.run_action('walk_ready') # 运行站立动作
        signal.signal(signal.SIGINT, self.shutdown) # 注册SIGINT处理函数

        # 头部的pid追踪
        self.head_pan_range = [125, 875]   # 左右转动限制在这个范围， 125为右
        self.head_tilt_range = [250, 500]  # 上下限制在这个范围， 250为下
        self.rl_dis = None
        self.ud_dis = None
        self.pid_rl = pid.PID(0.1, 0.0, 0.001)
        self.pid_ud = pid.PID(0.1, 0.0, 0.001)
         # 初始化PID追踪类
        self.rl_track = PIDTrack(self.pid_rl, self.head_pan_range, self.head_pan_init)
        self.ud_track = PIDTrack(self.pid_ud, self.head_tilt_range, self.head_tilt_init)

        # 躯体的追踪参数
        self.yaw_stop = 0  # 躯体停止转动时头部左右舵机脉宽值和中位500的差值

        # 头部预设5个搜寻位置，左右舵机，上下舵机，时间ms
        # left_down, left_up, center_up, right_up, right_down
        self.head_move_position_block = [[650, 300, 1000],
                                         [650, 500, 1000],
                                         [500, 500, 1000],
                                         [350, 500, 1000],
                                         [350, 300, 1000]
                                         ]
        self.head_move_position_tag = [[550, 300, 1000],
                                       [550, 500, 1000],
                                       [500, 500, 1000],
                                       [450, 500, 1000],
                                       [450, 300, 1000]
                                       ]
        self.head_move_position = copy.deepcopy(self.head_move_position_block)
        
        rospy.ServiceProxy('/color_detection/start', Empty)()   # 启动颜色识别
        self.set_color_srv_callback(String(self.current_color)) # 设置识别蓝色
        rospy.ServiceProxy('/apriltag_ros/stop', Empty)()       # 停止apriltag识别

        # 订阅apriltag识别结果
        rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.tag_callback)
        # rospy.ServiceProxy('/color_detection/stop', Empty)()
        # 停止apriltag识别
        # rospy.ServiceProxy('/apriltag_ros/stop', Empty)()
        # rospy.ServiceProxy('/apriltag_ros/start', Empty)()
        # 订阅颜色识别结果
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)
        # 设置颜色服务回调函数
        rospy.Service('~set_color', SetString, self.set_color_srv_callback)  

        # 是否启动
        if rospy.get_param('~start', True):
            # 通知颜色识别准备，此时只显示摄像头原画
            self.enter_func(None)
            self.start_srv_callback(None)  # 开启颜色识别
            common.loginfo('start transport block')
    # 关闭节点
    def shutdown(self, signum, frame):
        with self.lock:
            self.running = False
            common.loginfo('%s shutdown' % self.name)

    # 设置颜色服务回调函数
    def set_color_srv_callback(self, msg):
        # 设置追踪颜色
        param = ColorDetect()
        param.color_name = msg.data
        param.detect_type = 'rect'
        param.use_name = True
        param.image_process_size = self.image_process_size
        param.min_area = 10
        param.max_area = 16 * self.image_process_size[0] * self.image_process_size[1]
        self.detect_pub.publish([param])
        common.loginfo('{} set_color {}'.format(self.name, msg.data))
        return [True, 'set_color']
    
    # 颜色识别结果回调函数
    def get_color_callback(self, msg):
        # 获取颜色识别结果
        for object_info in msg.data:
            if object_info.type == 'rect':
                self.object_info = [object_info.x, object_info.y, object_info.angle, object_info.width, object_info.height]
                if self.transport_state == 'pick_block' and self.head_track_state != 'stop' and not self.start_find_block:
                    self.rl_dis, self.ud_dis = self.head_track_process(object_info.x, object_info.y, object_info.width, object_info.height)
                elif self.head_track_state == 'stop':
                    self.rl_dis, self.ud_dis = None, None

    # apriltag识别结果回调函数
    def tag_callback(self, msg):
        if msg.detections != []:
            for i in msg.detections:
                if i.id[0] == self.transport_dict[self.current_color]:
                    roll, pitch, yaw = common.qua2rpy(i.pose.pose.pose.orientation)
                    x = int((i.corners[0].x + i.corners[2].x)/2)
                    y = int((i.corners[0].y + i.corners[2].y)/2)
                    self.object_info = [x, y, math.degrees(yaw), 640, 480]
                    if self.transport_state == 'place_block' and self.head_track_state != 'stop' and not self.start_find_block:
                        self.rl_dis, self.ud_dis = self.head_track_process(x, y, 640, 480)
                    elif self.head_track_state == 'stop':
                        self.rl_dis, self.ud_dis = None, None
                self.tag_data[str(i.id[0])] = True

    # 初始化搬运方块
    def find_block_init(self):
        self.y_stop = self.block_y_stop
        self.head_track_state = 'track_block'
        self.transport_state = 'pick_block'
        # 更新接近停止的判断阈值
        # 10:x方向位移误差阈值,单位是像素
        # 40:y方向位移误差阈值,单位是像素
        # 7:航向角误差阈值,单位是度
        self.approach_object.update_approach_stop_value(10, 40, 7)
        self.walking_param['hip_pitch_offset'] = 15  # 髋关节角度偏移量

        # 更新步态参数
        # 步长,步高,步速
        # 行走稳定步数
        # 步态参数
        self.approach_object.update_gait([400, 0.2, 0.025], 30, self.walking_param)
        # 复制block的头部位置列表
        self.head_move_position = copy.deepcopy(self.head_move_position_block)
        # 从剩余列表中移除当前颜色
        self.remain_block_list.remove(self.current_color)
        if self.remain_block_list == []:
            self.running = False
            # self.remain_block_list = copy.deepcopy(self.transport_index)
        else:
            self.current_color = self.remain_block_list[0]
        rospy.ServiceProxy('/color_detection/start', Empty)()
        self.set_color_srv_callback(String(self.current_color))  # 设置识别蓝色
        rospy.ServiceProxy('/apriltag_ros/stop', Empty)()

    # 初始化放置方块
    def find_tag_init(self):
        self.y_stop = self.tag_y_stop
        self.head_track_state = 'track_tag'
        self.transport_state = 'place_block'
        # 更新接近停止的判断阈值，同上
        # x方向位移误差阈值(像素)
        # y方向位移误差阈值(像素)  
        # 航向角误差阈值(度)
        self.approach_object.update_approach_stop_value(10, 10, 5)
        self.walking_param['hip_pitch_offset'] = 5  # 髋关节角度偏移量
        # 更新步态参数
        self.approach_object.update_gait([400, 0.1, 0.03], 0, self.walking_param)
        self.head_move_position = copy.deepcopy(self.head_move_position_tag)
        rospy.ServiceProxy('/color_detection/stop', Empty)()
        rospy.ServiceProxy('/apriltag_ros/start', Empty)()

    # 通过其他apriltag判断目标apriltag位置
    # apriltag摆放位置：红(tag36h11_1)，绿(tag36h11_2)，蓝(tag36h11_3)
    def get_direction(self, target_tag, tag_data):
        # 目标tag， 当前检测到的tag
        if target_tag == 1:  # 目标apriltag为1
            if not tag_data['2']:  # 没有检测到apriltag 2
                if tag_data['3']:  # 检测到apriltag 3， 则apriltag 1在apriltag 3左边，所以左转
                    return 'left'
            else:                  # 检测到apriltag 2，则则apriltag 1在apriltag 2左边，所以左转
                return 'left'
        elif target_tag == 2:
            if not tag_data['1']:
                if tag_data['3']:
                    return 'left'
            else:
                return 'right'
        elif target_tag == 3:
            if not tag_data['1']:
                if tag_data['2']:
                    return 'right'
            else:
                return 'right'

        return None
    
    # 头部PID跟踪
    def head_track_process(self, x, y, width, height):
        # 头部追踪
        if abs(x - width/2) < 10:
            x = width/2
        if abs(y - height/2) < 10:
            y = height/2
        rl_dis = self.rl_track.track(x, width/2)
        ud_dis = self.ud_track.track(y, height/2)
        self.motion_manager.set_servos_position(20, [[23, int(rl_dis)], [24, int(ud_dis)]])

        return rl_dis, ud_dis
    
    # 躯体跟踪
    def body_track_process(self, rl_dis, ud_dis, object_data):
        self.object_info = None # 清空上一次的检测信息
        # 躯体追踪
        x, y, angle, width, height = object_data    # 获取检测到的目标信息
        self.rl_dis, self.ud_dis = None, None       # 清空头部pid计算的输出
        if object_data is not None:
            # approach状态下进行跟踪
            if self.body_track_state == 'approach' and rl_dis is not None:
                self.approach_object.update_gait(step_mode=0)   # 更新步态为前进模式
                # 更新步态移动范围
                self.approach_object.update_gait_range(y_range=[-0.01, 0.01], yaw_range=[-6, 6])
                # 计算旋转停止的目标位置
                yaw_stop = 500 + self.calib_config['head_pan_offset'] + math.copysign(self.yaw_stop, rl_dis - (500 + self.calib_config['head_pan_offset']))
                # 进行跟踪,如果接近目标则切换状态
                if self.approach_object.process(500 - ud_dis, 0 + self.calib_config['center_x_offset'], (yaw_stop - rl_dis)/9, self.head_tilt_range[0] - 40, 0, 0, width, height):
                    self.head_track_state = 'stop'
                    self.body_track_state = 'align'
                    rospy.sleep(0.5)
                    # 回正头部位置
                    self.motion_manager.set_servos_position(200, [[23, self.head_pan_init], [24, self.head_tilt_init]])
                    rospy.sleep(0.5)
                    print('align')
            # align状态下进行旋转归中
            elif self.body_track_state == 'align':
                if self.transport_state == 'pick_block':
                    if angle > 40:  # 调整角度，不取45，因为如果在45时值的不稳定会导致反复移动
                        angle -= 90
                elif self.transport_state == 'place_block':
                    self.walking_param['hip_pitch_offset'] = 5      # 更新步态参数
                    self.approach_object.update_gait([400, 0.1, 0.03], 0, self.walking_param)
                    if angle < 0:   # 调整角度
                        angle += 360
                    angle -= 180
                self.approach_object.update_gait(step_mode=1)   # 更新步态为旋转模式
                # 更新步态移动范围
                self.approach_object.update_gait_range(y_range=[-0.012, 0.012], yaw_range=[-4, 4])    
                # 进行跟踪,如果旋转归中则切换状态
                if self.approach_object.process(y, x, angle, self.y_stop*width, self.block_x_stop*width, self.block_yaw_stop, width, height):
                    if self.transport_state == 'pick_block':     # 提起对象
                        self.gait_manager.disable()
                        self.motion_manager.run_action(self.move_up_action_name)
                    elif self.transport_state == 'place_block':  # 放下对象
                        self.walking_param['hip_pitch_offset'] = 5
                        self.gait_manager.set_step([400, 0.2, 0.025], 0.01, 0, 0, self.walking_param, 0, 2) 
                        self.gait_manager.disable()
                        self.motion_manager.run_action(self.put_down_action_name)
                        self.walking_param['hip_pitch_offset'] = 15
                        self.gait_manager.set_step([400, 0.2, 0.025], -0.01, 0, 0, self.walking_param, 0, 7)
                    self.body_track_state = 'approach'
                    return True
                else:
                    rospy.sleep(0.8)
        return False

    # 搜索过程
    def find_process(self):
        if rospy.get_time() > self.head_time_stamp:
            if self.start_index > len(self.head_move_position) - 1:
                self.start_index = 0
            rl_dis = self.head_move_position[self.start_index][0]
            ud_dis = self.head_move_position[self.start_index][1]
            self.rl_track.update_position(rl_dis)  # pid的输出值要跟着更新
            self.ud_track.update_position(ud_dis)
            self.motion_manager.set_servos_position(self.head_move_position[self.start_index][2], [[23, rl_dis], [24, ud_dis]])
            if self.transport_state == 'place_block':
                self.walking_param['hip_pitch_offset'] = 5
                if self.get_direction(self.transport_dict[self.current_color], self.tag_data) == 'left':
                    self.gait_manager.set_step([400, 0.2, 0.025], 0, 0, 5, self.walking_param, 0, 0) 
                else:
                    self.gait_manager.set_step([400, 0.2, 0.025], 0, 0, -5, self.walking_param, 0, 0)   # 右转
                self.tag_data = {'1': False, '2': False, '3': False}
            else:
                self.walking_param['hip_pitch_offset'] = 15
                self.gait_manager.set_step([400, 0.2, 0.025], 0, 0, -5, self.walking_param, 0, 0)
            self.head_time_stamp = rospy.get_time() + self.head_move_position[self.start_index][2]/1000.0
            self.start_index += 1
    
    # 主循环
    def run(self):
        while self.running:
            if self.start:
                if self.object_info is not None:    # 如果检测到目标
                    if self.start_find_block:       # 需要搜寻的标志置为False
                        self.start_find_block = False
                    # 进行跟踪 
                    elif self.body_track_process(self.rl_dis, self.ud_dis, self.object_info):
                        # 根据状态进行初始化
                        if self.transport_state == 'pick_block':    # 代表着去搬运方块的状态
                            self.find_tag_init()                            
                        elif self.transport_state == 'place_block': # 代表着去放置区放置方块的状态
                            self.find_block_init()
                    self.count_miss = 0  # 重置丢失次数
                else:                    # 如果未检测到目标
                    if not self.start_find_block:
                        self.count_miss += 1        # 记录丢失次数
                        if self.count_miss > 100:   # 如果超过阈值则置标志位开始搜寻
                            self.count_miss = 0
                            self.start_find_block = True
                            self.start_index = 0
                    else:
                        self.find_process()         # 搜寻过程
                rospy.sleep(0.01)
            else:
                rospy.sleep(0.01)

        self.init_action(self.head_pan_init, self.head_tilt_init)   # 停止头部动作
        self.stop_srv_callback(None)        # 停止检测
        rospy.signal_shutdown('shutdown')   # 关闭节点

if __name__ == "__main__":
    AutonomousTransportNode('autonomous_transport').run()
