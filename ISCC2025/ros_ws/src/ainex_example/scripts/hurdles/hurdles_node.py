#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/07/04
# @author:aiden
# 跨栏
import rospy
import signal
from ainex_sdk import common
from ainex_example.color_common import Common
from ainex_example.approach_object import ApproachObject
from ainex_interfaces.srv import SetString
from ainex_interfaces.msg import ObjectsInfo, ColorDetect

class HurdlesNode(Common):
    #  定义跨栏颜色识别的ROI区域
    hurdles_roi = [1 / 5, 1, 0, 1]
    #  定义跨栏动作的名称
    hurdles_action_name = 'hurdles'
    # 定义颜色识别的图像处理分辨率
    image_process_size = [160, 120]
    # 定义跨栏时的机器人停止条件
    hurdles_x_stop = 395/480
    hurdles_y_stop = 0.5
    hurdles_yaw_stop = 0

    def __init__(self, name):
        # 初始化节点
        rospy.init_node(name)
        self.name = name
        self.running = True
        self.objects_info = []
        self.current_state = 'hurdles'
        # 定义状态机,包括初始头部姿态,颜色识别参数,是否完成
        self.state = {'hurdles': [[500, 260], ['blue', self.hurdles_roi, self.image_process_size, self.set_hurdles_color], False]}
        # 获取状态机中的初始头部姿态
        self.head_pan_init = self.state[self.current_state][0][0]   # 左右舵机的初始值
        self.head_tilt_init = self.state[self.current_state][0][1]  # 上下舵机的初始值
        # 初始化父类,设置初始头部姿态 
        super().__init__(name, self.head_pan_init, self.head_tilt_init)
        # 创建接近对象实例
        self.approach_object = ApproachObject(self.gait_manager)
        # 更新ApproachObject对象的接近停止条件
        # x_approach_value: 接近目标中心x轴距离的阈值,单位厘米
        # yaw_approach_value: 接近目标航向角误差的阈值,单位度
        self.approach_object.update_approach_stop_value(x_approach_value=35, yaw_approach_value=5)
        
        signal.signal(signal.SIGINT, self.shutdown) # 设置退出信号处理函数

        # 订阅颜色识别结果
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)
        rospy.Service('~set_color', SetString, self.set_color_srv_callback)  # 设置颜色
        self.motion_manager.run_action('hand_back')  # 播放收回手臂动作
        if rospy.get_param('~start', True):
            self.enter_func(None)
            self.start_srv_callback(None)  # 开启颜色识别
            common.loginfo('start hurdles')
    # 关闭节点回调函数
    def shutdown(self, signum, frame):
        self.running = False
        common.loginfo('%s shutdown' % self.name)

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
        # 设置颜色服务回调函数
         # 生成颜色识别参数
        hurdles_param = self.set_hurdles_color(msg.data)
        # 发布设置的颜色识别参数
        self.detect_pub.publish([hurdles_param])
        common.loginfo('%s set_color' % self.name)

        return [True, 'set_color']

    def get_color_callback(self, msg):
        # 获取颜色识别结果
        self.objects_info = msg.data

    def hurdles_process(self, hurdles_data):
        # 跨栏处理函数
        
        # 调用接近对象的过程函数,控制机器人移动
        if self.approach_object.process(max(hurdles_data.y, hurdles_data.left_point[1], hurdles_data.right_point[1]), hurdles_data.x, hurdles_data.angle, 
                                   self.hurdles_x_stop*hurdles_data.height, self.hurdles_y_stop*hurdles_data.width, self.hurdles_yaw_stop, hurdles_data.width, hurdles_data.height):
            self.gait_manager.disable()  # 如果接近成功,停止运动,播放动作
            common.loginfo('hurdles')
            self.motion_manager.run_action(self.hurdles_action_name)
            return True
        return False

    def run(self):
        while self.running:
            if self.start:  # 如果打开start开关
                if self.state[self.current_state][2] == False:  # 如果当前状态尚未完成
                    self.state[self.current_state][2] = True
                    self.init_action(self.state[self.current_state][0][0], self.state[self.current_state][0][1])  # 头部姿态
                    # 生成颜色识别参数
                    param = self.state[self.current_state][1][3](self.state[self.current_state][1][0],
                                                                 self.state[self.current_state][1][1],
                                                                 self.state[self.current_state][1][2])
                    self.detect_pub.publish([param])  # 发布颜色识别参数  

                # 获取识别结果
                hurdles_data = None
                for object_info in self.objects_info:
                    if object_info.type == 'side':
                        hurdles_data = object_info
                # 如果有识别结果,并且是在跨栏状态
                if hurdles_data is not None and self.current_state == 'hurdles':
                    if self.hurdles_process(hurdles_data):   # 调用跨栏处理函数
                        self.running = False
                    else:
                        rospy.sleep(0.8)
                rospy.sleep(0.01)
            else:
                rospy.sleep(0.01)
        # 退出前动作
        self.init_action(self.head_pan_init, self.head_tilt_init)
        self.stop_srv_callback(None)
        rospy.signal_shutdown('shutdown')

if __name__ == "__main__":
    HurdlesNode('hurdles').run()
