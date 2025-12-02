#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/07/04
# @author:aiden
# 巡线低空穿越
import rospy
import signal
from std_msgs.msg import Float64, String
from ainex_sdk import misc, common
from ainex_example.color_common import Common
from ainex_example.visual_patrol import VisualPatrol
from ainex_interfaces.srv import SetString
from ainex_interfaces.msg import ObjectsInfo, ColorDetect, ROI

class CrawlUnderVisualPatrol(Common):
    # 按顺序检测三个roi，如果检测到黑线立刻跳出
    # y_min, y_max, x_min, x_max分别表示占图像的比例, 即实际大小为y_min*height
    line_roi = [(5 / 12, 6 / 12, 1 / 4, 3 / 4),
                (6 / 12, 7 / 12, 1 / 4, 3 / 4),
                (7 / 12, 8 / 12, 1 / 4, 3 / 4)
                ]

    # y_min, y_max, x_min, x_max分别表示占图像的比例
    door_roi = [1 / 5, 1, 0, 1]

    # 这里通过一个比例来设置阈值,表示当门的位置低于图像高度的240/480 = 0.5,
    # 即50%时,就认为门足够低,可以进入低空状态了。
    enter_crawl_under_y = 240 / 480

    # 图像处理时缩放到这个分辨率
    image_process_size = [160, 120]
    # 初始化节点
    def __init__(self, name):
        rospy.init_node(name)
        self.name = name
        self.running = True
        self.count_miss = 0
        self.objects_info = []
        self.head_raise_stamp = 0
        self.body_raise_stamp = 0
        self.head_state = 'drop'
        # 定义状态机中两个状态
        self.current_state = "visual_patrol"    # 巡线状态
        self.next_state = "crawl_under"         # 识别到门框后的状态
        
        # 定义状态机
        # 包括两个状态visual_patrol和crawl_under
        # 每个状态包含:
        #   - 头部位置[pan,tilt]
        #   - 颜色识别参数[颜色,ROI区域,图像大小,设置颜色的参数函数] 
        #   - 是否完成该状态的标志 
        self.state = {'visual_patrol':  [[500, 260], ['black', self.line_roi, self.image_process_size, self.set_visual_patrol_color], False],
                      'crawl_under':   [[500, 260], ['green', self.door_roi, self.image_process_size, self.set_door_color], False]}
       
        self.head_pan_init = self.state[self.current_state][0][0]   # 左右舵机的初始值
        self.head_tilt_init = self.state[self.current_state][0][1]  # 上下舵机的初始值
        super().__init__(name, self.head_pan_init, self.head_tilt_init)
        # 创建巡线类
        self.visual_patrol = VisualPatrol(self.gait_manager)
        self.visual_patrol.update_go_gait(dsp=[400, 0.2, 0.025], x_max=0.01)
        self.visual_patrol.update_turn_gait(dsp=[400, 0.2, 0.025], x_max=0.01)
        # Ctrl+C 退出
        signal.signal(signal.SIGINT, self.shutdown)

        # 订阅颜色识别结果
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)

        self.motion_manager.run_action('walk_ready')     # 进入站立状态

        if rospy.get_param('~start', True):
            self.enter_func(None)
            self.start_srv_callback(None)  # 开启颜色识别
            common.loginfo('start crawl_under')
    # Ctrl+C 回调函数
    def shutdown(self, signum, frame):
        self.running = False
        common.loginfo('%s shutdown' % self.name)
    # 设置巡线颜色的参数
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
    # 设置门颜色的参数
    def set_door_color(self, color, roi, image_process_size):
        # 设置门标志颜色
        door_param = ColorDetect()
        door_param.color_name = color
        door_param.detect_type = 'side'
        door_param.use_name = True
        door_param.image_process_size = image_process_size
        door_param.roi.y_min = int(roi[0] * image_process_size[1])
        door_param.roi.y_max = int(roi[1] * image_process_size[1])
        door_param.roi.x_min = int(roi[2] * image_process_size[0])
        door_param.roi.x_max = int(roi[3] * image_process_size[0])
        door_param.min_area = 10 * 20
        door_param.max_area = image_process_size[0] * image_process_size[1]

        return door_param

    # 接收到颜色识别结果的回调函数
    def get_color_callback(self, msg):
        # 获取颜色识别结果
        self.objects_info = msg.data
    # 状态初始化
    def state_init(self, current_state, next_state):
        # 如果是第一次运行当前状态,进行初始化
        if self.state[current_state][2] == False:
            self.state[current_state][2] = True
            # 头部位置还原
            self.init_action(self.state[current_state][0][0], self.state[current_state][0][1])  # 头部姿态
            param1 = self.state[current_state][1][3](self.state[current_state][1][0], self.state[current_state][1][1], self.state[current_state][1][2])
            param2 = self.state[next_state][1][3](self.state[next_state][1][0], self.state[next_state][1][1], self.state[next_state][1][2])
            self.detect_pub.publish([param1, param2])  # 颜色检测设置
    # 进入低空状态的判断
    def enter_crawl_under(self, door_data):
        if rospy.get_time() > self.head_raise_stamp:  # 不停抬头看门用来确定距离
            # 周期抬头动作
            if self.head_state == 'drop':
                self.motion_manager.set_servos_position(100, [[23, 500], [24, 500]])
                self.head_raise_stamp = rospy.get_time() + 1.5
                self.head_state = 'raise'   # 抬头
            else:
                self.motion_manager.set_servos_position(100, [[23, self.state[self.current_state][0][0]],
                                                              [24, self.state[self.current_state][0][1]]])
                self.head_raise_stamp = rospy.get_time() + 1.5
                self.head_state = 'drop'    # 放下头
         # 如果门的位置已经很低了
        if door_data is not None:
            # 这里door_data.height表示门的像素高度。所以判断条件就是:
            # 门的下边沿 < 阈值比例 * 门的像素高度
            # 当门低于其自身高度的一半时,就会进入低空状态。
            if max(door_data.y, door_data.left_point[1], door_data.right_point[1]) < self.enter_crawl_under_y * door_data.height:
                # 低姿态切换
                self.gait_manager.stop()
                self.gait_manager.set_body_height(0.06, 1)  # 设置身体高度为0.06m
                rospy.sleep(0.5)
                walking_param = self.gait_manager.get_gait_param()
                walking_param['body_height'] = 0.06         # 设置步态身体高度为0.06m
                # 更新正向步态参数
                self.visual_patrol.update_go_gait(walking_param=walking_param)
                # 更新转向步态参数
                self.visual_patrol.update_turn_gait(walking_param=walking_param)
                self.motion_manager.set_servos_position(100, [[23, self.state[self.current_state][0][0]],
                                                              [24, self.state[self.current_state][0][1]]])
                common.loginfo('crawl_under')
                self.head_state = 'drop'
                self.body_raise_stamp = rospy.get_time() + 7    # 设置7秒低空持续时间
                return True
        return False
    # 退出低空状态的判断
    def exit_crawl_under(self, line_data):
        if line_data is not None:
            if rospy.get_time() < self.body_raise_stamp:  # 低姿态持续行走的时间
                self.visual_patrol.process(line_data.x, line_data.width)
            else:
                # 切换正常巡线姿态
                self.gait_manager.stop()
                self.gait_manager.set_body_height(0.015, 1)
                rospy.sleep(0.5)
                walking_param = self.gait_manager.get_gait_param()
                walking_param['body_height'] = 0.015
                self.visual_patrol.update_go_gait(walking_param=walking_param)
                self.visual_patrol.update_turn_gait(walking_param=walking_param)
                return True
        return False

    def run(self):
        while self.running:
            if self.state:
                # 获取识别结果
                line_data = None
                door_data = None
                for object_info in self.objects_info:
                    if object_info.type == 'line':
                        if self.head_state == 'drop':
                            line_data = object_info
                    if object_info.type == 'side':
                        if self.head_state == 'raise':
                            door_data = object_info
                
                # 不同阶段处理
                if self.current_state == 'visual_patrol':
                    if line_data is not None:
                        self.visual_patrol.process(line_data.x, line_data.width)
                elif self.current_state == 'crawl_under':
                    if self.exit_crawl_under(line_data):
                        self.current_state = 'visual_patrol'
                        self.next_state = 'crawl_under'
                        self.state[self.current_state][2] = False
                 # 根据下一个状态进行处理 
                if self.next_state == 'crawl_under':
                    if self.enter_crawl_under(door_data):
                        self.current_state = 'crawl_under'
                        self.next_state = 'visual_patrol'
                        self.state[self.current_state][2] = False
                # 状态初始化
                self.state_init(self.current_state, self.next_state)

                rospy.sleep(0.01)
            else:
                rospy.sleep(0.01)
        
        self.init_action(self.head_pan_init, self.head_tilt_init)
        self.stop_srv_callback(None)        # 停止运行
        rospy.signal_shutdown('shutdown')

if __name__ == "__main__":
    CrawlUnderVisualPatrol('crawl_under_visual_patrol').run()
