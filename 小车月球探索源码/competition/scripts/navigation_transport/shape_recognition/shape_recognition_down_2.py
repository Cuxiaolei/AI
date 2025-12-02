#!/usr/bin/python3
# coding=utf8

# 通过深度图识别物体的外形进行分类
# 机械臂向下识别
# 可以识别长方体，球，圆柱体

from sklearn.linear_model import LinearRegression  # 此必须放置于第一行
import cv2
import os
import tone
import math
import queue
import rospy
import threading
import numpy as np
import sdk.common as common
import message_filters
import transforms3d as tfs
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import CameraInfo
from std_srvs.srv import SetBool, Trigger, TriggerResponse
from servo_msgs.msg import MultiRawIdPosDur
from interfaces.srv import GetRobotPose
from sdk import pid, fps
from servo_controllers.bus_servo_control import set_servos
from kinematics import kinematics_control


def xyz_quat_to_mat(xyz, quat):
    mat = tfs.quaternions.quat2mat(np.asarray(quat))
    mat = tfs.affines.compose(np.squeeze(np.asarray(xyz)), mat, [1, 1, 1])
    return mat


def xyz_euler_to_mat(xyz, euler, degrees=True):
    if degrees:
        mat = tfs.euler.euler2mat(math.radians(euler[0]), math.radians(euler[1]), math.radians(euler[2]))
    else:
        mat = tfs.euler.euler2mat(euler[0], euler[1], euler[2])
    mat = tfs.affines.compose(np.squeeze(np.asarray(xyz)), mat, [1, 1, 1])
    return mat


def mat_to_xyz_euler(mat, degrees=True):
    t, r, _, _ = tfs.affines.decompose(mat)
    if degrees:
        euler = np.degrees(tfs.euler.mat2euler(r))
    else:
        euler = tfs.euler.mat2euler(r)
    return t, euler


def depth_pixel_to_camera(pixel_coords, depth, intrinsics):
    fx, fy, cx, cy = intrinsics
    px, py = pixel_coords
    x = (px - cx) * depth / fx
    y = (py - cy) * depth / fy
    z = depth
    return np.array([x, y, z])


class RgbDepthImageNode:
    def __init__(self):
        rospy.init_node('shape_recognition', anonymous=True)
        self.fps = fps.FPS()
        self.last_shape = "none"  # 上一个形状
        self.rgb_sub = None  # rbg话题
        self.depth_sub = None  # 深度话题
        self.info_sub = None  # 相机内参话题
        self.sync = None  # 时间同步器名
        self.moving = False  # 是否开始夹取
        self.count = 0
        self.close = False  # 是否关闭节点
        self.endpoint = None  # 末端坐标
        self.shape = None  # 形状
        self.calibration_flat = False
        self.calibration_dist = False
        self.pick_state = False  # 是否开始检测
        offset = rospy.get_param('/offset')  # 识别的类别
        self.shape_dist = rospy.get_param('/shape_dist')  # 识别的类别
        self.offset_x = offset[0]
        self.offset_y = offset[1]
        self.offset_z = offset[2]

        self.target_shape = "None"  # 目标形状
        self.queue = queue.Queue(maxsize=1)  # 图像队列
        rospy.set_param('~status', 'start')  # 设置目前节点状态
        self.hand2cam_tf_matrix = [[0.0, 0.0, 1.0, -0.105],
                                   [-1.0, 0.0, 0.0, 0.0],
                                   [0.0, -1.0, 0.0, 0.044],
                                   [0.0, 0.0, 0.0, 1.0]]

        self.search_state = 0  # 0:未搜索 1:已左转 2:已右转
        self.original_pose = None  # 初始位姿
        self.search_attempts = 0  # 搜索尝试次数
        self.max_attempts = 20  # 最大尝试次数（可根据帧率调整）
        self.search_delay = 0  # 搜索延迟计数器
        self.search_delay_max = 10  # 延迟帧数（确保图像稳定）
        # 新增：记录1号关节累计旋转角度（单位：度），用于角度补偿
        self.joint1_rotation = 0.0  # 初始旋转角度为0


        # 舵机控制
        # self.servos_pub = rospy.Publisher('/robot_1/servo_controllers/port_id_1/multi_id_pos_dur', MultiRawIdPosDur, queue_size=1)
        self.servos_pub = rospy.Publisher('/servo_controllers/port_id_1/multi_id_pos_dur', MultiRawIdPosDur,
                                          queue_size=1)
        rospy.sleep(3)
        # 初始化目标形状参数，方便设置
        rospy.set_param('~target_shape', 'box')
        rospy.Service('~pick', Trigger, self.pick_callback)  # 进行夹取
        rospy.Service('~calibration_flat', Trigger, self.calibration_flat_callback)  # 进行夹取
        rospy.Service('~calibration_dist', Trigger, self.calibration_dist_callback)  # 进行夹取
        rospy.Service('~stop', Trigger, self.stop_callback)  # 停止节点
        rospy.Service('~start', Trigger, self.start_callback)  # 启动形状识别
        rospy.Service('~colse', Trigger, self.colse_callback)  # 关闭节点
        rospy.sleep(2)
        # rospy.wait_for_service('/robot_1/gemini_camera/set_ldp') #等待相机ldp服务
        rospy.wait_for_service('/gemini_camera/set_ldp')  # 等待相机ldp服务
        # rospy.ServiceProxy('/robot_1/gemini_camera/set_ldp', SetBool)(False) #关闭相机ldp服务器，近距离也能进行识别
        rospy.ServiceProxy('/gemini_camera/set_ldp', SetBool)(False)  # 关闭相机ldp服务器，近距离也能进行识别
        # 由于相机是斜着看地面的，使用线性回归校准数据成平面
        self.line_compensation = LinearRegression()
        # 此处参数皆为测试所得
        shape_flat = rospy.get_param('/shape_flat')

        self.line_compensation.fit([[20], [200], [350]], [[shape_flat[0]], [1], [shape_flat[1]]])
        self.line_depth_compensation = []

        # 由于相机只是再y轴翻转，使用这里只需要校准y轴 
        for i in range(399):
            self.line_depth_compensation.append(self.line_compensation.predict([[i]]))

    # 启动形状识别
    def start_callback(self, msg):
        threading.Thread(target=self.goto_default, args=()).start()  # 得到相机目前的末端位置
        self.rgb_sub = message_filters.Subscriber('/gemini_camera/rgb/image_raw', RosImage, queue_size=1)  # rgb话题
        # self.rgb_sub = message_filters.Subscriber('/robot_1/gemini_camera/rgb/image_raw', RosImage, queue_size=1) # rgb话题
        self.depth_sub = message_filters.Subscriber('/gemini_camera/depth/image_raw', RosImage, queue_size=1)  # 深度话题
        # self.depth_sub = message_filters.Subscriber('/robot_1/gemini_camera/depth/image_raw', RosImage, queue_size=1) # 深度话题
        self.info_sub = message_filters.Subscriber('/gemini_camera/depth/camera_info', CameraInfo,
                                                   queue_size=1)  # 相机内参话题
        # self.info_sub = message_filters.Subscriber('/robot_1/gemini_camera/depth/camera_info', CameraInfo, queue_size=1) # 相机内参话题
        # 同步时间戳, 时间允许有误差在0.03s
        self.sync = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub, self.info_sub], 3, 0.03)
        self.sync.registerCallback(self.multi_callback)  # 执行反馈函数

        # 新增：保存初始位姿
        while self.endpoint is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        self.original_pose = self.endpoint.copy()


        return TriggerResponse(success=True)

    # 停止形状识别
    def stop_callback(self, msg):
        self.rgb_sub.unregister()  # 关闭话题
        self.depth_sub.unregister()
        self.info_sub.unregister()
        return TriggerResponse(success=True)

    # 关闭节点
    def colse_callback(self, msg):
        self.close = True
        return TriggerResponse(success=True)

    # 进行夹取
    def pick_callback(self, msg):
        # 设置目标形状
        self.target_shape = rospy.get_param('/shape_recognition/target_shape', "box")
        # 进入机械臂进入夹取状态
        set_servos(self.servos_pub, 1, ((1, 500), (2, 500), (3, 150), (4, 130), (5, 500), (10, 200)))
        rospy.sleep(2)
        self.pick_state = True
        rospy.set_param('~pick', True)



        self.pick_state = True
        self.search_state = 0  # 重置搜索状态
        self.search_attempts = 0
        self.search_delay = 0
        self.original_pose = self.endpoint.copy()  # 重新记录初始位姿
        rospy.set_param('~pick', True)
        return TriggerResponse(success=True)

    def calibration_dist_callback(self, msg):
        # 设置目标形状
        self.target_shape = rospy.get_param('/shape_recognition/target_shape', "None")
        # 进入机械臂进入夹取状态
        set_servos(self.servos_pub, 1, ((1, 500), (2, 500), (3, 150), (4, 130), (5, 500), (10, 200)))
        rospy.sleep(2)
        self.calibration_dist = True
        rospy.set_param('~calibration', True)
        return TriggerResponse(success=True)

    def calibration_flat_callback(self, msg):
        # 设置目标形状
        self.target_shape = rospy.get_param('/shape_recognition/target_shape', "box")
        # 进入机械臂进入夹取状态
        set_servos(self.servos_pub, 1, ((1, 500), (2, 500), (3, 150), (4, 130), (5, 500), (10, 200)))
        rospy.sleep(2)
        self.calibration_flat = True
        rospy.set_param('~calibration', True)
        return TriggerResponse(success=True)

    # 得到机械臂末端坐标
    def goto_default(self):
        while not rospy.is_shutdown():
            endpoint = rospy.ServiceProxy('/kinematics/get_current_pose', GetRobotPose)()
            # print(endpoint)
            pose_t = endpoint.pose.position
            pose_r = endpoint.pose.orientation
            self.endpoint = xyz_quat_to_mat([pose_t.x, pose_t.y, pose_t.z], [pose_r.w, pose_r.x, pose_r.y, pose_r.z])

    # 夹取函数
    def move(self, shape, pose_t, angle):
        rospy.sleep(0.5)
        # 第一步：移动到目标上方安全位置
        pose_t[2] += 0.02
        ret1 = kinematics_control.set_pose_target(pose_t, 85)  # 根据逆运动学得到舵机脉宽
        if len(ret1[1]) > 0:
            set_servos(self.servos_pub, 1.5,
                       ((1, ret1[1][0]), (2, ret1[1][1]), (3, ret1[1][2]), (4, ret1[1][3]), (5, ret1[1][4])))
            rospy.sleep(1.5)

        # 第二步：下降到抓取位置
        pose_t[2] -= 0.05
        ret2 = kinematics_control.set_pose_target(pose_t, 85)

        # 关键修改：处理补偿后的角度（适配夹子关节）
        if angle != 0 and len(ret2[1]) > 0:
            # 1. 标准化角度到[-90, 90]度范围（避免超过夹子旋转极限）
            angle = angle % 180
            angle = angle - 180 if angle > 90 else (angle + 180 if angle < -90 else angle)

            # 2. 将角度转换为夹子关节（5号关节）的PWM值
            # 说明：5号关节的PWM范围通常为200-800，对应-120度到+120度
            # 这里根据机械臂实际参数调整映射关系
            pwm_center = 500  # 中间位置（0度）
            angle_range = 120  # 最大旋转角度（±120度）
            pwm_range = 300  # 对应最大角度的PWM变化量（500±300）
            target_pwm = pwm_center + int(angle * pwm_range / angle_range)

            # 3. 确保PWM值在安全范围内
            target_pwm = max(200, min(800, target_pwm))
        else:
            target_pwm = 500  # 默认中间位置

        # 第三步：调整夹子角度并闭合
        if len(ret2[1]) > 0:
            # 先调整角度
            set_servos(self.servos_pub, 0.5, ((5, target_pwm),))
            rospy.sleep(0.5)
            # 移动到抓取位置
            set_servos(self.servos_pub, 1,
                       ((1, ret2[1][0]), (2, ret2[1][1]), (3, ret2[1][2]), (4, ret2[1][3]), (5, target_pwm)))
            rospy.sleep(1)
            # 闭合夹子（10号关节PWM值750为夹紧）
            set_servos(self.servos_pub, 0.6, ((10, 750),))
            rospy.sleep(0.6)

        # 第四步：提升并复位
        if len(ret1[1]) > 0:
            set_servos(self.servos_pub, 1,
                       ((1, ret1[1][0]), (2, ret1[1][1]), (3, ret1[1][2]), (4, ret1[1][3]), (5, target_pwm)))
            rospy.sleep(1)
        # 回到初始位置
        set_servos(self.servos_pub, 1, ((1, 500), (2, 720), (3, 100), (4, 150), (5, 500), (10, 650)))
        rospy.sleep(1)

        # 重置状态
        rospy.set_param('~status', 'stop')
        self.pick_state = False
        self.moving = False

    # 时间同步回调函数
    def multi_callback(self, ros_rgb_image, ros_depth_image, depth_camera_info):
        if self.queue.empty():
            self.queue.put_nowait((ros_rgb_image, ros_depth_image, depth_camera_info))
            self.image_proc()
        # 判断是否需要关闭节点
        if self.close:
            rospy.signal_shutdown('shutdown')

    def rotate_arm(self, angle):
        """旋转1号关节指定角度（单位：度）"""
        try:
            # 获取当前舵机位置
            current_positions = [500, 500, 150, 130, 500, 200]  # 默认位置

            # 计算目标角度对应的PWM值 (假设-90度到+90度对应200到800)
            pwm_center = 500  # 中间位置
            pwm_range = 300  # 对应90度的PWM范围
            target_pwm = pwm_center + int(angle * pwm_range / 90)

            # 确保PWM值在有效范围内 (0-1000)
            target_pwm = max(200, min(800, target_pwm))

            # 发送舵机控制命令
            set_servos(self.servos_pub, 1, ((1, target_pwm), (2, 500), (3, 150), (4, 130), (5, 500), (10, 200)))
            rospy.sleep(1.5)  # 等待机械臂移动到位
            rospy.sleep(0.5)  # 额外延迟，确保相机稳定

            # 关键修改：记录累计旋转角度（用于后续角度补偿）
            self.joint1_rotation += angle


            return True
        except Exception as e:
            rospy.logerr(f"旋转关节失败: {str(e)}")
            return False


    # 开始检测
    def image_proc(self):
        try:
            # 得到图像信息
            ros_rgb_image, ros_depth_image, depth_camera_info = self.queue.get(block=True)
            # print("11")
            # print(self.calibration)
            if self.pick_state:
                # 转换图像格式
                rgb_image = np.ndarray(shape=(ros_rgb_image.height, ros_rgb_image.width, 3), dtype=np.uint8,
                                       buffer=ros_rgb_image.data)
                depth_image = np.ndarray(shape=(ros_depth_image.height, ros_depth_image.width), dtype=np.uint16,
                                         buffer=ros_depth_image.data)

                ih, iw = depth_image.shape[:2]

                depth_image = depth_image.copy()
                for j in range(399):
                    depth_image[j] = depth_image[j] * self.line_depth_compensation[j]
                # 屏蔽掉一些区域，降低识别条件，使识别跟可靠
                depth_image[:, 0:50] = np.array([[1000, ] * 50] * 400)
                depth_image[:, 590:640] = np.array([[1000, ] * 50] * 400)
                depth_image[320:400, :] = np.array([[1000, ] * 640] * 80)
                # depth_image[0:30, :] = np.array([[1000,]*640]* 30)
                depth = np.copy(depth_image).reshape((-1,))
                depth[depth <= 0] = 55555  # 距离为0可能是进入死区，或者颜色问题识别不到，将距离赋一个大值
                min_index = np.argmin(depth)  # 距离最小的像素
                min_y = min_index // iw
                min_x = min_index - min_y * iw

                min_dist = depth_image[min_y, min_x]  # 获取最小距离值
                # print(min_dist)
                sim_depth_image = np.clip(depth_image, 0, 300).astype(np.float64) / 300 * 255
                depth_image = np.where(depth_image > min_dist + 17, 0, depth_image)  # 将深度值大于最小距离15mm的像素置0
                sim_depth_image_sort = np.clip(depth_image, 0, 2000).astype(np.float64) / 2000 * 255
                depth_gray = sim_depth_image_sort.astype(np.uint8)
                depth_gray = cv2.GaussianBlur(depth_gray, (3, 3), 0)
                _, depth_bit = cv2.threshold(depth_gray, 1, 255, cv2.THRESH_BINARY)
                depth_bit = cv2.erode(depth_bit, np.ones((3, 3), np.uint8))
                depth_bit = cv2.dilate(depth_bit, np.ones((3, 3), np.uint8))
                # depth_color_map = cv2.applyColorMap(sim_depth_image.astype(np.uint8), cv2.COLORMAP_JET)

                contours, hierarchy = cv2.findContours(depth_bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                shape = 'None'
                contour = None
                for obj in contours:
                    if min_dist > self.shape_dist - 2:  # 最小距离不能大于220
                        break
                    area = cv2.contourArea(obj)
                    # print(area)
                    if area < 3000 or area > 15000 or self.moving is True:
                        continue
                    # cv2.drawContours(depth_color_map, obj, -1, (255, 255, 0), 4) # 绘制轮廓线
                    perimeter = cv2.arcLength(obj, True)  # 计算轮廓周长
                    approx = cv2.approxPolyDP(obj, 0.04 * perimeter, True)  # 获取轮廓角点坐标
                    # cv2.drawContours(depth_color_map, approx, -1, (255, 0, 0), 4) # 绘制轮廓线
                    CornerNum = len(approx)

                    x, y, w, h = cv2.boundingRect(approx)
                    x_contour_depth = depth_image[y + int(h / 2) - 2: y + int(h / 2) + 2, x: x + w]  # 获取轮廓中心一整行的深度数据
                    y_contour_depth = depth_image[y: y + h, x + int(w / 2) - 2: x + int(w / 2) + 2]  # 获取轮廓中心一整行的深度数据
                    # 我们根据X轴像素的标准差来判断是曲面类物体(圆柱体/球体)还是多面体(立方体、长方体)
                    # 曲面的标准差较大
                    # 因摄像头与桌面有一定角度， 所以Y轴的标准差在平面上也会比较大,不好判断. X轴不存在这个问题
                    x_depth = np.where(x_contour_depth == 0, np.nan, x_contour_depth)  # 计算这行数据中有效数据的标准差
                    y_depth = np.where(y_contour_depth == 0, np.nan, y_contour_depth)  # 计算这行数据中有效数据的标准差
                    x_depth_std = np.nanstd(x_depth)
                    y_depth_std = np.nanstd(y_depth)
                    print(w, h)
                    print(abs(w / h), abs(h / w))
                    print(x_depth_std, y_depth_std, CornerNum)
                    # print(x_depth_std - y_depth_std)
                    if x_depth_std <= 0.9 and y_depth_std <= 1.15 and CornerNum == 4:
                        objType = "cuboid_1"  # 立方体/长方体
                    else:
                        # print(CornerNum)
                        if abs(w / h) > 1.7 or abs(h / w) > 1.7 and CornerNum == 4:
                            if x_depth_std <= 0.9 and y_depth_std <= 1.15:
                                objType = "cuboid_2"  # 立方体/长方体
                            else:
                                if w > 120 or h > 120:
                                    objType = "cuboid_3"  # 立方体/长方体
                                else:
                                    objType = "cylinder_2"  # 圆柱体
                                    self.shape = 'cylinder'
                        else:
                            if abs(x_depth_std - y_depth_std) > 0.5:
                                if x_depth_std <= 0.9 and y_depth_std <= 1.15:
                                    objType = "cuboid_4"  # 立方体/长方体
                                else:
                                    if w > 120 or h > 120:
                                        objType = "cuboid_5"  # 立方体/长方体
                                    else:
                                        objType = "cylinder_3"  # 圆柱体
                                        self.shape = 'cylinder'
                            else:
                                if w > 120 or h > 120:
                                    objType = "cuboid_6"  # 立方体/长方体
                                    self.shape = 'cylinder'
                                else:
                                    objType = "cylinder_4"  # 圆柱体
                                    self.shape = 'cylinder'
                    shape = objType
                    print(shape)
                    contour = obj
                    if 'cubo' in objType:
                        # 判断是正方体还是长方体
                        # 这里取巧，我们的木块正方体的高度都不超过4.0直接判断一下
                        # 不取巧的方法就是通过最小外接矩形的四个角点及中心点像素坐标
                        # 调用 depth_pixel_to_camera获取各点在空间中的位置
                        # 然后计算其真实变成及中心点高度
                        # 再通过真实值判断其形状
                        if abs(w - h) < 10:
                            if w > 100 or h > 100:
                                objType = "cube_1"  # 正方体
                                self.shape = 'cube'
                            else:
                                objType = "box_1"  # 长方体
                                self.shape = 'box'
                        else:
                            objType = "cube_2"  # 正方体
                            self.shape = 'cube'

                    # cv2.rectangle(depth_color_map, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    # cv2.putText(depth_color_map, objType[:-2], (x + w // 2, y + (h //2) - 10), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
                    # cv2.putText(depth_color_map, objType[:-2], (x + w // 2, y + (h //2) - 10), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 1)
                    print(self.shape)
                    if self.shape == rospy.get_param('/shape_recognition/target_shape', "box"):
                        break
                    else:
                        self.shape = "None"
                # print(self.shape,self.target_shape)
                if self.last_shape == shape and shape != 'None' and self.shape == self.target_shape:
                    print(self.count)
                    self.count += 1
                    self.shape = 'None'
                # else:
                # self.count = 0
                if contour is not None:
                    angle = 0
                    if shape == "cylinder_1" or shape == "cuboid_1":
                        center, (width, height), angle = cv2.minAreaRect(contour)
                        if angle < -45:
                            angle += 90
                        if width > height and width / height > 1.5:
                            print("wh: ", width, height)
                            angle = angle + 90
                        # cv2.drawContours(depth_color_map, [np.int0(cv2.boxPoints((center, (width,height), angle)))], -1, (0, 0, 255), 2, cv2.LINE_AA)
                    if self.count > 3:
                        # 计算目标位置（核心修改：加入关节旋转角度补偿）
                        (cx, cy), r = cv2.minEnclosingCircle(obj)
                        K = depth_camera_info.K
                        position = depth_pixel_to_camera((cx, cy), min_dist / 1000, (K[0], K[4], K[2], K[5]))
                        position[0] -= 0.0171
                        pose_end = np.matmul(self.hand2cam_tf_matrix, xyz_euler_to_mat(position, (0, 0, 0)))
                        world_pose = np.matmul(self.endpoint, pose_end)
                        pose_t, pose_r = mat_to_xyz_euler(world_pose)

                        # 关键修改：补偿1号关节旋转带来的角度偏差
                        # 夹爪角度 = 目标角度 + 关节累计旋转角度（抵消旋转影响）
                        compensated_angle = angle + self.joint1_rotation

                        # 应用偏移量
                        pose_t[0] += self.offset_x
                        pose_t[1] += self.offset_y
                        pose_t[2] += self.offset_z
                        self.count = 0
                        self.moving = True

                        # 关键修改：传入补偿后的角度
                        threading.Thread(target=self.move, args=(shape[:-2], pose_t, compensated_angle)).start()

                        # 重置搜索状态和关节旋转记录
                        self.search_attempts = 0
                        self.search_delay = 0
                        self.search_state = 0
                        self.joint1_rotation = 0.0  # 重置旋转累计值

                    # 在"if self.count > 3:"判断之后添加
                    else:
                        # 未找到目标
                        if self.pick_state and self.shape != self.target_shape:
                            if self.search_delay < self.search_delay_max:
                                # 延迟一段时间，确保图像稳定
                                self.search_delay += 1
                                return

                            self.search_attempts += 1

                            if self.search_attempts >= self.max_attempts:
                                # 多次尝试未找到，进入搜索流程
                                if self.search_state == 0:
                                    # 第一次未找到，向左旋转30度
                                    self.rotate_arm(-30)
                                    self.search_state = 1
                                    self.search_attempts = 0
                                    self.search_delay = 0
                                    rospy.loginfo("未找到目标，向左旋转30度")
                                elif self.search_state == 1:
                                    # 左转后仍未找到，向右旋转60度（回到初始位置）
                                    self.rotate_arm(60)
                                    self.search_state = 2
                                    self.search_attempts = 0
                                    self.search_delay = 0
                                    rospy.loginfo("仍未找到目标，向右旋转30度")
                                elif self.search_state == 2:
                                    # 搜索完毕未找到，复位并结束
                                    self.rotate_arm(-30)
                                    self.search_state = 0
                                    self.pick_state = False
                                    self.search_attempts = 0
                                    self.search_delay = 0
                                    rospy.loginfo("搜索完毕，未找到目标")
                self.last_shape = shape

                # bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                self.fps.update()

            elif self.calibration_flat:
                # print("2")
                config_path = os.path.join(
                    os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0], '../../..')),
                    'config/config.yaml')
                transition_depth_image = np.zeros((400, 640), dtype=float)
                rgb_image = np.ndarray(shape=(ros_rgb_image.height, ros_rgb_image.width, 3), dtype=np.uint8,
                                       buffer=ros_rgb_image.data)
                depth_image = np.ndarray(shape=(ros_depth_image.height, ros_depth_image.width), dtype=np.uint16,
                                         buffer=ros_depth_image.data)
                print("data_20: ", np.max(depth_image[20]))
                print("data_200: ", np.max(depth_image[200]))
                print("data_350: ", np.max(depth_image[350]))
                calibration_20 = float(np.max(depth_image[200]) / np.max(depth_image[20]))
                calibration_350 = float(np.max(depth_image[200]) / np.max(depth_image[350]))
                print("data_200 / data_20: ", calibration_20)
                print("data_200 / data_350: ", calibration_350)
                calibration_compensation = LinearRegression()
                calibration_compensation.fit([[20], [200], [350]], [[calibration_20], [1], [calibration_350]])
                calibration_depth_compensation = []
                # 由于相机只是再y轴翻转，使用这里只需要校准y轴
                for i in range(399):
                    calibration_depth_compensation.append(calibration_compensation.predict([[i]]))
                for j in range(399):
                    transition_depth_image[j] = depth_image[j] * calibration_depth_compensation[j]

                print("transition_data_20: ", np.max(transition_depth_image[20]))
                print("transition_data_200: ", np.max(transition_depth_image[200]))
                print("transition_data_350: ", np.max(transition_depth_image[350]))
                print("target_dis: ", np.max(transition_depth_image[200]) - 10)

                print('正在保存参数shape_flat')
                config = common.get_yaml_data(config_path)
                config['shape_flat'][0] = calibration_20
                config['shape_flat'][1] = calibration_350
                common.save_yaml_data(config, config_path)
                rospy.sleep(2)
                print('保存完毕shape_flat')
                self.rgb_sub.unregister()  # 关闭话题
                self.depth_sub.unregister()
                self.info_sub.unregister()
                # depth_image = cv2.applyColorMap(depth_image.astype(np.uint8), cv2.COLORMAP_HOT)
                # cv2.line(depth_image,(200,350),(480,350),(0,255,0),3,cv2.LINE_4)
                # cv2.circle(depth_image, (320, 350), 5, (255, 0, 0), -1)  # 画出中心点
                # cv2.line(depth_image,(200,20),(480,20),(0,255,0),3,cv2.LINE_4)
                # cv2.circle(depth_image, (320, 20), 5, (255, 0, 0), -1)  # 画出中心点
                # cv2.imshow("depth", depth_image)
                # key = cv2.waitKey(1)

            elif self.calibration_dist:

                config_path = os.path.join(
                    os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0], '../../..')),
                    'config/config.yaml')
                rgb_image = np.ndarray(shape=(ros_rgb_image.height, ros_rgb_image.width, 3), dtype=np.uint8,
                                       buffer=ros_rgb_image.data)
                depth_image = np.ndarray(shape=(ros_depth_image.height, ros_depth_image.width), dtype=np.uint16,
                                         buffer=ros_depth_image.data)

                ih, iw = depth_image.shape[:2]

                depth_image = depth_image.copy()
                for j in range(399):
                    depth_image[j] = depth_image[j] * self.line_depth_compensation[j]

                print("transition_data_20: ", np.max(depth_image[20]))
                print("transition_data_200: ", np.max(depth_image[200]))
                print("transition_data_350: ", np.max(depth_image[350]))
                # sim_depth_image = cv2.applyColorMap(depth_image.astype(np.uint8), cv2.COLORMAP_HOT)
                # cv2.imshow("depth", sim_depth_image)
                # key = cv2.waitKey(1)
                # 屏蔽掉一些区域，降低识别条件，使识别跟可靠
                depth_image[:, 0:50] = np.array([[1000, ] * 50] * 400)
                depth_image[:, 590:640] = np.array([[1000, ] * 50] * 400)
                depth_image[320:400, :] = np.array([[1000, ] * 640] * 80)
                # depth_image[0:30, :] = np.array([[1000,]*640]* 30)
                depth = np.copy(depth_image).reshape((-1,))
                depth[depth <= 0] = 55555  # 距离为0可能是进入死区，或者颜色问题识别不到，将距离赋一个大值
                min_index = np.argmin(depth)  # 距离最小的像素
                min_y = min_index // iw
                min_x = min_index - min_y * iw

                min_dist = depth_image[min_y, min_x]  # 获取最小距离值
                print(min_dist)
                print('正在保存参数shape_dist')
                config = common.get_yaml_data(config_path)
                config['shape_dist'] = float(min_dist)
                common.save_yaml_data(config, config_path)
                rospy.sleep(2)
                print('保存完毕shape_flat')
                self.rgb_sub.unregister()  # 关闭话题
                self.depth_sub.unregister()
                self.info_sub.unregister()
            else:
                rospy.sleep(0.01)
        except Exception as e:
            rospy.logerr('callback error:', str(e))


if __name__ == "__main__":

    node = RgbDepthImageNode()
    try:
        rospy.spin()
    except exception as e:
        # mecnum_pub.publish(twist())
        rospy.logerr(str(e))
