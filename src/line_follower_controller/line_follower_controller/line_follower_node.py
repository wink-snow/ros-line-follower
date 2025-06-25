import rclpy
import math
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

def euler_from_quaternion(q):
    """
    将 geometry_msgs/Quaternion 消息转换为欧拉角 (roll, pitch, yaw)。
    yaw 是绕 z 轴的旋转，是我们最关心的。
    """
    t0 = +2.0 * (q.w * q.x + q.y * q.z)
    t1 = +1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (q.w * q.y - q.z * q.x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (q.w * q.z + q.x * q.y)
    t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z


class LineFollowerNode(Node):
    def __init__(self):
        super().__init__('line_follower_node_with_memory') # 建议给节点一个新名字
        
        # --- 状态与记忆模块 ---
        self.robot_state = 'RECORDING'  # 初始状态为记录模式
        self.path = []                  # 用于存储路径点的列表, [(x, y), ...]
        self.smoothed_path = []         # 用于存储平滑后的路径
        self.current_pose = None        # 存储当前位姿
        self.current_yaw = 0.0          # 存储当前偏航角

        # --- 可调参数 ---
        # 速度
        self.mapping_speed = 0.2         # 绕行时的速度
        self.forward_speed = 0.5       # 正常循线时的前进速度
        self.search_speed_angular = 0.15 # 丢失线时的旋转搜索速度
        self.search_direction = 1
        self.search_tmp_yaw = 0.0
        self.search_degree = 0.0
        self.last_search_yaw = 0.0      # 用于计算增量
        self.total_rotation_accumulator = 0.0
        self.search_degree_threshold = math.radians(60)
        # PD 控制器
        self.Kp = 0.008                 # 比例增益
        self.Kd = 0.005                 # 微分增益
        # 路径记录与检测
        self.path_record_threshold = 0.05   # 每隔多少米记录一个路径点
        # 路径平滑
        self.smoothing_window_size = 5      # 移动平均滤波的窗口大小

        self.obstacle_detection_distance = 0.5 # 前方多远有东西算障碍物
        self.obstacle_detection_angle = np.deg2rad(30)

        self.avoidance_state = "TURN" # 避障子状态: TURN -> FORWARD -> TURN_BACK
        self.avoidance_turn_angle = np.deg2rad(90)
        self.avoidance_forward_time = 5.0 # 秒
        self.avoidance_target_yaw = 0.0
        self.avoidance_timer = 0.0

        self.avoidance_path = []  # 用于存储避障路径点

        self.recovery_target_point = None   # 存储计算出的恢复目标点 (x, y)
        self.Kp_recovery = 1.0              # 恢复阶段的转向P控制器增益
        self.recovery_lookahead_distance = 2 # 恢复目标点的前瞻距离 (米)
        self.recovery_completion_threshold = 0.1 # 距离目标点多近算完成 (米)
        self.is_reached_recovery_target = False # 是否到达恢复目标点

        # --- 内部状态变量 ---
        self.last_error = 0.0
        self.last_seen_left_pixels = 0   # 记录最后一次看到线时，左侧区域的像素数
        self.last_seen_right_pixels = 0 

        self.scan_data = None
        
        # --- ROS 接口 ---
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.image_subscription = self.create_subscription(
            Image,
            '/rgb_camera',
            self.image_callback,
            20)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        
        self.get_logger().info(f'Node started in "{self.robot_state}" mode.')

    def scan_callback(self, msg):
        self.scan_data = msg

    def odom_callback(self, msg: Odometry):
        """更新机器人当前的位姿和偏航角"""
        self.current_pose = msg.pose.pose
        _, _, self.current_yaw = euler_from_quaternion(self.current_pose.orientation)

    def image_callback(self, msg: Image):
        """主回调函数，作为状态机分发器"""
        if self.current_pose is None:
            self.get_logger().warn('Waiting for initial odometry data...', throttle_duration_sec=2)
            return
    
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image in main callback: {e}')
            return

        # --- 状态机 RECORDING -> SEARCHING -> PROCESSING -> STOPPED ---
        # --- 控制逻辑分发 ---
        if self.robot_state == 'RECORDING':
            self.perform_recording_lap(cv_image)
        elif self.robot_state == 'SEARCHING':
            self.perform_search(cv_image)
        elif self.robot_state == 'AVOIDANCE':
            self.perform_avoidance()
        elif self.robot_state == 'RECOVERY':
            self.perform_recovery(cv_image)
        elif self.robot_state == 'STOPPED':
            # 保持静止
            self.cmd_vel_publisher.publish(Twist())
            # 可以在这里只打印一次信息
            self.get_logger().info('Task complete. Robot is stopped.', throttle_duration_sec=10)
    
    @staticmethod
    def apply_trapezoidal_mask(image):
        height, width = image.shape[:2]
        
        # 梯形的四个顶点
        bottom_left = (0, height)
        bottom_right = (width, height)
        top_left = (int(width * 0.2), int(height * 0.6))
        top_right = (int(width * 0.8), int(height * 0.6))
        
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

        mask = np.zeros_like(image)
        
        # 在掩码上将梯形区域填充为白色
        if len(mask.shape) > 2: # 彩色图
            cv2.fillPoly(mask, vertices, (255, 255, 255))
        else: # 灰度图
            cv2.fillPoly(mask, vertices, 255)
        
        # 使用 bitwise_and 将掩码应用到原始图像上
        masked_image = cv2.bitwise_and(image, mask)
        
        return masked_image, mask # 同时返回掩码本身
    
    def process_image_for_line(self, cv_image):
        """图像处理部分"""
        height, width, _ = cv_image.shape
        roi_top = int(height * 2 / 3) 
        roi = cv_image[roi_top:, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary_full_roi = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        binary, _ = self.apply_trapezoidal_mask(binary_full_roi)
        M = cv2.moments(binary)
        return roi, binary, M
    
    def perform_recording_lap(self, cv_image):
        """执行循线、记录路径的任务"""

        roi, binary, M = self.process_image_for_line(cv_image)

        # PD控制与路径记录
        twist = Twist()

        if self.is_obstacle_ahead():
            self.get_logger().warn("Obstacle detected! Switching to AVOIDANCE mode.")
            self.robot_state = 'AVOIDANCE'
            self.avoidance_state = 'TURN' # 初始化避障子状态
            self.process_path()  # 在避障前处理过往路径
            # 决定向左还是向右转
            self.avoidance_target_yaw = self.current_yaw + (self.avoidance_turn_angle * self.get_avoidance_direction())
            # 规范化角度到 [-pi, pi]
            self.avoidance_target_yaw = np.arctan2(np.sin(self.avoidance_target_yaw), np.cos(self.avoidance_target_yaw))
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_publisher.publish(twist)
            return
        
        if M['m00'] > 0:
            self.search_tmp_yaw = 0.0
            self.search_degree = 0.0
            # --- 正常循线 ---
            cx = int(M['m10'] / M['m00'])
            cv2.circle(roi, (cx, roi.shape[0] // 2), 10, (0, 0, 255), -1)

            center_of_image = roi.shape[1] // 2
            current_error = cx - center_of_image
            derivative_error = current_error - self.last_error
            
            angular_z = -self.Kp * current_error - self.Kd * derivative_error
            
            twist.linear.x = self.forward_speed
            twist.angular.z = float(angular_z)
            self.last_error = current_error

            # --- 路径记录 ---
            self.record_path_point()

             # --- 分区分析 ---
            h, w = binary.shape
            # 将二值图的宽度分成三部分
            left_zone = binary[:, 0 : w//3]
            right_zone = binary[:, w*2//3 : w]
            
            # 计算并存储非零像素（即线的像素）数量
            self.last_seen_left_pixels = cv2.countNonZero(left_zone)
            self.last_seen_right_pixels = cv2.countNonZero(right_zone)

        else:
            # --- 线丢失处理 ---
            self.get_logger().warn('Line LOST! Initiating search...')
            self.robot_state = "SEARCHING"
            twist.linear.x = 0.0
            
            # 定义一个阈值，避免微小噪点的影响
            pixel_diff_threshold = 50 
            
            if self.last_seen_right_pixels > self.last_seen_left_pixels + pixel_diff_threshold:
                # 右侧像素明显更多，说明线往右拐了
                self.get_logger().info("Hint: Line likely curved right. Searching right.")
                self.search_direction = -1

            elif self.last_seen_left_pixels > self.last_seen_right_pixels + pixel_diff_threshold:
                # 左侧像素明显更多，说明线往左拐了
                self.get_logger().info("Hint: Line likely curved left. Searching left.")
                self.search_direction = 1
            else:
                # 如果两侧差不多，或者都没有，退回到原来的策略
                self.get_logger().info("Hint: No clear curve hint. Falling back to last error.")
                self.search_direction = -1 if self.last_error > 0 else 1
                
            twist.angular.z = self.search_speed_angular * self.search_direction

        self.cmd_vel_publisher.publish(twist)
        
        # 可视化 (可选)
        cv2.imshow("Binary Image", binary)
        cv2.imshow("Line Follower View", roi)
        cv2.waitKey(1)

    def perform_search(self, cv_image):
        _, _, M = self.process_image_for_line(cv_image)

        # 对机器人偏角变化做一个累加
        if self.search_tmp_yaw == 0.0:
            self.search_tmp_yaw = self.current_yaw
            self.last_search_yaw = self.current_yaw # <--- 初始化 last_search_yaw
            self.total_rotation_accumulator = 0.0
            self.get_logger().warn('Line LOST! Initiating search, recording start yaw.')

        delta_yaw = self.current_yaw - self.search_tmp_yaw
        # 修正角度，使其在[-pi, pi]范围内
        if delta_yaw > np.pi:
            delta_yaw -= 2 * np.pi
        elif delta_yaw < -np.pi:
            delta_yaw += 2 * np.pi

        self.search_degree = abs(delta_yaw)

        incremental_delta = self.current_yaw - self.last_search_yaw
        if incremental_delta > np.pi: incremental_delta -= 2 * np.pi
        elif incremental_delta < -np.pi: incremental_delta += 2 * np.pi

        self.total_rotation_accumulator += abs(incremental_delta)
        self.last_search_yaw = self.current_yaw

        if self.total_rotation_accumulator > math.radians(360):
            self.get_logger().info(f'Completed a full 360-degree search. Total rotation: {math.degrees(self.total_rotation_accumulator):.1f} degrees. Stopping search.')
            self.robot_state = 'STOPPED'
            # 在切换状态前确保机器人停止
            self.cmd_vel_publisher.publish(Twist())
            return
        
        self.get_logger().info(f'Searching... Angle from start: {np.degrees(self.search_degree):.1f}, Total rotation: {np.degrees(self.total_rotation_accumulator):.1f} degrees')

        if M['m00'] > 0:
            is_valid_find, _ = self.validate_found_line()

            if is_valid_find:
                self.get_logger().info('Line re-acquired! Switching back to RECORDING state.')
                self.robot_state = 'RECORDING'
                # 立即应用循线控制，避免延迟
                self.perform_recording_lap(cv_image) 
                return
            
            self.get_logger().warn("Invalid find: It's the path just traveled. Continuing search.")

        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = self.search_speed_angular * self.search_direction

        self.cmd_vel_publisher.publish(twist)

    def perform_avoidance(self):
         # 核心：一个三段式避障机动（转向->直行->转回）
        self.record_avoidance_path_point()  # 记录避障路径点
        twist = Twist()
        if self.avoidance_state == "TURN":
            angle_diff = self.avoidance_target_yaw - self.current_yaw
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            if abs(angle_diff) < np.deg2rad(5):
                self.avoidance_state = "FORWARD"
                self.avoidance_timer = time.time() # 启动直行计时器
                self.get_logger().info("Avoidance: Turn complete, moving forward.")
            else:
                twist.angular.z = 0.8 * (1 if angle_diff > 0 else -1)

        elif self.avoidance_state == "FORWARD":
            if time.time() - self.avoidance_timer > self.avoidance_forward_time:
                self.robot_state = "RECOVERY"
                self.recovery_target_point = None
                self.get_logger().info("Avoidance: Forward complete, switching to RECOVERY.")
            else:
                twist.linear.x = self.mapping_speed # 绕行时慢一点
                # 轻微调整方向，避免撞到侧面
                if self.is_obstacle_ahead(dist_thresh=0.3):
                    twist.angular.z = -0.5 * self.get_avoidance_direction()

        self.cmd_vel_publisher.publish(twist)

    def perform_recovery(self, cv_image):
        """
        执行从避障结束点恢复到原始路径的任务。
        """
        twist = Twist()

        _, _, M = self.process_image_for_line(cv_image)
        if M['m00'] > 0:
            self.get_logger().info("Recovery successful: Line re-acquired! Switching to RECORDING mode.")
            self.robot_state = 'RECORDING'
            self.recovery_target_point = None  # 为下次避障重置
            self.avoidance_path = []           # 清空本次避障路径
            self.is_reached_recovery_target = False # 重置恢复目标到未到达状态
            self.cmd_vel_publisher.publish(twist) # 发送停止指令以防万一
            return

        # 如果还没有计算恢复目标点，则进行一次性计算
        if self.recovery_target_point is None:
            self.get_logger().info("Calculating recovery target point...")
            
            # 安全检查：确保我们有足够的路径点来确定方向
            if len(self.smoothed_path) < 2:
                self.get_logger().error("Not enough points in smoothed_path to determine recovery direction. Stopping.")
                self.robot_state = 'STOPPED'
                self.cmd_vel_publisher.publish(twist) # 发布停止指令
                return

            # --- 获取关键点和向量 ---
            # 避障前的最后两个点
            p_last = np.array(self.smoothed_path[-1])
            p_before_last = np.array(self.smoothed_path[-2])
            
            # 机器人当前位置
            p_current = np.array([self.current_pose.position.x, self.current_pose.position.y])

            # 原始路径的最后方向向量
            v_dir = p_last - p_before_last
            # 归一化方向向量，得到单位方向
            v_dir_normalized = v_dir / np.linalg.norm(v_dir)

            # 从路径上的点指向机器人当前位置的向量
            v_robot_to_line = p_current - p_last

            # --- 计算投影点 ---
            # 将 v_robot_to_line 投影到 v_dir 上
            # 投影长度 t = (v_robot_to_line · v_dir) / ||v_dir||^2
            # 由于我们使用归一化的 v_dir_normalized，分母为1
            projection_length = np.dot(v_robot_to_line, v_dir_normalized)
            
            # 投影点 P_proj = P_last + projection_length * v_dir_normalized
            p_proj = p_last + projection_length * v_dir_normalized

            # --- 计算最终目标点（带前瞻距离） ---
            p_goal = p_proj + self.recovery_lookahead_distance * v_dir_normalized
            
            self.recovery_target_point = p_goal
            self.get_logger().info(f"Recovery target set to: ({p_goal[0]:.2f}, {p_goal[1]:.2f})")

        # 使用Go-to-Goal控制器导航到目标点
        if self.recovery_target_point is not None:
            p_current = np.array([self.current_pose.position.x, self.current_pose.position.y])
            p_goal = self.recovery_target_point

            # 计算到目标的距离和角度
            distance_to_goal = np.linalg.norm(p_goal - p_current)
            
            # 检查是否已到达目标点
            if distance_to_goal < self.recovery_completion_threshold or self.is_reached_recovery_target:
                
                twist.linear.x = self.forward_speed # 继续前进，直到找到线
                twist.angular.z = 0.0
                self.is_reached_recovery_target = True # 标记已到达恢复目标点
                self.get_logger().info(f"Reached recovery target, but line not found yet. Continuing search...")
            
            else:

                # --- Go-to-Goal 控制器 ---
                # 目标方向
                angle_to_goal = np.arctan2(p_goal[1] - p_current[1], p_goal[0] - p_current[0])
                # 角度误差
                angle_error = angle_to_goal - self.current_yaw
                # 规范化角度误差到 [-pi, pi]
                angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

                # 设置速度
                twist.linear.x = self.forward_speed
                twist.angular.z = self.Kp_recovery * angle_error
                
                self.get_logger().info(f"Recovering... Dist to goal: {distance_to_goal:.2f} m, Angle err: {np.degrees(angle_error):.1f} deg")

        self.cmd_vel_publisher.publish(twist)

    def is_obstacle_ahead(self, dist_thresh=None):
        if self.scan_data is None: return False
        if dist_thresh is None: dist_thresh = self.obstacle_detection_distance

        center_index = len(self.scan_data.ranges) // 2
        angle_increment = self.scan_data.angle_increment
        num_rays_to_check = int(self.obstacle_detection_angle / angle_increment)
        
        min_dist = float('inf')
        for i in range(center_index - num_rays_to_check, center_index + num_rays_to_check):
            if self.scan_data.ranges[i] < min_dist:
                min_dist = self.scan_data.ranges[i]
                
        return min_dist < dist_thresh
    
    def get_avoidance_direction(self):
        # 检查左边和右边哪个更开阔，决定绕行方向
        if self.scan_data is None: return 1 # 默认向左
        
        angle_increment = self.scan_data.angle_increment
        # 定义左右检测区域 (例如 30度到90度)
        left_start_angle = np.deg2rad(30)
        left_end_angle = np.deg2rad(90)
        right_start_angle = -np.deg2rad(90)
        right_end_angle = -np.deg2rad(30)
        
        # 计算索引
        left_start_idx = int((left_start_angle - self.scan_data.angle_min) / angle_increment)
        left_end_idx = int((left_end_angle - self.scan_data.angle_min) / angle_increment)
        right_start_idx = int((right_start_angle - self.scan_data.angle_min) / angle_increment)
        right_end_idx = int((right_end_angle - self.scan_data.angle_min) / angle_increment)

        left_ranges = self.scan_data.ranges[left_start_idx:left_end_idx]
        right_ranges = self.scan_data.ranges[right_start_idx:right_end_idx]
        
        avg_left_dist = np.mean([r for r in left_ranges if not np.isinf(r)])
        avg_right_dist = np.mean([r for r in right_ranges if not np.isinf(r)])
        
        if avg_left_dist > avg_right_dist:
            self.get_logger().info("Choosing LEFT for avoidance (more space).")
            return 1 # 向左转
        else:
            self.get_logger().info("Choosing RIGHT for avoidance (more space).")
            return -1 # 向右转

    def validate_found_line(self):
        """验证寻找到的线是否有效"""
        if abs(self.search_degree - math.radians(180)) < self.search_degree_threshold:
            return False, 'REPEAT'
        
        return True, 'OK'

    def record_path_point(self):
        """如果距离上一个点足够远，就记录一个新点。"""
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        if not self.path or np.linalg.norm([current_x - self.path[-1][0], current_y - self.path[-1][1]]) > self.path_record_threshold:
            self.path.append((current_x, current_y))
            self.get_logger().info(f'Path point recorded: ({current_x:.2f}, {current_y:.2f})', throttle_duration_sec=1)

    def record_avoidance_path_point(self):
        """记录避障路径点"""
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        if not self.avoidance_path or np.linalg.norm([current_x - self.avoidance_path[-1][0], current_y - self.avoidance_path[-1][1]]) > self.path_record_threshold:
            self.avoidance_path.append((current_x, current_y))
            self.get_logger().info(f'Avoidance path point recorded: ({current_x:.2f}, {current_y:.2f})', throttle_duration_sec=1)

    def process_path(self):
        """处理记录的路径"""
        self.get_logger().info(f'Processing recorded path with {len(self.path)} points.')
        
        if len(self.path) > self.smoothing_window_size:
            self.smoothed_path = self.smooth_path_moving_average(self.path, self.smoothing_window_size)
            self.get_logger().info(f'Path smoothed. New path has {len(self.smoothed_path)} points.')
            # 可以在这里将 self.smoothed_path 保存到文件
            # np.savetxt("smoothed_path.txt", self.smoothed_path)
        else:
            self.get_logger().warn('Not enough points to smooth the path.')
            self.smoothed_path = self.path

    def smooth_path_moving_average(self, path, window_size):
        """使用移动平均滤波器平滑路径"""
        path_np = np.array(path)
        x = path_np[:, 0]
        y = path_np[:, 1]
        
        # 使用卷积实现移动平均
        box = np.ones(window_size) / window_size
        x_smooth = np.convolve(x, box, mode='valid')
        y_smooth = np.convolve(y, box, mode='valid')
        
        return list(zip(x_smooth, y_smooth))


def main(rclpy_args=None):
    rclpy.init(args=rclpy_args)
    line_follower_node = LineFollowerNode()
    try:
        rclpy.spin(line_follower_node)
    except KeyboardInterrupt:
        pass
    finally:
        # 在关闭前确保机器人停止
        line_follower_node.get_logger().info('Shutting down, stopping robot...')
        line_follower_node.cmd_vel_publisher.publish(Twist())
        line_follower_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()