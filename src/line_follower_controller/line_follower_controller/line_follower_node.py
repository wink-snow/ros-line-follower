import rclpy
import math
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

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

        # --- 内部状态变量 ---
        self.last_error = 0.0
        self.last_seen_left_pixels = 0   # 记录最后一次看到线时，左侧区域的像素数
        self.last_seen_right_pixels = 0 
        
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
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        
        self.get_logger().info(f'Node started in "{self.robot_state}" mode.')

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
        elif self.robot_state == 'PROCESSING':
            self.process_path()
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
        """执行循线、记录路径和检测圈末的任务"""

        roi, binary, M = self.process_image_for_line(cv_image)

        # PD控制与路径记录
        twist = Twist()
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
            self.robot_state = 'PROCESSING'
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

    def process_path(self):
        """处理记录的路径，然后切换到STOPPED状态"""
        self.get_logger().info(f'Processing recorded path with {len(self.path)} points.')
        
        if len(self.path) > self.smoothing_window_size:
            self.smoothed_path = self.smooth_path_moving_average(self.path, self.smoothing_window_size)
            self.get_logger().info(f'Path smoothed. New path has {len(self.smoothed_path)} points.')
            # 可以在这里将 self.smoothed_path 保存到文件
            # np.savetxt("smoothed_path.txt", self.smoothed_path)
        else:
            self.get_logger().warn('Not enough points to smooth the path.')
            self.smoothed_path = self.path

        # 切换到最终状态
        self.robot_state = 'STOPPED'
        self.get_logger().info('Path processing complete. Switching to STOPPED state.')
    
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