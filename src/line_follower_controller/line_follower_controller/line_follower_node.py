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
        self.search_speed_angular = 0.3 # 丢失线时的旋转搜索速度
        self.search_tmp_yaw = 0.0
        self.search_degree = 0.0
        self.search_degree_threshold = math.radians(130)
        # PD 控制器
        self.Kp = 0.008                 # 比例增益
        self.Kd = 0.005                 # 微分增益
        # 路径记录与检测
        self.path_record_threshold = 0.05   # 每隔多少米记录一个路径点
        # 路径平滑
        self.smoothing_window_size = 5      # 移动平均滤波的窗口大小

        # --- 内部状态变量 ---
        self.last_error = 0.0
        
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
            10)
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

        # --- 状态机 ---
        if self.robot_state == 'RECORDING':
            self.perform_recording_lap(msg)
        elif self.robot_state == 'PROCESSING':
            self.process_path()
        elif self.robot_state == 'STOPPED':
            # 保持静止
            self.cmd_vel_publisher.publish(Twist())
            # 可以在这里只打印一次信息
            self.get_logger().info('Task complete. Robot is stopped.', throttle_duration_sec=10)

    def perform_recording_lap(self, msg: Image):
        """执行循线、记录路径和检测圈末的任务"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return
            
        height, width, _ = cv_image.shape
        roi_top = int(height * 2 / 3) 
        roi = cv_image[roi_top:, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        M = cv2.moments(binary)

        # PD控制与路径记录
        twist = Twist()
        if M['m00'] > 0:
            self.search_tmp_yaw = 0.0
            self.search_degree = 0.0
            # --- 正常循线 ---
            cx = int(M['m10'] / M['m00'])
            center_of_image = width // 2
            current_error = cx - center_of_image
            derivative_error = current_error - self.last_error
            
            angular_z = -self.Kp * current_error - self.Kd * derivative_error
            
            twist.linear.x = self.forward_speed
            twist.angular.z = float(angular_z)
            self.last_error = current_error

            # --- 路径记录 ---
            current_x = self.current_pose.position.x
            current_y = self.current_pose.position.y
            if not self.path or np.linalg.norm([current_x - self.path[-1][0], current_y - self.path[-1][1]]) > self.path_record_threshold:
                self.path.append((current_x, current_y))
                self.get_logger().info(f'Path point recorded: ({current_x:.2f}, {current_y:.2f})')

        else:
            # --- 线丢失处理 ---
            self.get_logger().warn('Line LOST! Initiating search...')
            twist.linear.x = 0.0
            twist.angular.z = -self.search_speed_angular if self.last_error > 0 else self.search_speed_angular

            # 对机器人偏角变化做一个累加
            if self.search_tmp_yaw == 0.0:
                self.search_tmp_yaw = self.current_yaw
                self.get_logger().warn('Line LOST! Initiating search, recording start yaw.')

            delta_yaw = self.current_yaw - self.search_tmp_yaw
            # 修正角度，使其在[-pi, pi]范围内
            if delta_yaw > np.pi:
                delta_yaw -= 2 * np.pi
            elif delta_yaw < -np.pi:
                delta_yaw += 2 * np.pi

            # 更新总旋转角度（取绝对值）
            self.search_degree = abs(delta_yaw)
            
            self.get_logger().info(f'Searching... Total rotation: {np.degrees(self.search_degree):.1f} degrees')

            # 判断是否完成一圈
            # 用弧度进行比较
            if self.search_degree > self.search_degree_threshold: 
                self.get_logger().info('Search rotation exceeded the max degrees. Assuming lap completed!')
                self.get_logger().info('Switching to PROCESSING state.')
                self.robot_state = 'PROCESSING'
                self.cmd_vel_publisher.publish(Twist()) # 立即停止
                return


        self.cmd_vel_publisher.publish(twist)
        
        # 可视化 (可选)
        # cv2.imshow("Binary Image", binary)
        # cv2.waitKey(1)

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