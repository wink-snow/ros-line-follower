import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class PIDLineFollower(Node):
    def __init__(self):
        super().__init__('pid_line_follower_node')

        # 声明PID参数和速度参数，方便从launch文件调整
        self.declare_parameter('kp', 0.005)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.01)
        self.declare_parameter('linear_vel', 0.2) # 恒定的前进速度 m/s
        
        # 获取参数值
        self._kp = self.get_parameter('kp').get_parameter_value().double_value
        self._ki = self.get_parameter('ki').get_parameter_value().double_value
        self._kd = self.get_parameter('kd').get_parameter_value().double_value
        self._linear_vel = self.get_parameter('linear_vel').get_parameter_value().double_value
        
        # PID状态变量
        self._integral = 0.0
        self._previous_error = 10.0
        self._last_time = None

        # 创建订阅者，订阅语义标签图像
        self.subscription = self.create_subscription(
            Image,
            '/semantic_camera/labels_map',
            self.image_callback,
            10)
        
        # 创建发布者，发布速度指令
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        # 初始化cv_bridge
        self.bridge = CvBridge()
        
        self.get_logger().info('PID Line Follower Node has been started.')
        self.get_logger().info(f'PID Gains: Kp={self._kp}, Ki={self._ki}, Kd={self._kd}')
        self.get_logger().info(f'Constant Linear Velocity: {self._linear_vel} m/s')

    def image_callback(self, msg):
        """
        图像处理和PID控制的核心回调函数
        """
        try:
            # 将ROS Image消息转换为OpenCV图像 (numpy array)
            # 语义标签图像通常是单通道的，所以我们使用'passthrough'
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        # 图像中心是我们的目标位置
        h, w = cv_image.shape[:2]
        
        # 只关注图像底部的一块区域 (Region of Interest, ROI)
        roi_start_row = int(h * 3 / 4)
        roi = cv_image[roi_start_row:, :]

        # 创建一个二值掩码，其中像素值为1的(黑线)部分是白色(255)，其余是黑色(0)
        line_mask = cv2.inRange(roi, 1, 1)

        # 计算掩码的矩 (moments)，以找到其质心
        M = cv2.moments(line_mask)
        
        error = 0.0
        if M['m00'] > 0:
            # 如果在ROI中找到了线，计算其质心的x坐标
            cx = int(M['m10'] / M['m00'])
            
            # 图像中心的x坐标
            center_of_image = w // 2
            
            # 误差 = 质心位置 - 图像中心位置
            error = float(cx - center_of_image)
        else:
            # 如果在ROI中没看到线，保持上一次的误差，或者可以设置一个默认行为
            # 比如原地旋转寻找线。这里我们简单地使用上一次的误差。
            error = self._previous_error

        current_time = self.get_clock().now()
        
        if self._last_time is None:
            self._last_time = current_time
            return # 第一次回调时，我们没有dt，所以跳过

        dt = (current_time - self._last_time).nanoseconds / 1e9

        # 比例项
        p_term = self._kp * error

        # 积分项
        self._integral += error * dt
        i_term = self._ki * self._integral
        
        # 微分项
        derivative = (error - self._previous_error) / dt
        d_term = self._kd * derivative

        # PID总输出. 这个输出将用于控制角速度
        pid_output = p_term + i_term + d_term

        # 更新状态
        self._previous_error = error
        self._last_time = current_time

        twist_msg = Twist()
        twist_msg.linear.x = self._linear_vel
        
        # 关键: 将PID输出转换为角速度
        # 如果error为正(线在右边)，我们需要向右转(负的角速度)
        # 因此，角速度与error符号相反
        twist_msg.angular.z = -pid_output
        
        self.publisher_.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    pid_line_follower = PIDLineFollower()
    rclpy.spin(pid_line_follower)
    pid_line_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()