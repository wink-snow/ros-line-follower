import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image 
from geometry_msgs.msg import Twist 
from cv_bridge import CvBridge 
import cv2 
import numpy as np 

class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower_node')
        
        # 参数
        self.declare_parameter('camera_topic', '/my_robot/camera_sensor/image_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('target_line_color_lower', [0,0,0]) # 黑色HSV下限
        self.declare_parameter('target_line_color_upper', [180,255,50]) # 黑色HSV上限 
        self.declare_parameter('linear_velocity', 0.1) # m/s
        self.declare_parameter('angular_velocity_gain', 0.5) # 增益，用于调整转向的灵敏度
        self.declare_parameter('enable_image_processing', True) # 是否启用图像处理和移动

        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        
        self.bridge = CvBridge() # CvBridge实例

        # 订阅摄像头图像
        self.image_subscription = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10) # QoS profile深度为10
        self.image_subscription  # prevent unused variable warning

        # 发布速度指令
        self.velocity_publisher = self.create_publisher(Twist, cmd_vel_topic, 10)

        self.get_logger().info(f"Line Follower Node started. Subscribing to {camera_topic}, publishing to {cmd_vel_topic}")

    def image_callback(self, msg):
        if not self.get_parameter('enable_image_processing').get_parameter_value().bool_value:
            # 如果禁用了图像处理，则不执行任何操作或发布停止命令
            stop_msg = Twist()
            self.velocity_publisher.publish(stop_msg)
            return

        try:
            # 将ROS Image消息转换为OpenCV图像 (BGR8格式)
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {str(e)}')
            return

        # 转换到HSV色彩空间
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # 根据颜色阈值创建掩码 (Mask)
        lower_bound = np.array(self.get_parameter('target_line_color_lower').get_parameter_value().integer_array_value, dtype=np.uint8)
        upper_bound = np.array(self.get_parameter('target_line_color_upper').get_parameter_value().integer_array_value, dtype=np.uint8)
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # 找到掩码中的轮廓或计算质心
        h, w, d = cv_image.shape
        search_top = int(3*h/4) 
        search_bot = int(3*h/4 + 20) 
        mask[0:search_top, 0:w] = 0
        mask[search_bot:h, 0:w] = 0 
        
        M = cv2.moments(mask)
        
        cmd_vel_msg = Twist()

        if M['m00'] > 0:
            cx = int(M['m10']/M['m00'])
            # cy = int(M['m01']/M['m00']) # y 坐标，这里可能不太需要
            # cv2.circle(cv_image, (cx, cy), 5, (0,0,255), -1) # 在原图上画出质心 (调试用)

            err = w/2 - cx
            cmd_vel_msg.linear.x = self.get_parameter('linear_velocity').get_parameter_value().double_value
            cmd_vel_msg.angular.z = self.get_parameter('angular_velocity_gain').get_parameter_value().double_value * (err / (w/2)) # 归一化误差
            
            # 限制角速度的绝对值 (可选)
            # max_angular_vel = 0.5 
            # cmd_vel_msg.angular.z = np.clip(cmd_vel_msg.angular.z, -max_angular_vel, max_angular_vel)

            self.get_logger().debug(f'Line detected. CX: {cx}, Error: {err}, Angular Vel: {cmd_vel_msg.angular.z}')
        else:
            # 没有检测到线条
            cmd_vel_msg.linear.x = 0.0
            cmd_vel_msg.angular.z = 0.1 # 例如，轻微右转寻找
            self.get_logger().info('No line detected. Stopping or searching.')
        
        self.velocity_publisher.publish(cmd_vel_msg)

        # 显示处理后的图像 (如果你的WSLg支持且资源允许)
        # cv2.imshow("Original Image", cv_image)
        # cv2.imshow("Mask", mask)
        # cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    line_follower = LineFollower()
    rclpy.spin(line_follower)
    line_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()