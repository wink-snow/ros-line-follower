# ~/ros2_ws/src/line_follower_controller/line_follower_controller/line_follower_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class LineFollowerNode(Node):
    def __init__(self):
        super().__init__('line_follower_node')
        
        # --- 参数定义 ---
        # 机器人速度
        self.forward_speed = 0.15      # 正常循线时的前进速度
        self.search_speed_angular = 0.1 # 丢失线时的旋转搜索速度

        # 控制器增益 (PD)
        self.Kp = 0.008  # 比例增益 (保持你调好的值)
        self.Kd = 0.005   # 微分增益 (新参数，需要调试!)

        # --- 状态变量 ---
        self.last_error = 0.0      # 保存上一次的误差
        self.line_detected = False # 标记当前是否检测到线

        # --- ROS 接口 ---
        self.subscription = self.create_subscription(
            Image,
            '/rgb_camera',
            self.image_callback,
            10)
        
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        
        self.get_logger().info('Line Follower Node (PD Control) has been started.')

    def image_callback(self, msg):
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
        
        twist = Twist()
        current_error = 0.0

        if M['m00'] > 0:
            # --- 正常循线逻辑 ---
            self.line_detected = True
            
            # 1. 计算质心和当前误差
            cx = int(M['m10'] / M['m00'])
            center_of_image = width // 2
            current_error = cx - center_of_image
            
            # 2. 计算误差的变化率 (微分项)
            # (current_error - self.last_error) 是误差的变化量
            # 我们假设每次回调的时间间隔是固定的，所以可以省略除以dt
            derivative_error = current_error - self.last_error
            
            # 3. PD控制器计算角速度
            angular_z = -self.Kp * current_error - self.Kd * derivative_error
            
            # 4. 设置速度指令
            twist.linear.x = self.forward_speed
            twist.angular.z = float(angular_z)
            
            # 更新上一次的误差
            self.last_error = current_error

            # 可视化
            cv2.circle(roi, (cx, roi.shape[0] // 2), 10, (0, 255, 0), -1) # 绿色表示检测到线
            self.get_logger().info(f'Line DETECTED | Error: {current_error:.2f}, D_Error: {derivative_error:.2f}, Angular Vel: {twist.angular.z:.3f}')

        else:
            # --- 线丢失处理逻辑 ---
            self.line_detected = False
            self.get_logger().warn('Line LOST! Initiating search...')
            
            # 停止前进
            twist.linear.x = 0.0
            
            # 根据最后一次的误差方向进行旋转搜索
            if self.last_error > 0:
                # 最后一次线在右边，向右转
                twist.angular.z = -self.search_speed_angular 
            else:
                # 最后一次线在左边或误差为0，向左转
                twist.angular.z = self.search_speed_angular

        # 发布最终的速度指令
        self.publisher_.publish(twist)

        # 可视化
        cv2.imshow("Binary Image", binary)
        cv2.imshow("Region of Interest", roi)
        cv2.waitKey(1)

def main(rclpy_args=None):
    rclpy.init(args=rclpy_args)
    line_follower_node = LineFollowerNode()
    rclpy.spin(line_follower_node)
    
    stop_twist = Twist()
    line_follower_node.publisher_.publish(stop_twist)
    cv2.destroyAllWindows()
    line_follower_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()