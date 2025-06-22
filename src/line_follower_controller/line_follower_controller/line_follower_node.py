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
        # 控制器增益
        self.Kp = 0.008  # 比例增益，这是需要调试的关键参数！
        # 机器人速度
        self.forward_speed = 0.15 # 恒定的前进速度

        # --- ROS 接口 ---
        self.subscription = self.create_subscription(
            Image,
            '/rgb_camera',
            self.image_callback,
            10)
        
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        
        self.get_logger().info('Line Follower Node has been started. Waiting for images...')

    def image_callback(self, msg):
        """
        核心逻辑：处理图像，计算误差，发布控制指令。
        """
        try:
            # 1. 将ROS图像消息转换为OpenCV格式 (BGR)
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return
            
        # 获取图像尺寸
        height, width, _ = cv_image.shape

        # 2. 图像处理
        # a) 裁剪图像，只关注下半部分，这是我们的“感兴趣区域”(Region of Interest, ROI)
        # 这样可以排除远处景物的干扰
        roi_top = int(height * 2 / 3) 
        roi = cv_image[roi_top:, :]

        # b) 转换为灰度图
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # c) 二值化处理
        # 将灰度图变成非黑即白的二值图，方便寻找轮廓
        # 阈值设为50，因为我们的线是纯黑(0)，地面是灰色(0.8*255=204)，50是个安全值
        # cv2.THRESH_BINARY_INV：反转二值化，使得黑线变为白色(255)，背景变为黑色(0)
        # 这样做的目的是，我们接下来要找的"质心"是白色区域的质心
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        # 3. 计算质心 (寻找线的中心)
        # cv2.moments会计算二值图像中所有白色像素的矩
        M = cv2.moments(binary)

        error = 0.0
        
        # 确保分母不为零，即图像中检测到了线 (白色像素)
        if M['m00'] > 0:
            # 计算质心的x坐标
            cx = int(M['m10'] / M['m00'])
            
            # 4. 计算误差
            # 误差 = 质心位置 - 图像中心位置
            # 图像中心在 width / 2
            center_of_image = width // 2
            error = cx - center_of_image
            
            # --- 可视化调试 (非常重要!) ---
            # 在质心位置画一个红圈
            cv2.circle(roi, (cx, roi.shape[0] // 2), 10, (0, 0, 255), -1)

        else:
            # 如果没有检测到线，我们暂时先让它停下来
            # 后面我们会处理“丢失线”的情况
            self.get_logger().warn('Line not detected!')
            # 保持上一次的误差或者设置为0，这里我们先让它直行
            error = 0.0


        # 5. P控制器计算角速度
        # 角速度 = -Kp * 误差
        # 负号用于修正方向：
        # - 如果线在右边, error > 0, 我们需要向右转, angular.z < 0
        # - 如果线在左边, error < 0, 我们需要向左转, angular.z > 0
        angular_z = -self.Kp * error

        # 6. 创建并发布Twist消息
        twist = Twist()
        twist.linear.x = self.forward_speed
        twist.angular.z = float(angular_z) # 确保是float类型
        self.publisher_.publish(twist)

        # 打印调试信息
        self.get_logger().info(f'Error: {error:.2f}, Angular Vel: {twist.angular.z:.3f}')
        
        # --- 可视化调试 (续) ---
        # 显示处理后的二值图像和带有标记的ROI
        cv2.imshow("Binary Image", binary)
        cv2.imshow("Region of Interest with Centroid", roi)
        cv2.waitKey(1) # 必须有这行，cv2.imshow才会刷新


def main(rclpy_args=None):
    rclpy.init(args=rclpy_args)
    line_follower_node = LineFollowerNode()
    rclpy.spin(line_follower_node)
    
    # 在关闭前停止机器人
    stop_twist = Twist()
    line_follower_node.publisher_.publish(stop_twist)
    cv2.destroyAllWindows()
    line_follower_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()