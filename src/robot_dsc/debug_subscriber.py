import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class ImageDebugSubscriber(Node):
    def __init__(self):
        super().__init__('image_debug_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/semantic_camera/labels_map',
            self.listener_callback,
            10)
        self.bridge = CvBridge()
        self.get_logger().info('Debug subscriber started. Waiting for image...')

    def listener_callback(self, msg):
        self.get_logger().info('--- Received Image ---')
        
        # 打印消息头信息
        self.get_logger().info(f'Timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}')
        self.get_logger().info(f'Dimensions: {msg.width}x{msg.height}')
        self.get_logger().info(f'Encoding: {msg.encoding}')

        # 使用cv_bridge将ROS图像转为OpenCV图像
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        # 提取并打印特定像素的值
        # 例如，打印图像中心点的像素值
        center_x = msg.width // 2
        center_y = msg.height // 2
        pixel_value = cv_image[center_y, center_x]
        self.get_logger().info(f'Pixel value at center ({center_x}, {center_y}): {pixel_value}')

        # 例如，统计图像中值为 1 (黑线) 的像素数量
        line_pixels = np.count_nonzero(cv_image == 1)
        self.get_logger().info(f'Number of line pixels (value 1): {line_pixels}')

        # 销毁节点，这样只会处理并打印第一帧收到的图像，避免刷屏
        self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    image_debug_subscriber = ImageDebugSubscriber()
    rclpy.spin(image_debug_subscriber)
    rclpy.shutdown()

if __name__ == '__main__':
    main()