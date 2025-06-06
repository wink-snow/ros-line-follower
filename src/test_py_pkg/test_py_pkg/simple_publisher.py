import rclpy
from rclpy.node import Node
from std_msgs.msg import String 

class SimplePublisher(Node):

    def __init__(self):
        super().__init__('simple_publisher_node') 
        self.publisher_ = self.create_publisher(String, 'my_topic', 10) 
        timer_period = 1.0 
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('Simple Publisher node has been started and is publishing to "my_topic".')

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.get_clock().now()}' 
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    simple_publisher = SimplePublisher()
    rclpy.spin(simple_publisher)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    simple_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()