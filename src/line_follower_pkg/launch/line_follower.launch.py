from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_topic',
            default_value='/my_robot/camera_sensor/image_raw', # 与 Gazebo 插件输出对应
            description='Topic for input camera images'),
        DeclareLaunchArgument(
            'cmd_vel_topic',
            default_value='/cmd_vel', # 与 Gazebo 差速驱动插件输入对应
            description='Topic to publish cmd_vel'),
        DeclareLaunchArgument(
            'enable_image_processing',
            default_value='true',
            description='Enable/disable image processing and robot movement'),
        # 你也可以在这里为HSV阈值等设置默认参数

        Node(
            package='line_follower_pkg',
            executable='line_follower_node',
            name='line_follower', # 节点在ROS网络中的名字
            output='screen',
            parameters=[{ # 设置参数
                'camera_topic': LaunchConfiguration('camera_topic'),
                'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
                'enable_image_processing': LaunchConfiguration('enable_image_processing'),
                # 示例HSV值，你需要根据实际情况调整
                'target_line_color_lower': [0, 0, 0],    # HSV Lower bound for black
                'target_line_color_upper': [180, 255, 50], # HSV Upper bound for black
                'linear_velocity': 0.08, # 调整前进速度
                'angular_velocity_gain': 0.7 # 调整转向灵敏度
            }]
        )
    ])