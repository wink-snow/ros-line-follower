import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # 原有的机器人描述包的路径
    pkg_my_robot_description = get_package_share_directory('robot_dsc') 
    
    # 仿真启动文件
    start_simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_my_robot_description, 'launch', 'spawn_robot_in_gazebo.launch.py')
        )
    )

    # 启动我们新创建的PID控制器节点
    pid_follower_node = Node(
        package='pid_line_follower',
        executable='pid_follower',
        name='pid_line_follower_node',
        output='screen',
        parameters=[
            # 在这里调整你的PID参数和速度！
            {'kp': 0.005},
            {'ki': 0.0001},
            {'kd': 0.01},
            {'linear_vel': 0.15} # 降低一点速度，方便调试
        ]
    )

    return LaunchDescription([
        start_simulation_launch,
        pid_follower_node
    ])