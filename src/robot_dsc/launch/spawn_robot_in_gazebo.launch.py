import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # 获取相关功能包的路径
    # ✅ Iron环境下，我们应该使用官方提供的ros_gz_sim
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim') 
    pkg_my_robot_description = get_package_share_directory('robot_dsc')

    # 设置默认路径
    default_world_path = os.path.join(pkg_my_robot_description, 'worlds', 'line_follower_world_v2.world')
    default_model_path = os.path.join(pkg_my_robot_description, 'urdf', 'diff_drive.urdf.xacro')

    # 声明启动参数
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world = LaunchConfiguration('world', default=default_world_path)
    model = LaunchConfiguration('model', default=default_model_path)
    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='-1.0')

    # ✅ 使用官方的gz_sim.launch.py，这是Iron的标准方式
    start_gazebo_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': ['-r ', world]}.items(), # '-r' 表示启动后运行
        # launch_arguments={'gz_args': ['-r ', world], 'gz_version': '7'}.items(),
    )

    robot_description_content = ParameterValue(
        Command(['xacro ', model]),
        value_type=str
    )

    # 机器人状态发布节点 (robot_state_publisher)
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': use_sim_time
        }]
    )

    # Gazebo 生成节点 (spawn_entity)
    spawn_entity_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'my_simple_bot',
            '-x', x_pose,
            '-y', y_pose,
            '-z', '0.1'
        ],
        output='screen'
    )

    # Bridge 节点
    # ✅ 在Iron+Garden中，使用方括号语法更佳
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # 摄像头 (Gazebo -> ROS)
            '/rgb_camera@sensor_msgs/msg/Image[gz.msgs.Image',
            '/semantic_camera/labels_map@sensor_msgs/msg/Image[gz.msgs.Image',
            '/semantic_camera/colored_map@sensor_msgs/msg/Image[gz.msgs.Image',
            
            # 控制 (ROS -> Gazebo)
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            
            # 反馈 (Gazebo -> ROS)
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V'
        ],
        output='screen'
    )

    return LaunchDescription([
        # 声明所有启动参数
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('world', default_value=default_world_path),
        DeclareLaunchArgument('model', default_value=default_model_path),
        DeclareLaunchArgument('x_pose', default_value='0.0'),
        DeclareLaunchArgument('y_pose', default_value='-1.0'),

        # 启动所有节点
        start_gazebo_cmd,
        robot_state_publisher_node,
        spawn_entity_node,
        bridge
    ])