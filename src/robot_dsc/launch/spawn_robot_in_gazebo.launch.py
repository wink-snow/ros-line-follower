import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

import xacro

def generate_launch_description():
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_my_robot_description = get_package_share_directory('robot_dsc')

    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_file_name = 'line_follower_world.world' # 世界文件名
    world_path = os.path.join(pkg_my_robot_description, 'worlds', world_file_name)
    
    xacro_file_name = 'diff_drive.urdf.xacro'
    xacro_path = os.path.join(pkg_my_robot_description, 'urdf', xacro_file_name)

    # Gazebo server and client
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world_path, 'verbose': 'true'}.items()
    )

    # gzclient_cmd = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
    #     )
    # )
    
    # Robot State Publisher
    # Converts XACRO to URDF string
    doc = xacro.process_file(xacro_path) # <--- 使用 xacro.process_file
    robot_description_config = doc.toprettyxml(indent='  ') # <--- 获取 XML 字符串

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_config, # <--- 使用新的 robot_description_config
            'use_sim_time': use_sim_time
        }]
    )

    # Spawn Entity node (spawns robot from URDF to Gazebo)
    spawn_entity_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description', # Topic where URDF is published by robot_state_publisher
            '-entity', 'my_simple_bot',    # Name of the entity in Gazebo
            '-x', '0.0',                   # Initial x position
            '-y', '-1.0',                  # Initial y position (e.g., beside the line)
            '-z', '0.1',                   # Initial z position
            '-Y', '0.0'                    # Initial yaw orientation
        ],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'),
        
        gzserver_cmd,
        # gzclient_cmd, # 如果不需要GUI，可以注释掉这行以节省资源
        robot_state_publisher_node,
        spawn_entity_node
    ])