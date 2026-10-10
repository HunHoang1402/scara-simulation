import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_model = os.path.join(
        get_package_share_directory('scara_sim'), 'models', 'scara_real.xml')
    return LaunchDescription([
        DeclareLaunchArgument('model_path', default_value=default_model,
                              description='Path to the MuJoCo XML model'),
        Node(
            package='scara_sim',
            executable='mujoco_bridge',
            name='mujoco_bridge',
            output='screen',
            parameters=[{'model_path': LaunchConfiguration('model_path')}],
        ),
    ])
