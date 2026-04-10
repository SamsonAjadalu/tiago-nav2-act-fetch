from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    params_file = LaunchConfiguration("params_file")

    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("nav2_bringup"), "launch", "navigation_launch.py"]
            )
        ),
        launch_arguments={
            "namespace": namespace,
            "use_sim_time": use_sim_time,
            "autostart": autostart,
            "use_composition": "False",
            "use_respawn": "True",
            "params_file": params_file,
        }.items(),
    )

    navigation_group = GroupAction(
        [
            PushRosNamespace(namespace=namespace),
            navigation_launch,
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value="tiago"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("autostart", default_value="true"),
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("tiago_nav"), "config", "nav2_params.yaml"]
                ),
            ),
            navigation_group,
        ]
    )
