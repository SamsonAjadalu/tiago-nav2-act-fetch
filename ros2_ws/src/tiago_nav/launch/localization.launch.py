from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import PushRosNamespace
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    params_file = LaunchConfiguration("params_file")
    map_yaml_file = LaunchConfiguration("map")
    initial_pose_x = LaunchConfiguration("initial_pose_x")
    initial_pose_y = LaunchConfiguration("initial_pose_y")
    initial_pose_z = LaunchConfiguration("initial_pose_z")
    initial_pose_yaw = LaunchConfiguration("initial_pose_yaw")

    params_file_resolved = RewrittenYaml(
        source_file=params_file,
        param_rewrites={
            "amcl.ros__parameters.initial_pose.x": initial_pose_x,
            "amcl.ros__parameters.initial_pose.y": initial_pose_y,
            "amcl.ros__parameters.initial_pose.z": initial_pose_z,
            "amcl.ros__parameters.initial_pose.yaw": initial_pose_yaw,
        },
        convert_types=True,
    )

    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("nav2_bringup"), "launch", "localization_launch.py"]
            )
        ),
        launch_arguments={
            "namespace": namespace,
            "map": map_yaml_file,
            "use_sim_time": use_sim_time,
            "autostart": autostart,
            "use_composition": "False",
            "use_respawn": "True",
            "params_file": params_file_resolved,
        }.items(),
    )

    localization_group = GroupAction(
        [
            PushRosNamespace(namespace=namespace),
            localization_launch,
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
                    [FindPackageShare("tiago_nav"), "config", "amcl.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "initial_pose_x",
                default_value="-0.84791",
            ),
            DeclareLaunchArgument(
                "initial_pose_y",
                default_value="-0.79706",
            ),
            DeclareLaunchArgument(
                "initial_pose_z",
                default_value="0.0",
            ),
            DeclareLaunchArgument(
                "initial_pose_yaw",
                default_value="3.141592653589793",
            ),
            DeclareLaunchArgument(
                "map",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("tiago_nav"), "maps", "tiago_house.yaml"]
                ),
            ),
            localization_group,
        ]
    )
