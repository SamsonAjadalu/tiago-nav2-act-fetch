from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    params_file = LaunchConfiguration("params_file")
    map_yaml_file = LaunchConfiguration("map")
    use_rviz = LaunchConfiguration("use_rviz")
    rviz_config_file = LaunchConfiguration("rviz_config")
    use_clock_fallback = LaunchConfiguration("use_clock_fallback")
    clock_source_topic = LaunchConfiguration("clock_source_topic")
    initial_pose_x = LaunchConfiguration("initial_pose_x")
    initial_pose_y = LaunchConfiguration("initial_pose_y")
    initial_pose_z = LaunchConfiguration("initial_pose_z")
    initial_pose_yaw = LaunchConfiguration("initial_pose_yaw")

    # Path-based rewrites; plain string replace matched INITIAL_POSE_Y inside INITIAL_POSE_YAW.
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

    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("nav2_bringup"), "launch", "bringup_launch.py"]
            )
        ),
        launch_arguments={
            "namespace": namespace,
            "use_namespace": "True",
            "slam": "False",
            "map": map_yaml_file,
            "use_sim_time": use_sim_time,
            "autostart": autostart,
            "use_composition": "False",
            "use_respawn": "True",
            "params_file": params_file_resolved,
        }.items(),
    )

    nav2_rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("nav2_bringup"), "launch", "rviz_launch.py"]
            )
        ),
        condition=IfCondition(use_rviz),
        launch_arguments={
            "namespace": namespace,
            "use_namespace": "True",
            "rviz_config": rviz_config_file,
        }.items(),
    )

    clock_fallback = ExecuteProcess(
        cmd=[
            "python3",
            PathJoinSubstitution([FindPackageShare("tiago_nav"), "scripts", "clock_fallback.py"]),
            "--ros-args",
            "-p",
            PythonExpression(["'clock_source_topic:=", clock_source_topic, "'"]),
        ],
        output="screen",
        condition=IfCondition(
            PythonExpression(
                ["'", use_sim_time, "' == 'true' and '", use_clock_fallback, "' == 'true'"]
            )
        ),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value="tiago"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("autostart", default_value="true"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("use_clock_fallback", default_value="true"),
            DeclareLaunchArgument("clock_source_topic", default_value="/tiago/clock"),
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("tiago_nav"), "config", "nav2_params.yaml"]
                ),
                description="Base params; AMCL initial_pose is filled from initial_pose_* args",
            ),
            DeclareLaunchArgument(
                "initial_pose_x",
                default_value="-0.84791",
                description="AMCL initial pose x (map frame), Isaac prim translate X",
            ),
            DeclareLaunchArgument(
                "initial_pose_y",
                default_value="-0.79706",
                description="AMCL initial pose y (map frame), Isaac prim translate Y",
            ),
            DeclareLaunchArgument(
                "initial_pose_z",
                default_value="0.0",
                description="AMCL initial pose z (map frame)",
            ),
            DeclareLaunchArgument(
                "initial_pose_yaw",
                default_value="3.141592653589793",
                description="AMCL initial yaw in radians (map frame); pi = 180 deg Isaac Z",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("tiago_nav"), "config", "tiago_nav_default_view.rviz"]
                ),
            ),
            DeclareLaunchArgument(
                "map",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("tiago_nav"), "maps", "tiago_house.yaml"]
                ),
            ),
            clock_fallback,
            nav2_bringup,
            nav2_rviz,
        ]
    )
