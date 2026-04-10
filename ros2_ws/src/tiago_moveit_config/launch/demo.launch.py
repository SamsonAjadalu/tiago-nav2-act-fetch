import os
import re

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
import yaml


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def _urdf_mesh_paths_to_file_uri(urdf_xml: str) -> str:
    def sub(m: re.Match) -> str:
        p = m.group(1)
        if p.startswith(("package://", "file://", "http://", "https://")):
            return m.group(0)
        if p.startswith("/"):
            return f'filename="file://{p}"'
        return m.group(0)

    return re.sub(r'filename="([^"]+)"', sub, urdf_xml)


def _load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory("tiago_moveit_config")
    default_urdf = "/media/robotswithai/data1/robots/tiago_hil/tiago_urdf/tiago_dual_simple.urdf"

    allow_exec = LaunchConfiguration("allow_execution").perform(context).lower() in ("true", "1", "yes")
    mirror = LaunchConfiguration("mirror_joint_states").perform(context).lower() in ("true", "1", "yes")
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() in ("true", "1", "yes")
    relay_sim_clock = LaunchConfiguration("relay_sim_clock").perform(context).lower() in ("true", "1", "yes")

    namespace = LaunchConfiguration("namespace").perform(context)
    sim_clock_params = [{"use_sim_time": True}] if use_sim_time else []

    tf_remaps = [("tf", "/tf"), ("tf_static", "/tf_static")]
    joint_remaps = [("/joint_states", "joint_states")]

    robot_description = {"robot_description": _urdf_mesh_paths_to_file_uri(_load_text(default_urdf))}
    robot_description_semantic = {
        "robot_description_semantic": _load_text(os.path.join(pkg_share, "config", "tiago_dual.srdf"))
    }
    robot_description_kinematics = _load_yaml(os.path.join(pkg_share, "config", "kinematics.yaml"))
    robot_description_planning = _load_yaml(os.path.join(pkg_share, "config", "joint_limits.yaml"))

    ompl_planning_pipeline_config = {
        "move_group": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": (
                "default_planner_request_adapters/AddTimeOptimalParameterization "
                "default_planner_request_adapters/FixWorkspaceBounds "
                "default_planner_request_adapters/FixStartStateBounds "
                "default_planner_request_adapters/FixStartStateCollision "
                "default_planner_request_adapters/FixStartStatePathConstraints"
            ),
            "start_state_max_bounds_error": 0.1,
            "jiggle_fraction": 0.1,
        }
    }
    ompl_planning_pipeline_config["move_group"].update(
        _load_yaml(os.path.join(pkg_share, "config", "ompl_planning.yaml"))
    )

    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
        "publish_robot_description": True,
        "publish_robot_description_semantic": True,
    }

    move_group_params = [
        robot_description,
        robot_description_semantic,
        robot_description_kinematics,
        robot_description_planning,
        ompl_planning_pipeline_config,
        planning_scene_monitor_parameters,
        {"allow_trajectory_execution": allow_exec},
    ]
    if allow_exec:
        move_group_params.append(_load_yaml(os.path.join(pkg_share, "config", "moveit_controllers.yaml")))

    actions = []

    if use_sim_time and relay_sim_clock:
        actions.append(
            Node(
                package="tiago_moveit_config",
                executable="tiago_clock_relay.py",
                arguments=["--src", f"/{namespace}/clock", "--dst", "/clock"],
                output="screen",
            )
        )

    start_mg = LaunchConfiguration("start_move_group").perform(context).lower() in ("true", "1", "yes")
    if start_mg:
        actions.append(
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                namespace=namespace,
                output="screen",
                parameters=sim_clock_params + move_group_params,
                remappings=tf_remaps + joint_remaps,
            )
        )

    start_rsp = LaunchConfiguration("start_robot_state_publisher").perform(context).lower() in (
        "true",
        "1",
        "yes",
    )
    if start_rsp:
        actions.append(
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                namespace=namespace,
                output="screen",
                parameters=sim_clock_params + [robot_description],
                remappings=tf_remaps + joint_remaps,
            )
        )

    actions.append(
        Node(
            package="tiago_moveit_config",
            executable="fake_joint_state_publisher.py",
            namespace=namespace,
            output="screen",
            parameters=sim_clock_params,
            remappings=joint_remaps,
            condition=UnlessCondition(
                PythonExpression(["'", LaunchConfiguration("allow_execution"), "' == 'true'"])
            ),
        )
    )

    actions.append(
        Node(
            package="tiago_moveit_config",
            executable="isaac_joint_trajectory_bridge.py",
            namespace=namespace,
            output="screen",
            parameters=sim_clock_params + [{"mirror_joint_states": mirror}],
            remappings=joint_remaps,
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration("allow_execution"), "' == 'true'"])
            ),
        )
    )

    start_static_odom = LaunchConfiguration("start_odom_static_tf").perform(context).lower() in (
        "true",
        "1",
        "yes",
    )
    if start_static_odom:
        actions.append(
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                namespace=namespace,
                arguments=[
                    "--frame-id",
                    "odom",
                    "--child-frame-id",
                    "base_footprint",
                    "--x",
                    "0",
                    "--y",
                    "0",
                    "--z",
                    "0",
                    "--qx",
                    "0",
                    "--qy",
                    "0",
                    "--qz",
                    "0",
                    "--qw",
                    "1",
                ],
                output="screen",
                parameters=sim_clock_params,
                remappings=tf_remaps,
            )
        )

    actions.append(
        Node(
            package="rviz2",
            executable="rviz2",
            output="screen",
            arguments=["-d", os.path.join(pkg_share, "config", "moveit.rviz")],
            parameters=sim_clock_params
            + [
                robot_description,
                robot_description_semantic,
                ompl_planning_pipeline_config,
                robot_description_kinematics,
            ],
            condition=IfCondition(LaunchConfiguration("use_rviz")),
        )
    )

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value="tiago"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument(
                "allow_execution",
                default_value="true",
                description="If true, start trajectory bridge and enable MoveIt execution (Plan & Execute).",
            ),
            DeclareLaunchArgument(
                "mirror_joint_states",
                default_value="true",
                description="If true, bridge republishes /ns/joint_states for RViz. Use false with Isaac if sim publishes joint_states.",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Set true with Isaac Sim (or any /clock publisher). Aligns MoveIt and RViz with sim-stamped joint_states.",
            ),
            DeclareLaunchArgument(
                "relay_sim_clock",
                default_value="true",
                description="If true and use_sim_time, republish /ns/clock -> /clock (MoveItPy must not remap /clock).",
            ),
            DeclareLaunchArgument(
                "start_robot_state_publisher",
                default_value="true",
                description="Set false when Isaac (or another source) already publishes TF from /ns/joint_states.",
            ),
            DeclareLaunchArgument(
                "start_odom_static_tf",
                default_value="true",
                description="Set false when Isaac already publishes odom/base_footprint (or equivalent) on TF.",
            ),
            DeclareLaunchArgument(
                "start_move_group",
                default_value="true",
                description="Set false when running tiago_moveit_py_grasp_near.py so only one MoveItCpp instance runs.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
