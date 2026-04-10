#!/usr/bin/env python3
"""MoveItPy: plan + execute a small Cartesian offset on the tool frame (default: arm_right), then verify motion."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tempfile
import time
from typing import Dict, List, Tuple

import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped

DEFAULT_URDF = "/media/robotswithai/data1/robots/tiago_hil/tiago_urdf/tiago_dual_simple.urdf"
MOVEIT_PY_NODE_NAME = "tiago_moveit_py"


def _fully_qualified_node_name(namespace: str, node_name: str) -> str:
    ns = namespace.strip("/")
    if ns:
        return f"/{ns}/{node_name}"
    return f"/{node_name}"


def _write_ros2_params_file(
    ros_parameters: dict,
    fully_qualified_node_name: str,
    *,
    use_sim_time_global_wildcard: bool = False,
) -> str:
    """ROS 2 params files must use the namespaced node path (e.g. /tiago/tiago_moveit_py), not bare node_name.

    With MoveItPy + use_sim_time, putting use_sim_time on the node key can trigger
    rclcpp InvalidParameterValueException on qos_overrides./clock (moveit2#2940).
    Supplying use_sim_time only under /** avoids that.
    """
    params = dict(ros_parameters)
    if use_sim_time_global_wildcard:
        params.pop("use_sim_time", None)
        payload = {
            "/**": {"ros__parameters": {"use_sim_time": True}},
            fully_qualified_node_name: {"ros__parameters": params},
        }
    else:
        payload = {fully_qualified_node_name: {"ros__parameters": params}}
    handle = tempfile.NamedTemporaryFile(mode="w", prefix="tiago_moveit_py_", suffix=".yaml", delete=False)
    try:
        yaml.dump(payload, handle, default_flow_style=False, sort_keys=False, allow_unicode=True)
        handle.flush()
        return handle.name
    finally:
        handle.close()

ARM_LEFT_JOINTS: Tuple[str, ...] = (
    "torso_lift_joint",
    "arm_left_1_joint",
    "arm_left_2_joint",
    "arm_left_3_joint",
    "arm_left_4_joint",
    "arm_left_5_joint",
    "arm_left_6_joint",
    "arm_left_7_joint",
)

ARM_RIGHT_JOINTS: Tuple[str, ...] = (
    "torso_lift_joint",
    "arm_right_1_joint",
    "arm_right_2_joint",
    "arm_right_3_joint",
    "arm_right_4_joint",
    "arm_right_5_joint",
    "arm_right_6_joint",
    "arm_right_7_joint",
)

MONITOR_JOINTS_BY_GROUP = {
    "arm_left": ARM_LEFT_JOINTS,
    "arm_right": ARM_RIGHT_JOINTS,
}


def _monitor_joints(group: str) -> Tuple[str, ...]:
    key = group.strip()
    if key not in MONITOR_JOINTS_BY_GROUP:
        raise ValueError(
            f"Unknown --group {group!r}; use arm_left or arm_right (or extend MONITOR_JOINTS_BY_GROUP)."
        )
    return MONITOR_JOINTS_BY_GROUP[key]


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _urdf_mesh_paths_to_file_uri(urdf_xml: str) -> str:
    """MoveIt/resource_retriever often fails on bare absolute paths; file:///... loads reliably."""

    def sub(m: re.Match) -> str:
        p = m.group(1)
        if p.startswith(("package://", "file://", "http://", "https://")):
            return m.group(0)
        if p.startswith("/"):
            return f'filename="file://{p}"'
        return m.group(0)

    return re.sub(r'filename="([^"]+)"', sub, urdf_xml)


def _load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _merge_moveit_node_params(
    pkg_share: str,
    urdf_path: str,
    *,
    allow_trajectory_execution: bool,
    initial_state_wait: float = 30.0,
) -> dict:
    robot_description = {"robot_description": _urdf_mesh_paths_to_file_uri(_load_text(urdf_path))}
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
        "publish_robot_description": False,
        "publish_robot_description_semantic": False,
    }

    planning_scene_monitor_options = {
        "planning_scene_monitor_options": {
            "wait_for_initial_state_timeout": float(initial_state_wait),
        }
    }

    planning_pipelines = {
        "planning_pipelines": {"pipeline_names": ["move_group"], "namespace": ""},
        "default_planning_pipeline": "move_group",
    }

    merged: dict = {}
    for block in (
        robot_description,
        robot_description_semantic,
        robot_description_kinematics,
        robot_description_planning,
        ompl_planning_pipeline_config,
        planning_scene_monitor_parameters,
        planning_scene_monitor_options,
        planning_pipelines,
        {"allow_trajectory_execution": allow_trajectory_execution},
    ):
        merged.update(block)
    if allow_trajectory_execution:
        merged.update(_load_yaml(os.path.join(pkg_share, "config", "moveit_controllers.yaml")))
    merged["moveit_manage_controllers"] = allow_trajectory_execution
    return merged


def _run_check_params(pkg_share: str, urdf_path: str, namespace: str, node_name: str) -> int:
    merged = _merge_moveit_node_params(pkg_share, urdf_path, allow_trajectory_execution=True)
    urdf_txt = merged.get("robot_description") or ""
    srdf_txt = merged.get("robot_description_semantic") or ""
    if len(urdf_txt.strip()) < 50:
        print("check-params: robot_description missing or too short.", file=sys.stderr)
        return 1
    if len(srdf_txt.strip()) < 50 or "<robot" not in srdf_txt:
        print("check-params: robot_description_semantic missing or not valid SRDF XML.", file=sys.stderr)
        return 1
    fq = _fully_qualified_node_name(namespace, node_name)
    path = _write_ros2_params_file(merged, fq)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            roundtrip = yaml.safe_load(handle)
        if fq not in roundtrip or "ros__parameters" not in roundtrip[fq]:
            print(f"check-params: YAML round-trip missing key {fq!r}.", file=sys.stderr)
            return 1
        rp = roundtrip[fq]["ros__parameters"]
        if not (rp.get("robot_description_semantic") or "").strip():
            print("check-params: ros__parameters.robot_description_semantic empty after round-trip.", file=sys.stderr)
            return 1
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    print(
        f"check-params OK: node {fq}, URDF {len(urdf_txt)} bytes, SRDF {len(srdf_txt)} bytes "
        "(MoveItPy must use launch_params_filepaths with this naming; bare node_name breaks namespaces).",
        flush=True,
    )
    return 0


def _run_smoke_init_moveit(pkg_share: str, urdf_path: str, namespace: str, node_name: str) -> int:
    """Construct MoveItPy without sim time or /clock remap (avoids MoveItPy use_sim_time QoS bugs)."""
    from moveit.planning import MoveItPy

    ns = namespace.strip("/")
    merged = _merge_moveit_node_params(
        pkg_share,
        urdf_path,
        allow_trajectory_execution=True,
        initial_state_wait=0.0,
    )
    fq = _fully_qualified_node_name(ns, node_name)
    params_path = _write_ros2_params_file(merged, fq)
    remaps = {"tf": "/tf", "tf_static": "/tf_static"}
    moveit = None
    try:
        moveit = MoveItPy(
            node_name,
            ns,
            launch_params_filepaths=[params_path],
            config_dict=None,
            provide_planning_service=False,
            remappings=remaps,
        )
        rm = moveit.get_robot_model()
        print(f"self-test: MoveItPy init OK (model={rm.name!r}).", flush=True)
        return 0
    finally:
        try:
            os.unlink(params_path)
        except OSError:
            pass
        if moveit is not None:
            moveit.shutdown()


def _joint_vector(state, names: Tuple[str, ...]) -> List[float]:
    jp = state.joint_positions
    return [float(jp[n]) for n in names]


def _l2(a: List[float], b: List[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _quat_mul(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_conj(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, z, w = q
    return (-x, -y, -z, w)


def _quat_norm(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / n, y / n, z / n, w / n)


def _rotate_vec_by_quat_inv(v: Tuple[float, float, float], q: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    qn = _quat_norm(q)
    qinv = _quat_conj(qn)
    vx, vy, vz = v
    rx, ry, rz, _ = _quat_mul(_quat_mul(qinv, (vx, vy, vz, 0.0)), qn)
    return (rx, ry, rz)


def _snapshot_arm(
    psm, model_frame: str, eef_link: str, monitor_joints: Tuple[str, ...]
) -> Tuple[Dict[str, float], PoseStamped]:
    with psm.read_only() as scene:
        st = scene.current_state
        pose = st.get_pose(eef_link)
        stamp = PoseStamped()
        stamp.header.frame_id = model_frame
        stamp.header.stamp.sec = 0
        stamp.header.stamp.nanosec = 0
        stamp.pose = pose
        joints = {n: float(st.joint_positions[n]) for n in monitor_joints}
    return joints, stamp


def _pose_in_frame(psm, model_frame: str, source_in_model: PoseStamped, target_frame: str) -> PoseStamped:
    target = target_frame.strip()
    if not target or target == model_frame:
        return source_in_model

    with psm.read_only() as scene:
        try:
            target_in_model = scene.current_state.get_pose(target)
        except Exception as exc:
            raise ValueError(
                f"Planning frame {target!r} is not available as a robot link in the current state."
            ) from exc

    pm = source_in_model.pose.position
    qm = source_in_model.pose.orientation
    pt = target_in_model.position
    qt = target_in_model.orientation

    delta = (pm.x - pt.x, pm.y - pt.y, pm.z - pt.z)
    q_target = (qt.x, qt.y, qt.z, qt.w)
    q_model = (qm.x, qm.y, qm.z, qm.w)
    p_target = _rotate_vec_by_quat_inv(delta, q_target)
    q_rel = _quat_norm(_quat_mul(_quat_conj(_quat_norm(q_target)), _quat_norm(q_model)))

    out = PoseStamped()
    out.header.frame_id = target
    out.header.stamp.sec = 0
    out.header.stamp.nanosec = 0
    out.pose.position.x = p_target[0]
    out.pose.position.y = p_target[1]
    out.pose.position.z = p_target[2]
    out.pose.orientation.x = q_rel[0]
    out.pose.orientation.y = q_rel[1]
    out.pose.orientation.z = q_rel[2]
    out.pose.orientation.w = q_rel[3]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan/execute arm_right (or arm_left) to a near pose (tool frame + offset in model frame). "
        "Run demo.launch.py with start_move_group:=false so only this node owns MoveItCpp."
    )
    parser.add_argument("--namespace", default="tiago")
    parser.add_argument("--urdf", default=os.environ.get("TIAGO_MOVEIT_URDF", DEFAULT_URDF))
    parser.add_argument("--group", default="arm_right")
    parser.add_argument("--eef-link", default="arm_right_tool_link")
    parser.add_argument("--offset-x", type=float, default=0.05)
    parser.add_argument("--offset-y", type=float, default=0.0)
    parser.add_argument("--offset-z", type=float, default=0.02)
    parser.add_argument("--goal-x", type=float, default=None, help="Absolute goal X in --planning-frame.")
    parser.add_argument("--goal-y", type=float, default=None, help="Absolute goal Y in --planning-frame.")
    parser.add_argument("--goal-z", type=float, default=None, help="Absolute goal Z in --planning-frame.")
    parser.add_argument(
        "--planning-frame",
        default="base_link",
        help="Frame for goal pose offsets. Use base_link for robot-centric planning.",
    )
    parser.add_argument("--use-sim-time", action="store_true", help="Set use_sim_time on the MoveItPy node.")
    parser.add_argument(
        "--isaac-tf",
        action="store_true",
        help="Subscribe to TF on /{namespace}/tf and /{namespace}/tf_static (Isaac) instead of global /tf.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Plan but do not execute (no joint_command stream).",
    )
    parser.add_argument(
        "--verify-tolerance",
        type=float,
        default=0.4,
        help="Max L2 joint error vs last waypoint (rad). Isaac/bridge timing often needs >0.25.",
    )
    parser.add_argument(
        "--check-params",
        action="store_true",
        help="Load URDF/SRDF, write a ROS 2 params YAML with the namespaced node key, verify round-trip (no MoveIt init).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Runs --check-params then a short MoveItPy construct/shutdown (needs moveit_py sourced; no Isaac).",
    )
    args = parser.parse_args()

    pkg_share = get_package_share_directory("tiago_moveit_config")
    if args.check_params and not args.self_test:
        return _run_check_params(pkg_share, args.urdf, args.namespace, MOVEIT_PY_NODE_NAME)

    if args.self_test:
        code = _run_check_params(pkg_share, args.urdf, args.namespace, MOVEIT_PY_NODE_NAME)
        if code != 0:
            return code
        try:
            return _run_smoke_init_moveit(pkg_share, args.urdf, args.namespace, MOVEIT_PY_NODE_NAME)
        except RuntimeError as exc:
            print(f"self-test: MoveItPy failed: {exc}", file=sys.stderr)
            return 1

    try:
        from moveit.planning import MoveItPy, PlanRequestParameters
    except ImportError:
        print(
            "moveit_py is not on PYTHONPATH. Source the MoveIt overlay before this workspace, e.g.\n"
            "  source /media/robotswithai/data1/robots/tiago_hil/moveit2_humble_ws/install/setup.bash",
            file=sys.stderr,
        )
        return 2

    try:
        monitor_joints = _monitor_joints(args.group)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    params = _merge_moveit_node_params(
        pkg_share,
        args.urdf,
        allow_trajectory_execution=not args.plan_only,
        initial_state_wait=120.0 if args.use_sim_time else 30.0,
    )
    sim_global = False
    if args.use_sim_time:
        params["use_sim_time"] = True
        sim_global = True
        print(
            "Sim time: /clock via use_sim_time under /** in params (MoveItPy+Humble workaround). "
            "With Isaac publishing /tiago/clock, run demo.launch with use_sim_time:=true and "
            "relay_sim_clock:=true (or tiago_clock_relay.py). Do not remap /clock on this node.",
            flush=True,
        )

    ns = args.namespace.strip("/")
    fq = _fully_qualified_node_name(ns, MOVEIT_PY_NODE_NAME)
    params_path = _write_ros2_params_file(params, fq, use_sim_time_global_wildcard=sim_global)

    if args.isaac_tf:
        remaps = {"tf": f"/{ns}/tf", "tf_static": f"/{ns}/tf_static"}
    else:
        remaps = {"tf": "/tf", "tf_static": "/tf_static"}

    moveit = None
    try:
        moveit = MoveItPy(
            MOVEIT_PY_NODE_NAME,
            ns,
            launch_params_filepaths=[params_path],
            config_dict=None,
            provide_planning_service=False,
            remappings=remaps,
        )

        rm = moveit.get_robot_model()
        model_frame = rm.model_frame
        psm = moveit.get_planning_scene_monitor()

        joints_before, tool_in_model = _snapshot_arm(psm, model_frame, args.eef_link, monitor_joints)
        try:
            goal = _pose_in_frame(psm, model_frame, tool_in_model, args.planning_frame)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
        has_abs_goal = args.goal_x is not None or args.goal_y is not None or args.goal_z is not None
        if has_abs_goal and not (args.goal_x is not None and args.goal_y is not None and args.goal_z is not None):
            print("Provide all of --goal-x --goal-y --goal-z, or provide none of them.", file=sys.stderr)
            return 2
        if has_abs_goal:
            goal.pose.position.x = float(args.goal_x)
            goal.pose.position.y = float(args.goal_y)
            goal.pose.position.z = float(args.goal_z)
        else:
            goal.pose.position.x += args.offset_x
            goal.pose.position.y += args.offset_y
            goal.pose.position.z += args.offset_z
        print(f"Goal frame: {goal.header.frame_id} (model frame: {model_frame})")

        arm = moveit.get_planning_component(args.group)
        arm.set_start_state_to_current_state()
        arm.set_goal_state(pose_stamped_msg=goal, pose_link=args.eef_link)

        plan_params = PlanRequestParameters(moveit)
        plan_params.planning_pipeline = "move_group"
        plan_params.planning_time = 15.0
        plan_params.max_velocity_scaling_factor = 0.25
        plan_params.max_acceleration_scaling_factor = 0.25

        sol = arm.plan(plan_params)
        if not sol:
            ec = getattr(getattr(sol, "error_code", None), "val", None)
            print(f"Planning failed: error_code={ec}", file=sys.stderr)
            return 1

        traj = sol.trajectory
        n = len(traj)
        last = traj[n - 1]
        target_vec = _joint_vector(last, monitor_joints)
        print(
            f"Planned {n} waypoints; |dq| from start snapshot ({args.group} joints) ~ "
            f"{_l2(target_vec, list(joints_before.values())):.3f} rad"
        )

        if args.plan_only:
            print("Plan-only: skipping execute and joint verification.")
            return 0

        status = moveit.execute(args.group, traj, blocking=True)
        print(f"Execute finished: {status.status}")

        if not status:
            print("Trajectory execution reported failure.", file=sys.stderr)
            return 1

        time.sleep(3.0)
        joints_after, _ = _snapshot_arm(psm, model_frame, args.eef_link, monitor_joints)
        after_vec = [joints_after[n] for n in monitor_joints]
        err = _l2(after_vec, target_vec)
        print(f"Joint L2 vs last waypoint: {err:.3f} rad (tolerance {args.verify_tolerance})")
        if err > args.verify_tolerance:
            print(
                "Verification failed: sim/joint_states may not match the plan, or Isaac did not apply joint_command.",
                file=sys.stderr,
            )
            return 1

        print("Grasp demo OK: monitored joints are close to the final planned waypoint.")
        return 0
    finally:
        try:
            os.unlink(params_path)
        except OSError:
            pass
        if moveit is not None:
            moveit.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
