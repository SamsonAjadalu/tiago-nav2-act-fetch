#!/usr/bin/env python3
"""FollowJointTrajectory action servers -> JointState on joint_command (+ optional joint_states for RViz)."""

from __future__ import annotations

import threading
from typing import Dict, List, Tuple

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


def _duration_to_sec(d: Duration) -> float:
    return float(d.sec) + float(d.nanosec) * 1e-9


def _interp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class IsaacJointTrajectoryBridge(Node):
    FULL_JOINT_NAMES: Tuple[str, ...] = (
        "wheel_front_left_joint",
        "wheel_front_right_joint",
        "wheel_rear_left_joint",
        "wheel_rear_right_joint",
        "torso_lift_joint",
        "arm_left_1_joint",
        "arm_left_2_joint",
        "arm_left_3_joint",
        "arm_left_4_joint",
        "arm_left_5_joint",
        "arm_left_6_joint",
        "arm_left_7_joint",
        "gripper_left_left_finger_joint",
        "gripper_left_right_finger_joint",
        "arm_right_1_joint",
        "arm_right_2_joint",
        "arm_right_3_joint",
        "arm_right_4_joint",
        "arm_right_5_joint",
        "arm_right_6_joint",
        "arm_right_7_joint",
        "gripper_right_left_finger_joint",
        "gripper_right_right_finger_joint",
        "head_1_joint",
        "head_2_joint",
    )

    def __init__(self) -> None:
        super().__init__("isaac_joint_trajectory_bridge")
        self.declare_parameter("mirror_joint_states", True)
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("idle_publish_rate_hz", 20.0)

        self._mirror = bool(self.get_parameter("mirror_joint_states").value)
        self._cmd_pub = self.create_publisher(JointState, "joint_command", 10)
        self._state_pub = self.create_publisher(JointState, "joint_states", 10)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(JointState, "joint_states", self._on_joint_states, qos)

        self._positions: Dict[str, float] = {}
        for name in self.FULL_JOINT_NAMES:
            self._positions[name] = 0.0
        self._positions["gripper_left_left_finger_joint"] = 0.04
        self._positions["gripper_left_right_finger_joint"] = 0.04
        self._positions["gripper_right_left_finger_joint"] = 0.04
        self._positions["gripper_right_right_finger_joint"] = 0.04

        self._lock = threading.Lock()
        self._executing = False

        idle_hz = float(self.get_parameter("idle_publish_rate_hz").value)
        self.create_timer(1.0 / max(idle_hz, 1.0), self._idle_tick)

        cb = ReentrantCallbackGroup()
        ActionServer(
            self,
            FollowJointTrajectory,
            "arm_left_controller/follow_joint_trajectory",
            self._execute_left,
            callback_group=cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
        )
        ActionServer(
            self,
            FollowJointTrajectory,
            "arm_right_controller/follow_joint_trajectory",
            self._execute_right,
            callback_group=cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
        )

    def _on_joint_states(self, msg: JointState) -> None:
        if self._mirror:
            return
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                if name in self._positions:
                    self._positions[name] = float(pos)

    def _full_joint_state_msg(self) -> JointState:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.FULL_JOINT_NAMES)
        msg.position = [self._positions[j] for j in self.FULL_JOINT_NAMES]
        return msg

    def _idle_tick(self) -> None:
        if self._executing:
            return
        with self._lock:
            msg = self._full_joint_state_msg()
        if self._mirror:
            self._state_pub.publish(msg)
            self._cmd_pub.publish(msg)

    @staticmethod
    def _goal_cb(_goal) -> GoalResponse:
        return GoalResponse.ACCEPT

    @staticmethod
    def _cancel_cb(_goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _execute_left(self, goal_handle):
        return self._run_trajectory(goal_handle)

    def _execute_right(self, goal_handle):
        return self._run_trajectory(goal_handle)

    def _run_trajectory(self, goal_handle) -> FollowJointTrajectory.Result:
        traj = goal_handle.request.trajectory
        joint_names = list(traj.joint_names)
        points = list(traj.points)
        result = FollowJointTrajectory.Result()

        if not joint_names or not points:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            return result

        times_sec = [_duration_to_sec(p.time_from_start) for p in points]
        if times_sec[0] > 1e-6:
            prep = JointTrajectoryPoint()
            prep.time_from_start.sec = 0
            prep.time_from_start.nanosec = 0
            with self._lock:
                for jn in joint_names:
                    prep.positions.append(self._positions.get(jn, 0.0))
            points.insert(0, prep)
            times_sec.insert(0, 0.0)

        publish_hz = float(self.get_parameter("publish_rate_hz").value)
        dt = 1.0 / max(publish_hz, 10.0)
        t_end = times_sec[-1]
        if t_end < 1e-6:
            t_end = 0.5

        self._executing = True
        try:
            t_run = 0.0
            while t_run <= t_end + 1e-6:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                    return result

                seg = 0
                while seg + 1 < len(times_sec) and t_run > times_sec[seg + 1]:
                    seg += 1
                if seg >= len(points) - 1:
                    seg = max(0, len(points) - 2)
                t0, t1 = times_sec[seg], times_sec[seg + 1]
                denom = max(t1 - t0, 1e-9)
                alpha = (t_run - t0) / denom if t_run < t1 else 1.0
                alpha = max(0.0, min(1.0, alpha))
                p_a, p_b = points[seg], points[seg + 1]

                with self._lock:
                    for i, jn in enumerate(joint_names):
                        if jn not in self._positions:
                            continue
                        qa = float(p_a.positions[i]) if i < len(p_a.positions) else self._positions[jn]
                        qb = float(p_b.positions[i]) if i < len(p_b.positions) else qa
                        self._positions[jn] = _interp(qa, qb, alpha)
                    cmd = self._full_joint_state_msg()

                self._cmd_pub.publish(cmd)
                if self._mirror:
                    self._state_pub.publish(cmd)

                fb = FollowJointTrajectory.Feedback()
                fb.header = cmd.header
                fb.joint_names = list(joint_names)
                des = JointTrajectoryPoint()
                for jn in joint_names:
                    if jn in self._positions:
                        des.positions.append(self._positions[jn])
                fb.desired = des
                goal_handle.publish_feedback(fb)

                t_run += dt

            goal_handle.succeed()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            return result
        finally:
            self._executing = False


def main() -> None:
    rclpy.init()
    node = IsaacJointTrajectoryBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
