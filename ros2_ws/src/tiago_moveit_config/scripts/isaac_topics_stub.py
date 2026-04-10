#!/usr/bin/env python3
"""Publish /ns/clock, /ns/joint_states, /ns/tf_static; mirror /ns/joint_command into joint_states (Isaac stand-in for HIL tests)."""

from __future__ import annotations

import threading

import rclpy
from builtin_interfaces.msg import Time as BuiltinTime
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from tf2_ros import StaticTransformBroadcaster


class IsaacTopicsStub(Node):
    def __init__(self) -> None:
        super().__init__("isaac_topics_stub")
        self.declare_parameter("namespace", "tiago")
        ns = str(self.get_parameter("namespace").value).strip("/")
        self._ns = ns
        prefix = f"/{ns}/"

        self._clock_pub = self.create_publisher(Clock, prefix + "clock", 10)
        self._js_pub = self.create_publisher(JointState, prefix + "joint_states", 10)

        self._names = [
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
        ]
        pos = {n: 0.0 for n in self._names}
        # SRDF group_state "home" / "home_right" (tiago_dual.srdf); wrist_right_ft vs gripper_right mesh overlap suppressed in SRDF.
        pos.update(
            {
                "torso_lift_joint": 0.35,
                "arm_left_1_joint": 0.2,
                "arm_left_2_joint": -0.3,
                "arm_left_3_joint": 1.2,
                "arm_left_4_joint": 1.4,
                "arm_left_5_joint": 0.0,
                "arm_left_6_joint": 0.2,
                "arm_left_7_joint": 0.0,
                "arm_right_1_joint": -0.2,
                "arm_right_2_joint": -0.3,
                "arm_right_3_joint": 1.2,
                "arm_right_4_joint": 1.4,
                "arm_right_5_joint": 0.0,
                "arm_right_6_joint": 0.2,
                "arm_right_7_joint": 0.0,
                "gripper_left_left_finger_joint": 0.04,
                "gripper_left_right_finger_joint": 0.04,
                "gripper_right_left_finger_joint": 0.04,
                "gripper_right_right_finger_joint": 0.04,
            }
        )
        self._positions = [pos[n] for n in self._names]
        self._name_to_i = {n: i for i, n in enumerate(self._names)}
        self._lock = threading.Lock()
        # Avoid TF buffer "jump back" if other nodes briefly publish /clock before sim time dominates.
        self._sim_time_sec = 100_000.0
        self._timer_period = 0.02

        t = TransformStamped()
        t.header.stamp = BuiltinTime(sec=0, nanosec=0)
        t.header.frame_id = "odom"
        t.child_frame_id = "base_footprint"
        t.transform.rotation.w = 1.0
        self._tf_static = StaticTransformBroadcaster(self)
        self._tf_static.sendTransform(t)

        self.create_subscription(JointState, prefix + "joint_command", self._on_joint_command, 10)
        self.create_timer(self._timer_period, self._tick)

    def _on_joint_command(self, msg: JointState) -> None:
        with self._lock:
            for name, p in zip(msg.name, msg.position):
                i = self._name_to_i.get(name)
                if i is not None:
                    self._positions[i] = float(p)

    def _tick(self) -> None:
        self._sim_time_sec += self._timer_period
        clk = Clock()
        s = int(self._sim_time_sec)
        clk.clock.sec = s
        clk.clock.nanosec = int((self._sim_time_sec - s) * 1_000_000_000)
        self._clock_pub.publish(clk)
        with self._lock:
            js = JointState()
            js.header.stamp = clk.clock
            js.name = list(self._names)
            js.position = list(self._positions)
        self._js_pub.publish(js)


def main() -> None:
    rclpy.init()
    node = IsaacTopicsStub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
