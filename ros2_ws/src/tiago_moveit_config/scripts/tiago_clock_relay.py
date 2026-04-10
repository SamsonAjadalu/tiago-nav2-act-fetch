#!/usr/bin/env python3
"""Republish simulation clock from /{ns}/clock to /clock so use_sim_time nodes avoid /clock remaps (MoveItPy QoS bug)."""

from __future__ import annotations

import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock


class ClockRelay(Node):
    def __init__(self, src: str, dst: str) -> None:
        super().__init__("tiago_clock_relay")
        sub_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        pub_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._pub = self.create_publisher(Clock, dst, pub_qos)
        self.create_subscription(Clock, src, self._on_clock, sub_qos)

    def _on_clock(self, msg: Clock) -> None:
        self._pub.publish(msg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/tiago/clock", help="Input Clock topic (Isaac namespaced clock).")
    parser.add_argument("--dst", default="/clock", help="Output Clock topic (ROS 2 use_sim_time default).")
    args, _unknown = parser.parse_known_args()

    rclpy.init()
    node = ClockRelay(args.src, args.dst)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
