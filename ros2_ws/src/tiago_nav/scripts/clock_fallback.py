#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock


class ClockFallback(Node):
    def __init__(self) -> None:
        super().__init__("clock_fallback")
        self.declare_parameter("clock_source_topic", "/tiago/clock")
        self.source_topic = str(self.get_parameter("clock_source_topic").value)

        self.publisher = self.create_publisher(Clock, "/clock", 10)
        self.subscription = self.create_subscription(
            Clock,
            self.source_topic,
            self._source_clock_callback,
            10,
        )

        self.last_source_msg_time = 0.0
        self.source_seen = False
        self.fallback_after_s = 0.5
        self.timer = self.create_timer(1.0 / 30.0, self._publish_fallback_if_needed)

        self.get_logger().info(
            f"Clock relay started: source={self.source_topic} -> target=/clock; "
            "fallback=wall-time when source is idle"
        )

    def _source_clock_callback(self, msg: Clock) -> None:
        self.last_source_msg_time = time.monotonic()
        self.source_seen = True
        self.publisher.publish(msg)

    def _publish_fallback_if_needed(self) -> None:
        if self.source_seen:
            return

        if (time.monotonic() - self.last_source_msg_time) <= self.fallback_after_s:
            return

        now_ns = self.get_clock().now().nanoseconds
        msg = Clock()
        msg.clock.sec = int(now_ns // 1_000_000_000)
        msg.clock.nanosec = int(now_ns % 1_000_000_000)
        self.publisher.publish(msg)


def main() -> None:
    rclpy.init()
    node = ClockFallback()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
