#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class FakeJointStatePublisher(Node):
    def __init__(self) -> None:
        super().__init__("fake_joint_state_publisher")
        self._pub = self.create_publisher(JointState, "joint_states", 10)
        self._timer = self.create_timer(0.05, self._tick)
        self._joint_names = [
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
        self._positions = [0.0] * len(self._joint_names)
        self._positions[self._joint_names.index("gripper_left_left_finger_joint")] = 0.04
        self._positions[self._joint_names.index("gripper_left_right_finger_joint")] = 0.04
        self._positions[self._joint_names.index("gripper_right_left_finger_joint")] = 0.04
        self._positions[self._joint_names.index("gripper_right_right_finger_joint")] = 0.04

    def _tick(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self._joint_names
        msg.position = self._positions
        self._pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = FakeJointStatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
