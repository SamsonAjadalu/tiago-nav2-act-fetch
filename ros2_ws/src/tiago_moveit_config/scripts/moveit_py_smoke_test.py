#!/usr/bin/env python3

import argparse
import sys

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest, PlanningOptions


class MoveItPySmoke(Node):
    def __init__(self, namespace: str) -> None:
        super().__init__("moveit_py_smoke_test")
        action_name = f"/{namespace}/move_action"
        self._client = ActionClient(self, MoveGroup, action_name)

    def plan_to_joint_goal(self, group_name: str) -> bool:
        if not self._client.wait_for_server(timeout_sec=20.0):
            self.get_logger().error("MoveGroup action server not available.")
            return False

        joint_targets_by_group = {
            "arm_left": {
                "torso_lift_joint": 0.12,
                "arm_left_1_joint": 0.30,
                "arm_left_2_joint": -0.40,
                "arm_left_3_joint": 1.20,
                "arm_left_4_joint": 1.30,
                "arm_left_5_joint": 0.00,
                "arm_left_6_joint": 0.20,
                "arm_left_7_joint": 0.00,
            },
            "arm_right": {
                "torso_lift_joint": 0.12,
                "arm_right_1_joint": -0.30,
                "arm_right_2_joint": -0.40,
                "arm_right_3_joint": 1.20,
                "arm_right_4_joint": 1.30,
                "arm_right_5_joint": 0.00,
                "arm_right_6_joint": 0.20,
                "arm_right_7_joint": 0.00,
            },
        }
        joint_targets = joint_targets_by_group.get(group_name)
        if joint_targets is None:
            self.get_logger().error(f"No joint goal template for group '{group_name}'. Use arm_left or arm_right.")
            return False

        constraints = Constraints()
        constraints.name = "joint_goal"
        for joint_name, position in joint_targets.items():
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = joint_name
            joint_constraint.position = position
            joint_constraint.tolerance_above = 0.02
            joint_constraint.tolerance_below = 0.02
            joint_constraint.weight = 1.0
            constraints.joint_constraints.append(joint_constraint)

        request = MotionPlanRequest()
        request.group_name = group_name
        request.num_planning_attempts = 3
        request.allowed_planning_time = 5.0
        request.max_velocity_scaling_factor = 0.2
        request.max_acceleration_scaling_factor = 0.2
        request.goal_constraints = [constraints]

        goal = MoveGroup.Goal()
        goal.request = request
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = True

        goal_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, goal_future)
        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("MoveGroup goal rejected.")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        if result is None:
            self.get_logger().error("MoveGroup result is empty.")
            return False

        error_code = result.result.error_code.val
        if error_code == 1:
            self.get_logger().info("MoveIt planning succeeded.")
            return True

        self.get_logger().error(f"MoveIt planning failed with error code: {error_code}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="tiago")
    parser.add_argument("--group", default="arm_left")
    args = parser.parse_args()

    rclpy.init()
    node = MoveItPySmoke(args.namespace)
    ok = node.plan_to_joint_goal(args.group)
    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
