#!/usr/bin/env python3
"""Fails if AMCL initial_pose rewrites are not numeric (catches ReplaceString/YAW bugs)."""

import os
import sys

import yaml
from launch import LaunchContext
from launch.substitutions import LaunchConfiguration
from launch.utilities import perform_substitutions
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import RewrittenYaml


def _share(*parts: str) -> str:
    ctx = LaunchContext()
    return perform_substitutions(ctx, [FindPackageShare(*parts)])


def main() -> int:
    ctx = LaunchContext()
    ctx.launch_configurations["initial_pose_x"] = "-0.84791"
    ctx.launch_configurations["initial_pose_y"] = "-0.79706"
    ctx.launch_configurations["initial_pose_z"] = "0.0"
    ctx.launch_configurations["initial_pose_yaw"] = "3.141592653589793"

    nav2_yaml = os.path.join(_share("tiago_nav"), "config", "nav2_params.yaml")
    rw = RewrittenYaml(
        source_file=nav2_yaml,
        param_rewrites={
            "amcl.ros__parameters.initial_pose.x": LaunchConfiguration("initial_pose_x"),
            "amcl.ros__parameters.initial_pose.y": LaunchConfiguration("initial_pose_y"),
            "amcl.ros__parameters.initial_pose.z": LaunchConfiguration("initial_pose_z"),
            "amcl.ros__parameters.initial_pose.yaw": LaunchConfiguration("initial_pose_yaw"),
        },
        convert_types=True,
    )
    out_path = rw.perform(ctx)
    try:
        with open(out_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        pose = data["amcl"]["ros__parameters"]["initial_pose"]
        for k in ("x", "y", "z", "yaw"):
            v = pose[k]
            if isinstance(v, str):
                print(f"FAIL: initial_pose.{k} is str {v!r} (AMCL needs numeric types)", file=sys.stderr)
                return 1
        if not isinstance(pose["yaw"], (int, float)):
            print(f"FAIL: initial_pose.yaw type {type(pose['yaw'])}", file=sys.stderr)
            return 1
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass

    print("OK: RewrittenYaml initial_pose fields are numeric")
    return 0


if __name__ == "__main__":
    sys.exit(main())
