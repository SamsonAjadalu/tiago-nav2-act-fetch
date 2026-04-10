"""
Standalone TIAGo runtime joint/EE jog tool for demos (house_with_can2).

This script is intentionally independent from ACT inference scripts.
It loads the house scene and TIAGo, then allows live keyboard control:

- Joint jog: torso, wrist joints, gripper open/close
- EE jog (IK): move end-effector target in XYZ
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False, "enable_ros2": True})

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tiago_repo_paths import (
    default_tiago_usd_path,
    default_urdf_path,
    default_yaml_path,
    house_usd_v2_path,
)

import carb
import omni.appwindow
import omni.timeline
import omni.usd
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Gf, UsdGeom


HOUSE_USD_PATH = house_usd_v2_path()
TIAGO_USD_PATH = default_tiago_usd_path()
TIAGO_URDF_PATH = default_urdf_path()
TIAGO_YAML_PATH = default_yaml_path()
RIGHT_EE_FRAME_NAME = os.environ.get("TIAGO_LULA_EE_FRAME", "arm_right_tool_link")

JOG_JOINT_STEP_RAD = float(os.environ.get("TIAGO_JOG_JOINT_STEP_RAD", "0.01"))
JOG_TORSO_STEP_M = float(os.environ.get("TIAGO_JOG_TORSO_STEP_M", "0.004"))
JOG_EE_STEP_M = float(os.environ.get("TIAGO_JOG_EE_STEP_M", "0.004"))
GRIPPER_STEP = float(os.environ.get("TIAGO_JOG_GRIPPER_STEP", "0.003"))

GRIPPER_OPEN = 0.045
GRIPPER_CLOSED = 0.0

DEFAULT_JOINT_POSITIONS = {
    "arm_left_1_joint": 0.1032,
    "arm_left_2_joint": -0.1531,
    "arm_left_3_joint": 2.8088,
    "arm_left_4_joint": 1.5408,
    "arm_left_5_joint": 1.3844,
    "arm_left_6_joint": -0.1917,
    "arm_left_7_joint": 0.71,
    "arm_right_1_joint": 0.1032,
    "arm_right_2_joint": -0.1531,
    "arm_right_3_joint": 2.8088,
    "arm_right_4_joint": 1.5408,
    "arm_right_5_joint": 1.3844,
    "arm_right_6_joint": -0.1917,
    "arm_right_7_joint": 0.71,
    "gripper_left_left_finger_joint": 0.045,
    "gripper_left_right_finger_joint": 0.045,
    "gripper_right_left_finger_joint": 0.045,
    "gripper_right_right_finger_joint": 0.045,
    "head_1_joint": -0.1997,
    "head_2_joint": -0.6589,
    "torso_lift_joint": 0.35,
}


def _find_prim_path_by_name(stage, prim_name: str) -> Optional[str]:
    for prim in stage.Traverse():
        if prim.IsValid() and prim.GetName() == prim_name:
            return str(prim.GetPath())
    return None


def _find_first_path_with_any_token(stage, tokens: list[str]) -> Optional[str]:
    toks = [t.lower() for t in tokens]
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        p = str(prim.GetPath())
        if any(t in p.lower() for t in toks):
            return p
    return None


def _apply_house_transform_like_recorder(stage) -> None:
    legacy_tx = float(os.environ.get("TIAGO_HOUSE_LEGACY_TX", "0.4"))
    legacy_ty = float(os.environ.get("TIAGO_HOUSE_LEGACY_TY", "-1.44"))
    legacy_tz = float(os.environ.get("TIAGO_HOUSE_LEGACY_TZ", "0.0"))
    legacy_yaw_deg = float(os.environ.get("TIAGO_HOUSE_LEGACY_YAW_DEG", "162.0"))
    place_env_not_robot = os.environ.get("TIAGO_HOUSE_ENV_CARRIES_LEGACY_ROBOT_POSE", "1").strip() not in (
        "0",
        "false",
        "False",
        "no",
        "NO",
    )
    if not place_env_not_robot:
        return
    house_prim = stage.GetPrimAtPath("/World/HouseScene")
    if not house_prim.IsValid():
        return
    mt = Gf.Matrix4d(1.0)
    mt.SetTranslate(Gf.Vec3d(legacy_tx, legacy_ty, legacy_tz))
    mr = Gf.Matrix4d(1.0)
    mr.SetRotate(Gf.Rotation(Gf.Vec3d.ZAxis(), legacy_yaw_deg))
    m_inv = (mt * mr).GetInverse()
    xf = UsdGeom.Xformable(house_prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(m_inv)


def _print_keymap() -> None:
    msg = """
[TIAGO-JOG] Controls
  T/G : torso up/down
  Y/H : arm_right_6 +/- (wrist pitch-ish)
  U/J : arm_right_7 +/- (wrist roll-ish)
  Z/X : right gripper open/close
  I/K : EE +X/-X (IK target)
  L/O : EE +Y/-Y (IK target)
  P/; : EE +Z/-Z (IK target)
  M   : toggle IK EE jogging on/off
  R   : reset to default joint pose
  SPACE: pause/resume timeline
"""
    print(msg, flush=True)


def main() -> None:
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()

    if not Path(HOUSE_USD_PATH).exists():
        raise FileNotFoundError(f"Scene USD not found: {HOUSE_USD_PATH}")
    add_reference_to_stage(HOUSE_USD_PATH, "/World/HouseScene")
    _apply_house_transform_like_recorder(stage)

    add_reference_to_stage(TIAGO_USD_PATH, "/World/Tiago")
    tiago_root = stage.GetPrimAtPath("/World/Tiago")
    if tiago_root.IsValid():
        xform = UsdGeom.Xformable(tiago_root)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        xform.AddRotateZOp().Set(0.0)

    tiago = world.scene.add(Robot(prim_path="/World/Tiago/tiago_dual_functional", name="tiago"))

    world.reset()
    for _ in range(30):
        world.step(render=False)

    q = tiago.get_joint_positions()
    if q is None:
        raise RuntimeError("Articulation not initialized.")
    q0 = np.array(q, dtype=float)
    for jn, v in DEFAULT_JOINT_POSITIONS.items():
        if jn in tiago.dof_names:
            q0[tiago.dof_names.index(jn)] = v
    for _ in range(60):
        tiago.apply_action(ArticulationAction(joint_positions=q0))
        world.step(render=False)

    lula = LulaKinematicsSolver(robot_description_path=TIAGO_YAML_PATH, urdf_path=TIAGO_URDF_PATH)
    ik_solver = ArticulationKinematicsSolver(tiago, lula, RIGHT_EE_FRAME_NAME)
    ee0, _ = ik_solver.compute_end_effector_pose()
    ee_target = np.array(ee0, dtype=float)

    idx = {n: i for i, n in enumerate(tiago.dof_names)}
    required = ["torso_lift_joint", "arm_right_6_joint", "arm_right_7_joint", "gripper_right_left_finger_joint", "gripper_right_right_finger_joint"]
    missing = [n for n in required if n not in idx]
    if missing:
        raise RuntimeError(f"Missing required joints: {missing}")

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard() if app_window else None
    input_iface = carb.input.acquire_input_interface()
    pressed: Set[Any] = set()
    ik_jog_enabled = True
    request_reset = False

    def _on_keyboard(event: carb.input.KeyboardEvent, *_args: Any) -> bool:
        nonlocal ik_jog_enabled, request_reset
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            pressed.add(event.input)
            if event.input == carb.input.KeyboardInput.M:
                ik_jog_enabled = not ik_jog_enabled
                print(f"[TIAGO-JOG] IK EE jog: {'ON' if ik_jog_enabled else 'OFF'}", flush=True)
            elif event.input == carb.input.KeyboardInput.R:
                request_reset = True
            elif event.input == carb.input.KeyboardInput.SPACE:
                timeline = omni.timeline.get_timeline_interface()
                if timeline.is_playing():
                    timeline.pause()
                else:
                    timeline.play()
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            pressed.discard(event.input)
        return True

    sub = None
    if keyboard is not None:
        sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_keyboard)
    else:
        print("[TIAGO-JOG] No keyboard detected.", flush=True)

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()

    _print_keymap()
    frame = 0
    try:
        while simulation_app.is_running():
            world.step(render=True)
            frame += 1

            q_now = tiago.get_joint_positions()
            if q_now is None:
                continue
            q_cmd = np.array(q_now, dtype=float)

            if request_reset:
                request_reset = False
                for jn, v in DEFAULT_JOINT_POSITIONS.items():
                    if jn in idx:
                        q_cmd[idx[jn]] = v
                ee_now, _ = ik_solver.compute_end_effector_pose()
                ee_target[:] = np.array(ee_now, dtype=float)
                print("[TIAGO-JOG] Reset to default pose.", flush=True)

            # Joint jogs
            if carb.input.KeyboardInput.T in pressed:
                q_cmd[idx["torso_lift_joint"]] += JOG_TORSO_STEP_M
            if carb.input.KeyboardInput.G in pressed:
                q_cmd[idx["torso_lift_joint"]] -= JOG_TORSO_STEP_M
            if carb.input.KeyboardInput.Y in pressed:
                q_cmd[idx["arm_right_6_joint"]] += JOG_JOINT_STEP_RAD
            if carb.input.KeyboardInput.H in pressed:
                q_cmd[idx["arm_right_6_joint"]] -= JOG_JOINT_STEP_RAD
            if carb.input.KeyboardInput.U in pressed:
                q_cmd[idx["arm_right_7_joint"]] += JOG_JOINT_STEP_RAD
            if carb.input.KeyboardInput.J in pressed:
                q_cmd[idx["arm_right_7_joint"]] -= JOG_JOINT_STEP_RAD
            if carb.input.KeyboardInput.Z in pressed:
                q_cmd[idx["gripper_right_left_finger_joint"]] = min(
                    GRIPPER_OPEN, q_cmd[idx["gripper_right_left_finger_joint"]] + GRIPPER_STEP
                )
                q_cmd[idx["gripper_right_right_finger_joint"]] = min(
                    GRIPPER_OPEN, q_cmd[idx["gripper_right_right_finger_joint"]] + GRIPPER_STEP
                )
            if carb.input.KeyboardInput.X in pressed:
                q_cmd[idx["gripper_right_left_finger_joint"]] = max(
                    GRIPPER_CLOSED, q_cmd[idx["gripper_right_left_finger_joint"]] - GRIPPER_STEP
                )
                q_cmd[idx["gripper_right_right_finger_joint"]] = max(
                    GRIPPER_CLOSED, q_cmd[idx["gripper_right_right_finger_joint"]] - GRIPPER_STEP
                )

            # EE jog via IK
            d = np.zeros(3, dtype=float)
            if carb.input.KeyboardInput.I in pressed:
                d[0] += JOG_EE_STEP_M
            if carb.input.KeyboardInput.K in pressed:
                d[0] -= JOG_EE_STEP_M
            if carb.input.KeyboardInput.L in pressed:
                d[1] += JOG_EE_STEP_M
            if carb.input.KeyboardInput.O in pressed:
                d[1] -= JOG_EE_STEP_M
            if carb.input.KeyboardInput.P in pressed:
                d[2] += JOG_EE_STEP_M
            if carb.input.KeyboardInput.SEMICOLON in pressed:
                d[2] -= JOG_EE_STEP_M

            if ik_jog_enabled and np.linalg.norm(d) > 0:
                ee_target += d
                ik_action, ok = ik_solver.compute_inverse_kinematics(target_position=ee_target)
                if ok and ik_action.joint_positions is not None:
                    ik_q = np.array(ik_action.joint_positions, dtype=float)
                    if hasattr(ik_action, "joint_indices") and ik_action.joint_indices is not None:
                        for i, j in enumerate(np.array(ik_action.joint_indices, dtype=int)):
                            if i < len(ik_q) and j < len(q_cmd):
                                q_cmd[j] = ik_q[i]
                    else:
                        q_cmd[: len(ik_q)] = ik_q

            tiago.apply_action(ArticulationAction(joint_positions=q_cmd))

            if frame % 120 == 0:
                ee_pos, _ = ik_solver.compute_end_effector_pose()
                print(
                    f"[TIAGO-JOG] ee={np.array(ee_pos, dtype=float).round(4).tolist()} "
                    f"torso={q_cmd[idx['torso_lift_joint']]:.3f} "
                    f"w6={q_cmd[idx['arm_right_6_joint']]:.3f} w7={q_cmd[idx['arm_right_7_joint']]:.3f}",
                    flush=True,
                )
    finally:
        if sub is not None:
            try:
                sub = None
            except Exception:
                pass
        simulation_app.close()


if __name__ == "__main__":
    main()

