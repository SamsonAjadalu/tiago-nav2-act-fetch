"""
ACT inference with Phase-A **safety shell** (no RL): near-target motion guard, post-close
failure check, and kick detection. Does not replace ``tiago_inference_act.py``.

- **Space**: toggle inference (same as base script).
- On **safety trip**: inference turns off, ACT queue resets; press Space again to retry.

Launcher: extra/isaacsim/run_inference_safe.sh. ISAACSIM_ROOT must be set in extra/local_paths.sh.
"""

from __future__ import annotations

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "enable_ros2": False,
    }
)

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import carb
import numpy as np
import omni.appwindow
import omni.replicator.core as rep
import omni.timeline
import omni.usd
import torch
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.physx.scripts import utils as physx_utils
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from tiago_repo_paths import (
    default_act_policy_dir,
    default_table_objects_usd_path,
    default_tiago_usd_path,
    default_urdf_path,
    default_yaml_path,
)

# ============================================================================
# Configuration (same as base inference)
# ============================================================================

TIAGO_USD_PATH = default_tiago_usd_path()
TABLE_OBJECTS_USD_PATH = default_table_objects_usd_path()
TIAGO_URDF_PATH = default_urdf_path()
TIAGO_YAML_PATH = default_yaml_path()

ROBOT_PRIM_PATH = "/World/Tiago/tiago_dual_functional"
RIGHT_EE_FRAME_NAME = "gripper_right_grasping_frame"

POLICY_DIR = default_act_policy_dir()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CAM_WIDTH = 1280
CAM_HEIGHT = 720

GRIPPER_JOINT_NAMES = [
    "gripper_right_left_finger_joint",
    "gripper_right_right_finger_joint",
]
GRIPPER_OPEN_POSITION = 0.045
GRIPPER_CLOSED_POSITION = 0.0

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
    "torso_lift_joint": 0.178,
}

ATTACH_DISTANCE_MAX = 0.12
# Grasp weld: use **min(left, right) finger joint** (same units as GRIPPER_OPEN_POSITION), not VR-style trigger.
# Attach when min crosses **below** ATTACH_FINGER_MIN_RAD (from >=); detach when min **above** DETACH_FINGER_MIN_RAD.
ATTACH_FINGER_MIN_RAD = 0.04
# Detach when min finger opens above this; must exceed ATTACH_FINGER_MIN_RAD to avoid immediate post-weld detach.
DETACH_FINGER_MIN_RAD = 0.042
# Optional keyboard force-attach (demo aid): V tries to weld nearest pickable to gripper.
ATTACH_ON_V_KEY = os.environ.get("TIAGO_ATTACH_ON_V_KEY", "0").strip() in ("1", "true", "True", "yes", "YES")
ATTACH_ON_V_DISTANCE_MAX = float(os.environ.get("TIAGO_ATTACH_ON_V_DISTANCE_MAX", "0.40"))

PICKABLE_OBJECT_PATHS: List[str] = [
    "/World/Table/_05_tomato_soup_can/_05_tomato_soup_can",
    "/World/Table/_06_mustard_bottle/_06_mustard_bottle",
    "/World/PickBox",
]

TOGGLE_DEBOUNCE_FRAMES = 30

# After attach, stop ACT and move to a deterministic carry pose.
POST_GRASP_STOP_INFERENCE = True
POST_GRASP_MOVE_FRAMES = 120
AUTO_DETACH_ENABLED = False
POST_GRASP_CARRY_TARGETS: Dict[str, float] = {
    "arm_right_1_joint": 0.15,
    "arm_right_2_joint": -0.45,
    "arm_right_3_joint": 2.35,
    "arm_right_4_joint": 1.75,
    "arm_right_5_joint": 1.10,
    "arm_right_6_joint": -0.35,
    "arm_right_7_joint": 0.65,
    "torso_lift_joint": 0.25,
    "gripper_right_left_finger_joint": 0.030,
    "gripper_right_right_finger_joint": 0.030,
}

# ============================================================================
# Phase A — safety limits (scene-dependent)
# ============================================================================

# Near-target guard: EE within this distance (m) of nearest pickable → limit joint step size.
NEAR_GUARD_DISTANCE_M = 0.22
# Max absolute joint delta (rad) per physics step for right arm joints when in guard zone.
MAX_ARM_DELTA_NEAR_RAD = 0.025
# Max joint delta (rad) per step outside guard zone.
MAX_ARM_DELTA_FAR_RAD = 0.12

# After min finger crosses into "closed" band, verify attach after N frames (drift check).
POST_CLOSE_VERIFY_FRAMES = 24
# If still not attached: any pickable moved this far (m) from session baseline → fail.
OBJECT_DRIFT_FAIL_M = 0.09

# Kick: while not attached, object step displacement (m) or inferred speed exceeds threshold.
KICK_STEP_DISPLACE_M = 0.018
KICK_SPEED_M_S = 0.35
# Physics dt for speed estimate (match simulation timestep if changed).
PHYSICS_DT_ASSUMED = 1.0 / 60.0

# Log gripper when virtual trigger changes by at least this much (0–1 scale).
GRIPPER_LOG_TRIGGER_DELTA = 0.02


def _gripper_debug_line(
    frame_count: int,
    joint_np: np.ndarray,
    gripper_joint_indices: List[int],
    min_finger_rad: float,
    trigger_legacy: float,
    attached_object: Optional[str],
    joint_prim_path: Optional[str],
) -> str:
    parts = [f"{float(joint_np[i]):.4f}" for i in gripper_joint_indices]
    fingers = ",".join(parts) if parts else "n/a"
    att = attached_object if attached_object else "none"
    fixed = "yes" if joint_prim_path else "no"
    return (
        f"[GRIPPER] frame={frame_count} min_finger_rad={min_finger_rad:.4f} "
        f"fingers_rad=[{fingers}] trigger_legacy={trigger_legacy:.3f} "
        f"attached={att} fixed_joint={fixed}"
    )


def find_gemini2_rgb_camera_prim_path(stage) -> Optional[str]:
    try:
        best: Optional[str] = None
        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            lower = path.lower()
            if ("gemini2" in lower or "orbbec_gemini2" in lower) and "stream_rgb" in lower:
                if lower.endswith("stream_rgb"):
                    return path
                best = path
        if best is not None:
            return best
        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            if path.lower().endswith("stream_rgb"):
                return path
    except Exception:
        return None
    return None


def setup_rgb_annotator(camera_prim_path: str) -> Any:
    render_product = rep.create.render_product(camera_prim_path, (CAM_WIDTH, CAM_HEIGHT))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(render_product)
    return rgb


def get_rgb_hwc_uint8(rgb_annotator: Any) -> Optional[np.ndarray]:
    rep.orchestrator.step()
    data = rgb_annotator.get_data()
    if data is None:
        return None
    arr = np.asarray(data)
    if arr.ndim != 3:
        return None
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _setup_stage(world: World) -> None:
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()

    ROBOT_POSITION = (0.0, 0.0, 0.0)
    TABLE_POSITION = (1.1, 0.0, 0.0)

    if os.path.exists(TIAGO_USD_PATH):
        add_reference_to_stage(TIAGO_USD_PATH, "/World/Tiago")
        tiago_prim = stage.GetPrimAtPath("/World/Tiago")
        if tiago_prim.IsValid():
            xform = UsdGeom.Xformable(tiago_prim)
            xform_op = xform.AddTranslateOp()
            xform_op.Set(Gf.Vec3d(*ROBOT_POSITION))

    if TABLE_OBJECTS_USD_PATH and os.path.exists(TABLE_OBJECTS_USD_PATH):
        add_reference_to_stage(TABLE_OBJECTS_USD_PATH, "/World/Table")
        table_prim = stage.GetPrimAtPath("/World/Table")
        if table_prim.IsValid():
            xform = UsdGeom.Xformable(table_prim)
            xform_op = xform.AddTranslateOp()
            xform_op.Set(Gf.Vec3d(*TABLE_POSITION))
            rotate_op = xform.AddRotateZOp()
            rotate_op.Set(90.0)


def setup_robot_and_ik(world: World) -> Tuple[Robot, ArticulationKinematicsSolver, List[int]]:
    tiago = world.scene.add(Robot(prim_path=ROBOT_PRIM_PATH, name="tiago"))

    world.reset()
    world.reset()
    for _ in range(30):
        world.step(render=False)

    joint_positions = tiago.get_joint_positions()
    if joint_positions is None:
        raise RuntimeError("[TIAGO-INFERENCE-SAFE] Articulation not initialized!")

    all_joint_names = tiago.dof_names
    restored_positions = np.array(joint_positions, dtype=float)

    for joint_name, target_position in DEFAULT_JOINT_POSITIONS.items():
        if joint_name in all_joint_names:
            joint_idx = all_joint_names.index(joint_name)
            restored_positions[joint_idx] = target_position

    for _ in range(50):
        tiago.apply_action(ArticulationAction(joint_positions=restored_positions))
        world.step(render=False)

    gripper_joint_indices: List[int] = []
    for gripper_name in GRIPPER_JOINT_NAMES:
        if gripper_name in all_joint_names:
            gripper_joint_indices.append(all_joint_names.index(gripper_name))

    lula_kin = LulaKinematicsSolver(robot_description_path=TIAGO_YAML_PATH, urdf_path=TIAGO_URDF_PATH)
    art_kin_solver = ArticulationKinematicsSolver(tiago, lula_kin, RIGHT_EE_FRAME_NAME)

    ee_pos, _ = art_kin_solver.compute_end_effector_pose()
    carb.log_info(f"[TIAGO-INFERENCE-SAFE] EE frame: {RIGHT_EE_FRAME_NAME}, test_pos={ee_pos}")

    return tiago, art_kin_solver, gripper_joint_indices


def find_gripper_rigidbody_prim(stage) -> Optional[str]:
    best = None
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        lower = path.lower()
        if "gripper_right_link" in lower and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return path
        if "gripper_right_link" in lower:
            best = best or path
    return best


def get_world_pos(stage, prim_path: str) -> Optional[np.ndarray]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    xform = UsdGeom.Xformable(prim)
    time = Usd.TimeCode.Default()
    transform = xform.ComputeLocalToWorldTransform(time)
    translation = transform.ExtractTranslation()
    return np.array([translation[0], translation[1], translation[2]])


def nearest_pickable_distance_m(stage, ee_pos: np.ndarray) -> Tuple[float, Optional[str]]:
    best_d = float("inf")
    best_p: Optional[str] = None
    for path in PICKABLE_OBJECT_PATHS:
        pos = get_world_pos(stage, path)
        if pos is None:
            continue
        d = float(np.linalg.norm(ee_pos - pos))
        if d < best_d:
            best_d = d
            best_p = path
    return best_d, best_p


def find_closest_pickable(stage, ee_pos: np.ndarray) -> Optional[Tuple[str, float]]:
    closest = None
    closest_dist = 1e9
    for path in PICKABLE_OBJECT_PATHS:
        pos = get_world_pos(stage, path)
        if pos is None:
            continue
        dist = float(np.linalg.norm(ee_pos - pos))
        if dist < closest_dist and dist <= ATTACH_DISTANCE_MAX:
            closest = path
            closest_dist = dist
    if closest is None:
        return None
    return closest, closest_dist


def virtual_trigger_from_gripper_joints(joint_positions: np.ndarray, gripper_joint_indices: List[int]) -> float:
    """Legacy 0–1 squeeze metric (VR teleop style); attach/detach no longer use this."""
    if not gripper_joint_indices:
        return 0.0
    vals = [float(joint_positions[i]) for i in gripper_joint_indices]
    mean_f = float(np.mean(vals))
    t = 1.0 - (mean_f / GRIPPER_OPEN_POSITION)
    return float(np.clip(t, 0.0, 1.0))


def min_gripper_finger_rad(joint_positions: np.ndarray, gripper_joint_indices: List[int]) -> float:
    """Smaller joint value = more closed (TIAGo finger joints, same units as GRIPPER_OPEN_POSITION)."""
    if not gripper_joint_indices:
        return float(GRIPPER_OPEN_POSITION)
    return min(float(joint_positions[i]) for i in gripper_joint_indices)


def arm_right_dof_indices(all_joint_names: List[str]) -> List[int]:
    return [i for i, n in enumerate(all_joint_names) if n.startswith("arm_right_") and n.endswith("_joint")]


def clamp_action_vs_previous(
    action_np: np.ndarray,
    previous_np: np.ndarray,
    arm_indices: List[int],
    max_delta_near: float,
    max_delta_far: float,
    near_zone: bool,
) -> np.ndarray:
    out = np.array(action_np, dtype=float, copy=True)
    cap = max_delta_near if near_zone else max_delta_far
    for i in arm_indices:
        lo = previous_np[i] - cap
        hi = previous_np[i] + cap
        out[i] = float(np.clip(out[i], lo, hi))
    return out


def snapshot_pickable_positions(stage) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for path in PICKABLE_OBJECT_PATHS:
        pos = get_world_pos(stage, path)
        if pos is not None:
            out[path] = pos.copy()
    return out


def build_joint_target_from_map(
    dof_names: List[str],
    base_joint_positions: np.ndarray,
    targets: Dict[str, float],
) -> np.ndarray:
    out = np.array(base_joint_positions, dtype=float, copy=True)
    for name, value in targets.items():
        if name in dof_names:
            out[dof_names.index(name)] = float(value)
    return out


def safety_trip_inference(
    reason: str,
    *,
    inference_active_holder: List[bool],
    policy: ACTPolicy,
) -> None:
    inference_active_holder[0] = False
    try:
        policy.reset()
    except Exception:
        pass
    msg = f">>> SAFETY: inference OFF — {reason} <<<"
    carb.log_warn(f"[TIAGO-INFERENCE-SAFE] {msg}")
    print(msg, flush=True)


def main() -> None:
    carb.log_info("[TIAGO-INFERENCE-SAFE] ACT inference + Phase-A safety shell...")

    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT_ASSUMED)
    _setup_stage(world)

    try:
        tiago, art_kin_solver, gripper_joint_indices = setup_robot_and_ik(world)
    except Exception as e:
        carb.log_error(f"[TIAGO-INFERENCE-SAFE] setup failed: {e}")
        simulation_app.close()
        return

    all_joint_names = list(tiago.dof_names)
    arm_right_indices = arm_right_dof_indices(all_joint_names)

    stage = omni.usd.get_context().get_stage()
    camera_prim_path = find_gemini2_rgb_camera_prim_path(stage)
    if camera_prim_path is None:
        carb.log_error("[TIAGO-INFERENCE-SAFE] No Gemini2 Stream_rgb prim.")
        simulation_app.close()
        return

    rgb_annot = setup_rgb_annotator(camera_prim_path)

    gripper_body_path = find_gripper_rigidbody_prim(stage)
    if gripper_body_path is None:
        carb.log_error("[TIAGO-INFERENCE-SAFE] No gripper rigid body prim")
        simulation_app.close()
        return

    policy_cfg = PreTrainedConfig.from_pretrained(POLICY_DIR)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=POLICY_DIR,
        preprocessor_overrides={"device_processor": {"device": DEVICE}},
    )

    policy = ACTPolicy.from_pretrained(POLICY_DIR, device=DEVICE)
    policy.eval()

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard() if app_window else None
    input_iface = carb.input.acquire_input_interface()

    class _Key:
        space_edge = False
        v_edge = False

    key_state = _Key()

    def _on_keyboard(event: carb.input.KeyboardEvent, *_args: Any) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.SPACE:
                key_state.space_edge = True
            elif event.input == carb.input.KeyboardInput.V:
                key_state.v_edge = True
        return True

    sub = None
    if keyboard is not None:
        sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_keyboard)

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()

    # Settle a few frames then snapshot object baselines for drift / kick detection.
    for _ in range(8):
        world.step(render=True)
    baseline_object_pos = snapshot_pickable_positions(stage)
    prev_object_pos = {k: v.copy() for k, v in baseline_object_pos.items()}

    frame_count = 0
    inference_active = False
    cooldown_until = 0
    prev_min_finger_rad = float(GRIPPER_OPEN_POSITION)

    joint_prim_path: Optional[str] = None
    attached_object: Optional[str] = None

    post_close_verify_frame: Optional[int] = None
    last_applied_joint: Optional[np.ndarray] = None
    last_logged_min_finger: Optional[float] = None
    last_logged_fingers: Optional[Tuple[float, ...]] = None
    post_grasp_active = False
    post_grasp_start_joint: Optional[np.ndarray] = None
    post_grasp_target_joint: Optional[np.ndarray] = None
    post_grasp_end_frame = -1

    # Mutable holder so helper can clear inference flag
    infer_holder: List[bool] = [False]

    carb.log_info(
        "[TIAGO-INFERENCE-SAFE] Space = toggle inference. "
        + ("V = force attach nearest pickable. " if ATTACH_ON_V_KEY else "")
        + "Safety trips turn inference off."
    )

    try:
        while simulation_app.is_running():
            if not timeline.is_playing():
                timeline.play()

            world.step(render=True)
            frame_count += 1

            infer_holder[0] = inference_active

            if key_state.space_edge:
                key_state.space_edge = False
                if frame_count >= cooldown_until:
                    cooldown_until = frame_count + TOGGLE_DEBOUNCE_FRAMES
                    inference_active = not inference_active
                    infer_holder[0] = inference_active
                    if inference_active:
                        if post_grasp_active:
                            msg = ">>> POST-GRASP routine active; wait to re-enable inference <<<"
                            carb.log_warn(msg)
                            print(msg, flush=True)
                            inference_active = False
                            infer_holder[0] = False
                            continue
                        policy.reset()
                        last_applied_joint = None
                        post_close_verify_frame = None
                        msg = ">>> INFERENCE ON (SPACE) <<<"
                    else:
                        msg = ">>> INFERENCE OFF (SPACE) <<<"
                    carb.log_info(msg)
                    print(msg, flush=True)

            jp_after = tiago.get_joint_positions()
            if jp_after is None:
                continue
            joint_np = np.array(jp_after, dtype=float)

            ee_pos, _ = art_kin_solver.compute_end_effector_pose()
            ee_np = np.array([ee_pos[0], ee_pos[1], ee_pos[2]], dtype=float)
            near_dist, _nearest_path = nearest_pickable_distance_m(stage, ee_np)
            in_near_guard = near_dist < NEAR_GUARD_DISTANCE_M

            # Optional manual attach hotkey (V): bypass finger-threshold edge, but keep distance sanity.
            if key_state.v_edge:
                key_state.v_edge = False
                if ATTACH_ON_V_KEY and joint_prim_path is None:
                    try:
                        near_d, near_p = nearest_pickable_distance_m(stage, ee_np)
                        if near_p is not None and near_d <= ATTACH_ON_V_DISTANCE_MAX:
                            gripper_prim = stage.GetPrimAtPath(gripper_body_path)
                            obj_prim = stage.GetPrimAtPath(near_p)
                            joint_prim = physx_utils.createJoint(stage, "Fixed", gripper_prim, obj_prim)
                            joint_prim_path = joint_prim.GetPath().pathString
                            attached_object = near_p
                            post_close_verify_frame = None
                            if POST_GRASP_STOP_INFERENCE:
                                inference_active = False
                                infer_holder[0] = False
                                try:
                                    policy.reset()
                                except Exception:
                                    pass
                            post_grasp_start_joint = np.array(joint_np, dtype=float, copy=True)
                            post_grasp_target_joint = build_joint_target_from_map(
                                all_joint_names, post_grasp_start_joint, POST_GRASP_CARRY_TARGETS
                            )
                            post_grasp_end_frame = frame_count + POST_GRASP_MOVE_FRAMES
                            post_grasp_active = True
                            msg = (
                                f"[ATTACH HOTKEY V] frame={frame_count} object={near_p} "
                                f"ee_to_obj_dist={near_d:.4f}m joint_prim={joint_prim_path}"
                            )
                            carb.log_info(f"[TIAGO-INFERENCE-SAFE] {msg}")
                            print(msg, flush=True)
                        else:
                            msg = (
                                f"[ATTACH HOTKEY V MISS] frame={frame_count} nearest={near_p} "
                                f"d={near_d:.4f}m max={ATTACH_ON_V_DISTANCE_MAX:.3f}m"
                            )
                            carb.log_warn(f"[TIAGO-INFERENCE-SAFE] {msg}")
                            print(msg, flush=True)
                    except Exception as e:
                        carb.log_error(f"ATTACH HOTKEY V error: {e}")

            # --- Kick detector (only while not attached): object jumps / slides ---
            for path in PICKABLE_OBJECT_PATHS:
                pos = get_world_pos(stage, path)
                if pos is None:
                    continue
                prev = prev_object_pos.get(path)
                if joint_prim_path is None and prev is not None:
                    step_d = float(np.linalg.norm(pos - prev))
                    speed = step_d / max(PHYSICS_DT_ASSUMED, 1e-6)
                    if step_d > KICK_STEP_DISPLACE_M or speed > KICK_SPEED_M_S:
                        safety_trip_inference(
                            f"object sudden motion ({path.split('/')[-1]} step={step_d:.4f} m)",
                            inference_active_holder=infer_holder,
                            policy=policy,
                        )
                        inference_active = infer_holder[0]
                prev_object_pos[path] = pos.copy()

            min_finger_rad = min_gripper_finger_rad(joint_np, gripper_joint_indices)
            trigger_legacy = virtual_trigger_from_gripper_joints(joint_np, gripper_joint_indices)

            finger_tuple = tuple(float(joint_np[i]) for i in gripper_joint_indices)
            gripper_changed = False
            if last_logged_min_finger is None:
                gripper_changed = True
            else:
                if abs(min_finger_rad - last_logged_min_finger) >= GRIPPER_LOG_TRIGGER_DELTA:
                    gripper_changed = True
                if last_logged_fingers is not None and finger_tuple != last_logged_fingers:
                    gripper_changed = True
            if gripper_changed:
                line = _gripper_debug_line(
                    frame_count,
                    joint_np,
                    gripper_joint_indices,
                    min_finger_rad,
                    trigger_legacy,
                    attached_object,
                    joint_prim_path,
                )
                carb.log_info(f"[TIAGO-INFERENCE-SAFE] {line}")
                print(line, flush=True)
                last_logged_min_finger = min_finger_rad
                last_logged_fingers = finger_tuple

            # Schedule post-close verify when min finger crosses into "closed" band (same edge as attach)
            if (
                prev_min_finger_rad >= ATTACH_FINGER_MIN_RAD
                and min_finger_rad < ATTACH_FINGER_MIN_RAD
                and joint_prim_path is None
            ):
                post_close_verify_frame = frame_count + POST_CLOSE_VERIFY_FRAMES

            if post_close_verify_frame is not None and frame_count >= post_close_verify_frame:
                post_close_verify_frame = None
                if joint_prim_path is None:
                    for path, base in baseline_object_pos.items():
                        pos = get_world_pos(stage, path)
                        if pos is None:
                            continue
                        if float(np.linalg.norm(pos - base)) > OBJECT_DRIFT_FAIL_M:
                            safety_trip_inference(
                                "closed without attach and object drifted (possible kick)",
                                inference_active_holder=infer_holder,
                                policy=policy,
                            )
                            inference_active = infer_holder[0]
                            break

            rgb = get_rgb_hwc_uint8(rgb_annot)

            if post_grasp_active and post_grasp_start_joint is not None and post_grasp_target_joint is not None:
                if post_grasp_end_frame <= frame_count:
                    tiago.apply_action(ArticulationAction(joint_positions=post_grasp_target_joint))
                    post_grasp_active = False
                    carb.log_info("[TIAGO-INFERENCE-SAFE] [POST-GRASP] carry pose reached.")
                    print("[POST-GRASP] carry pose reached.", flush=True)
                else:
                    alpha = float((frame_count - (post_grasp_end_frame - POST_GRASP_MOVE_FRAMES)) / max(POST_GRASP_MOVE_FRAMES, 1))
                    alpha = float(np.clip(alpha, 0.0, 1.0))
                    interp = (1.0 - alpha) * post_grasp_start_joint + alpha * post_grasp_target_joint
                    tiago.apply_action(ArticulationAction(joint_positions=interp))
                    last_applied_joint = interp.copy()
            elif inference_active and rgb is not None:
                joint_f32 = joint_np.astype(np.float32)
                obs = preprocess_observation(
                    {
                        "pixels": {"gemini2": rgb},
                        "agent_pos": joint_f32,
                    }
                )
                obs = preprocessor(obs)
                with torch.inference_mode():
                    action_t = policy.select_action(obs)
                action_t = postprocessor(action_t)
                action_np = action_t.squeeze(0).cpu().numpy().astype(float)

                if last_applied_joint is None or last_applied_joint.shape != action_np.shape:
                    last_applied_joint = joint_np.copy()
                action_np = clamp_action_vs_previous(
                    action_np,
                    last_applied_joint,
                    arm_right_indices,
                    MAX_ARM_DELTA_NEAR_RAD,
                    MAX_ARM_DELTA_FAR_RAD,
                    in_near_guard,
                )
                tiago.apply_action(ArticulationAction(joint_positions=action_np))
                last_applied_joint = action_np.copy()
            else:
                last_applied_joint = joint_np.copy()

            # Attach: rising edge min finger crosses below ATTACH_FINGER_MIN_RAD + EE–object in range
            if (
                prev_min_finger_rad >= ATTACH_FINGER_MIN_RAD
                and min_finger_rad < ATTACH_FINGER_MIN_RAD
                and joint_prim_path is None
            ):
                try:
                    ee_p, _ = art_kin_solver.compute_end_effector_pose()
                    ee_arr = np.array([ee_p[0], ee_p[1], ee_p[2]], dtype=float)
                    pick = find_closest_pickable(stage, ee_arr)
                    if pick is not None:
                        obj_path, dist_pick = pick
                        gripper_prim = stage.GetPrimAtPath(gripper_body_path)
                        obj_prim = stage.GetPrimAtPath(obj_path)
                        joint_prim = physx_utils.createJoint(stage, "Fixed", gripper_prim, obj_prim)
                        joint_prim_path = joint_prim.GetPath().pathString
                        attached_object = obj_path
                        post_close_verify_frame = None
                        if POST_GRASP_STOP_INFERENCE:
                            inference_active = False
                            infer_holder[0] = False
                            try:
                                policy.reset()
                            except Exception:
                                pass
                        post_grasp_start_joint = np.array(joint_np, dtype=float, copy=True)
                        post_grasp_target_joint = build_joint_target_from_map(
                            all_joint_names, post_grasp_start_joint, POST_GRASP_CARRY_TARGETS
                        )
                        post_grasp_end_frame = frame_count + POST_GRASP_MOVE_FRAMES
                        post_grasp_active = True
                        msg = (
                            f"[ATTACH OK] frame={frame_count} object={obj_path} "
                            f"ee_to_obj_dist={dist_pick:.4f}m joint_prim={joint_prim_path} "
                            f"min_finger_rad={min_finger_rad:.4f} (thr<{ATTACH_FINGER_MIN_RAD})"
                        )
                        carb.log_info(f"[TIAGO-INFERENCE-SAFE] {msg}")
                        print(msg, flush=True)
                        carb.log_info(
                            f"[TIAGO-INFERENCE-SAFE] [POST-GRASP] inference off={POST_GRASP_STOP_INFERENCE}; "
                            f"moving to carry pose over {POST_GRASP_MOVE_FRAMES} frames."
                        )
                        print(
                            f"[POST-GRASP] moving to carry pose over {POST_GRASP_MOVE_FRAMES} frames.",
                            flush=True,
                        )
                    else:
                        near_d, near_p = nearest_pickable_distance_m(stage, ee_arr)
                        msg = (
                            f"[ATTACH MISS] frame={frame_count} min_finger<{ATTACH_FINGER_MIN_RAD} but no pickable "
                            f"within {ATTACH_DISTANCE_MAX}m (nearest={near_p} d={near_d:.4f}m) "
                            f"min_finger_rad={min_finger_rad:.4f}"
                        )
                        carb.log_warn(f"[TIAGO-INFERENCE-SAFE] {msg}")
                        print(msg, flush=True)
                except Exception as e:
                    carb.log_error(f"ATTACH error: {e}")

            # Detach: min finger opens past DETACH_FINGER_MIN_RAD (hysteresis vs attach)
            if AUTO_DETACH_ENABLED and joint_prim_path is not None and min_finger_rad > DETACH_FINGER_MIN_RAD:
                detached_from = attached_object
                detached_joint = joint_prim_path
                try:
                    stage.RemovePrim(joint_prim_path)
                except Exception:
                    pass
                joint_prim_path = None
                attached_object = None
                msg = (
                    f"[DETACH] frame={frame_count} was_attached={detached_from} "
                    f"joint_prim={detached_joint} min_finger_rad={min_finger_rad:.4f} (thr>{DETACH_FINGER_MIN_RAD})"
                )
                carb.log_info(f"[TIAGO-INFERENCE-SAFE] {msg}")
                print(msg, flush=True)

            prev_min_finger_rad = min_finger_rad

    finally:
        if sub is not None:
            try:
                sub = None
            except Exception:
                pass
        simulation_app.close()


if __name__ == "__main__":
    main()
