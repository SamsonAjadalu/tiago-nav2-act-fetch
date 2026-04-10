"""
Run trained LeRobot ACT policy in Isaac Sim (no VR / OpenXR).

- **Space**: toggle policy inference on/off (debounced).
- **RGB**: Replicator annotator on the Gemini2 ``Stream_rgb`` camera (no ROS2 bridge required).
- **Normalization**: LeRobot preprocessor / postprocessor loaded from the pretrained folder.

Launcher: extra/isaacsim/run_inference.sh. ISAACSIM_ROOT must be set in extra/local_paths.sh.
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
from typing import Any, List, Optional, Tuple

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
# Configuration
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
# Weld uses min(left, right) finger joint value, not VR-style trigger (see min_gripper_finger_rad).
ATTACH_FINGER_MIN_RAD = 0.04
DETACH_FINGER_MIN_RAD = 0.042

PICKABLE_OBJECT_PATHS: List[str] = [
    "/World/Table/_05_tomato_soup_can/_05_tomato_soup_can",
    "/World/Table/_06_mustard_bottle/_06_mustard_bottle",
    "/World/PickBox",
]

TOGGLE_DEBOUNCE_FRAMES = 30


def find_gemini2_rgb_camera_prim_path(stage) -> Optional[str]:
    """Match teleop: Gemini RGB is often a prim whose path contains ``stream_rgb``."""
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
        raise RuntimeError("[TIAGO-INFERENCE] Articulation not initialized!")

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
    carb.log_info(f"[TIAGO-INFERENCE] EE frame: {RIGHT_EE_FRAME_NAME}, test_pos={ee_pos}")

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


def find_closest_pickable(stage, ee_pos: np.ndarray) -> Optional[Tuple[str, float]]:
    """Same whitelist + distance cap as ``tiago_vr_teleop_record_for_lerobot_npz``."""
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


def get_world_pos(stage, prim_path: str) -> Optional[np.ndarray]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    xform = UsdGeom.Xformable(prim)
    time = Usd.TimeCode.Default()
    transform = xform.ComputeLocalToWorldTransform(time)
    translation = transform.ExtractTranslation()
    return np.array([translation[0], translation[1], translation[2]])


def virtual_trigger_from_gripper_joints(joint_positions: np.ndarray, gripper_joint_indices: List[int]) -> float:
    """Legacy 0–1 metric for logging only; attach/detach use min_gripper_finger_rad."""
    if not gripper_joint_indices:
        return 0.0
    vals = [float(joint_positions[i]) for i in gripper_joint_indices]
    mean_f = float(np.mean(vals))
    t = 1.0 - (mean_f / GRIPPER_OPEN_POSITION)
    return float(np.clip(t, 0.0, 1.0))


def min_gripper_finger_rad(joint_positions: np.ndarray, gripper_joint_indices: List[int]) -> float:
    if not gripper_joint_indices:
        return float(GRIPPER_OPEN_POSITION)
    return min(float(joint_positions[i]) for i in gripper_joint_indices)


def main() -> None:
    carb.log_info("[TIAGO-INFERENCE] Starting ACT inference (keyboard toggle; no XR)...")

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
    _setup_stage(world)

    try:
        tiago, art_kin_solver, gripper_joint_indices = setup_robot_and_ik(world)
    except Exception as e:
        carb.log_error(f"[TIAGO-INFERENCE] setup_robot_and_ik failed: {e}")
        simulation_app.close()
        return

    stage = omni.usd.get_context().get_stage()
    camera_prim_path = find_gemini2_rgb_camera_prim_path(stage)
    if camera_prim_path is None:
        carb.log_error("[TIAGO-INFERENCE] No Gemini2 Stream_rgb prim — cannot render policy camera.")
        simulation_app.close()
        return
    carb.log_info(f"[TIAGO-INFERENCE] RGB camera prim: {camera_prim_path}")

    rgb_annot = setup_rgb_annotator(camera_prim_path)

    gripper_body_path = find_gripper_rigidbody_prim(stage)
    if gripper_body_path is None:
        carb.log_error("[TIAGO-INFERENCE] Could not find gripper rigid body prim")
        simulation_app.close()
        return

    policy_cfg = PreTrainedConfig.from_pretrained(POLICY_DIR)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=POLICY_DIR,
        preprocessor_overrides={"device_processor": {"device": DEVICE}},
    )

    carb.log_info(f"[TIAGO-INFERENCE] Loading policy from {POLICY_DIR}...")
    policy = ACTPolicy.from_pretrained(POLICY_DIR, device=DEVICE)
    policy.eval()

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard() if app_window else None
    input_iface = carb.input.acquire_input_interface()

    class _Key:
        space_edge = False

    key_state = _Key()

    def _on_keyboard(event: carb.input.KeyboardEvent, *_args: Any) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.SPACE:
                key_state.space_edge = True
        return True

    sub = None
    if keyboard is not None:
        sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_keyboard)
    else:
        carb.log_warn("[TIAGO-INFERENCE] No keyboard — use UI; Space toggle may not work.")

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()

    frame_count = 0
    inference_active = False
    cooldown_until = 0
    prev_min_finger_rad = float(GRIPPER_OPEN_POSITION)

    joint_prim_path: Optional[str] = None
    attached_object: Optional[str] = None

    carb.log_info("[TIAGO-INFERENCE] Press SPACE to toggle inference.")

    try:
        while simulation_app.is_running():
            if not timeline.is_playing():
                timeline.play()

            world.step(render=True)
            frame_count += 1

            if key_state.space_edge:
                key_state.space_edge = False
                if frame_count >= cooldown_until:
                    cooldown_until = frame_count + TOGGLE_DEBOUNCE_FRAMES
                    inference_active = not inference_active
                    if inference_active:
                        policy.reset()
                        msg = ">>> INFERENCE ON (SPACE) <<<"
                    else:
                        msg = ">>> INFERENCE OFF (SPACE) <<<"
                    carb.log_info(msg)
                    print(msg, flush=True)

            rgb = get_rgb_hwc_uint8(rgb_annot)

            if inference_active and rgb is not None:
                jp = tiago.get_joint_positions()
                if jp is not None:
                    joint_np = np.array(jp, dtype=np.float32)
                    obs = preprocess_observation(
                        {
                            "pixels": {"gemini2": rgb},
                            "agent_pos": joint_np,
                        }
                    )
                    obs = preprocessor(obs)
                    with torch.inference_mode():
                        action_t = policy.select_action(obs)
                    action_t = postprocessor(action_t)
                    action_np = action_t.squeeze(0).cpu().numpy()
                    tiago.apply_action(ArticulationAction(joint_positions=action_np))

            jp_after = tiago.get_joint_positions()
            if jp_after is not None:
                joint_np = np.array(jp_after, dtype=float)
                min_finger_rad = min_gripper_finger_rad(joint_np, gripper_joint_indices)

                if (
                    prev_min_finger_rad >= ATTACH_FINGER_MIN_RAD
                    and min_finger_rad < ATTACH_FINGER_MIN_RAD
                    and joint_prim_path is None
                ):
                    try:
                        ee_pos, _ = art_kin_solver.compute_end_effector_pose()
                        ee_arr = np.array([ee_pos[0], ee_pos[1], ee_pos[2]], dtype=float)
                        pick = find_closest_pickable(stage, ee_arr)
                        if pick is not None:
                            obj_path, _dist = pick
                            gripper_prim = stage.GetPrimAtPath(gripper_body_path)
                            obj_prim = stage.GetPrimAtPath(obj_path)
                            joint_prim = physx_utils.createJoint(stage, "Fixed", gripper_prim, obj_prim)
                            joint_prim_path = joint_prim.GetPath().pathString
                            attached_object = obj_path
                            carb.log_info(
                                f"PhysX attach: {attached_object} min_finger_rad={min_finger_rad:.4f}"
                            )
                    except Exception as e:
                        carb.log_error(f"ATTACH error: {e}")

                if joint_prim_path is not None and min_finger_rad > DETACH_FINGER_MIN_RAD:
                    try:
                        stage.RemovePrim(joint_prim_path)
                    except Exception:
                        pass
                    joint_prim_path = None
                    attached_object = None
                    carb.log_info("PhysX detach")

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
