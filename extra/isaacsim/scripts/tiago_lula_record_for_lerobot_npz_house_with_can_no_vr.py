"""
Record house-scene TIAGo pick episodes into NPZ using scripted Lula IK.

No VR/OpenXR/controller dependency.

IK targets the configured EE frame (often arm_right_tool_link / wrist). Default waypoints keep the
gripper well clear of the can (align / approach / pregrasp); tighten with TIAGO_WAYPOINT_*_M only if
needed. The can is never moved by H. Press H to inch the IK target forward along the approach when
ready to straddle; TIAGO_H_FORWARD_STEP_M / TIAGO_H_FORWARD_MAX_M cap that motion.

Run:
  bash tiago_hil/extra/isaacsim/run_python.sh \
    tiago_hil/extra/isaacsim/scripts/tiago_lula_record_for_lerobot_npz_house_with_can_no_vr.py
"""

from __future__ import annotations

import os
import re
import select
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False, "enable_ros2": True})

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tiago_repo_paths import (
    default_isaacsim_data_root,
    default_tiago_usd_path,
    default_urdf_path,
    default_yaml_path,
    house_usd_v1_path,
    tiago_repo_root,
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
from omni.physx.scripts import utils as physx_utils
from pxr import Gf, UsdGeom, UsdPhysics

# Paths
TIAGO_USD_PATH = default_tiago_usd_path()
HOUSE_USD_PATH = house_usd_v1_path()
TIAGO_URDF_PATH = default_urdf_path()
TIAGO_YAML_PATH = os.environ.get("TIAGO_LULA_YAML_PATH", "").strip() or default_yaml_path()

RIGHT_EE_FRAME_NAME = os.environ.get("TIAGO_LULA_EE_FRAME", "arm_right_tool_link")
GRIPPER_JOINT_NAMES = ["gripper_right_left_finger_joint", "gripper_right_right_finger_joint"]
GRIPPER_OPEN_POSITION = 0.045
GRIPPER_CLOSED_POSITION = 0.0
ATTACH_DISTANCE_MAX = float(os.environ.get("TIAGO_ATTACH_DIST_MAX", "0.10"))
# PhysX FixedJoint attach is off by default (no contact policy; add final approach manually later).
PHYSX_ATTACH = os.environ.get("TIAGO_PHYSX_ATTACH", "0").strip() in ("1", "true", "True", "yes", "YES")
# Prefer these USD link names for FixedJoint (forearm gripper_right_link is wrong visually).
ATTACH_LINK_PRIORITY = os.environ.get(
    "TIAGO_ATTACH_LINK_PRIORITY",
    "gripper_right_right_finger_link,gripper_right_left_finger_link,gripper_right_grasping_frame,gripper_right_tool_link,gripper_right_link",
).split(",")

ROBOT_TRANSLATE = (
    float(os.environ.get("TIAGO_HOUSE_ROBOT_X", "-0.11527")),
    float(os.environ.get("TIAGO_HOUSE_ROBOT_Y", "1.41377")),
    float(os.environ.get("TIAGO_HOUSE_ROBOT_Z", "0.0")),
)
ROBOT_YAW_DEG = float(os.environ.get("TIAGO_HOUSE_ROBOT_YAW_DEG", "162.0"))

CAN_TRANSLATE = (
    float(os.environ.get("TIAGO_HOUSE_CAN_X", "-0.63146")),
    float(os.environ.get("TIAGO_HOUSE_CAN_Y", "2.05573")),
    float(os.environ.get("TIAGO_HOUSE_CAN_Z", "0.81218")),
)
CAN_ROTATE_XYZ_DEG = (
    float(os.environ.get("TIAGO_HOUSE_CAN_ROLL_DEG", "-90.0")),
    float(os.environ.get("TIAGO_HOUSE_CAN_PITCH_DEG", "0.0")),
    float(os.environ.get("TIAGO_HOUSE_CAN_YAW_DEG", "0.0")),
)
# Default 0: never overwrite the can prim transform from house_with_can.usd (use authored pose).
# Set TIAGO_OVERRIDE_CAN_POSE=1 only when intentionally placing the can (e.g. matching a Property panel snapshot).
OVERRIDE_CAN_POSE = os.environ.get("TIAGO_OVERRIDE_CAN_POSE", "0").strip() in ("1", "true", "True", "yes", "YES")

TORSO_LIFT_START = float(os.environ.get("TIAGO_TORSO_LIFT_START", "0.35"))
# When 1, reuse last joint command if the Cartesian IK target is unchanged (reduces null-space jitter).
IK_HOLD_WITHOUT_RESOLVE = os.environ.get("TIAGO_IK_HOLD_WITHOUT_RESOLVE", "1").strip() not in (
    "0",
    "false",
    "False",
    "no",
    "NO",
)
IK_RETARGET_POS_EPS = float(os.environ.get("TIAGO_IK_RETARGET_POS_EPS", "0.0002"))

# H key: shift IK targets toward the can (can prim never moved by H).
H_FORWARD_STEP_M = float(os.environ.get("TIAGO_H_FORWARD_STEP_M", "0.018"))
H_FORWARD_MAX_M = float(os.environ.get("TIAGO_H_FORWARD_MAX_M", "0.10"))

NUM_EPISODES = int(os.environ.get("TIAGO_SCRIPTED_EPISODES", "10"))
# Slower scripted motion by default (more sim steps per waypoint).
WP_HOLD_STEPS = int(os.environ.get("TIAGO_SCRIPTED_WP_HOLD_STEPS", "72"))
CLOSE_HOLD_STEPS = int(os.environ.get("TIAGO_SCRIPTED_CLOSE_HOLD_STEPS", "36"))
SETTLE_STEPS = int(os.environ.get("TIAGO_SCRIPTED_SETTLE_STEPS", "32"))
WAIT_FOR_SPACE = os.environ.get("TIAGO_WAIT_FOR_SPACE", "1").strip() not in ("0", "false", "False", "no", "NO")
SPACE_WAIT_TIMEOUT_S = float(os.environ.get("TIAGO_SPACE_WAIT_TIMEOUT_S", "0"))
ABORT_ON_SELF_TEST_FAIL = os.environ.get("TIAGO_ABORT_ON_SELF_TEST_FAIL", "1").strip() not in (
    "0",
    "false",
    "False",
    "no",
    "NO",
)

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
    "torso_lift_joint": TORSO_LIFT_START,
}

_DEFAULT_DATA = str(default_isaacsim_data_root())
_ISAAC_DATA = os.environ.get("TIAGO_ISAACSIM_DATA_ROOT", _DEFAULT_DATA)
RECORD_OUT_DIR = os.environ.get("TIAGO_LEROBOT_NPZ_DIR", os.path.join(_ISAAC_DATA, "tiago_lerobot_npz"))
LOG_FILE_PATH = os.path.join(RECORD_OUT_DIR, "tiago_lula_record_for_lerobot_log.txt")
_EPISODE_NPZ_RE = re.compile(r"^episode_(\d{4})\.npz$")


def log_to_file(msg: str) -> None:
    os.makedirs(RECORD_OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def next_episode_index_from_npz_files() -> int:
    best = -1
    if not os.path.isdir(RECORD_OUT_DIR):
        return 0
    for name in os.listdir(RECORD_OUT_DIR):
        m = _EPISODE_NPZ_RE.match(name)
        if m:
            best = max(best, int(m.group(1)))
    return max(0, best + 1)


def save_episode(episode_index: int, episode_frames: List[Dict[str, Any]]) -> None:
    out_path = os.path.join(RECORD_OUT_DIR, f"episode_{episode_index:04d}.npz")
    if not episode_frames:
        log_to_file(f"Episode {episode_index}: no frames, skip save.")
        return
    stacked = {k: [] for k in episode_frames[0].keys()}
    for fr in episode_frames:
        for k, v in fr.items():
            stacked[k].append(v)
    np.savez_compressed(out_path, **{k: np.array(v) for k, v in stacked.items()})
    log_to_file(f"Saved: {out_path} frames={len(episode_frames)}")


def find_prim_path_by_name(stage, prim_name: str) -> Optional[str]:
    for prim in stage.Traverse():
        if prim.IsValid() and prim.GetName() == prim_name:
            return str(prim.GetPath())
    return None


def find_first_path_with_any_token(stage, tokens: List[str]) -> Optional[str]:
    toks = [t.lower() for t in tokens]
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        p = str(prim.GetPath())
        if any(t in p.lower() for t in toks):
            return p
    return None


def upsert_translate_rotate_xyz(stage, prim_path: str, translate_xyz: Tuple[float, float, float], rotate_xyz_deg: Tuple[float, float, float]) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")
    xf = UsdGeom.Xformable(prim)
    t_op = None
    r_op = None
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and t_op is None:
            t_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ and r_op is None:
            r_op = op
    if t_op is None:
        t_op = xf.AddTranslateOp()
    if r_op is None:
        r_op = xf.AddRotateXYZOp()
    t_op.Set(Gf.Vec3d(*translate_xyz))
    r_op.Set(Gf.Vec3f(*rotate_xyz_deg))


def maybe_apply_can_pose_override(stage, can_path: str) -> None:
    if not OVERRIDE_CAN_POSE:
        return
    upsert_translate_rotate_xyz(stage, can_path, CAN_TRANSLATE, CAN_ROTATE_XYZ_DEG)


def get_world_pos(stage, prim_path: str) -> Optional[np.ndarray]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
    t = mat.ExtractTranslation()
    return np.array([t[0], t[1], t[2]], dtype=float)


def find_gripper_attach_rigid_body(stage, tiago_prefix: str = "/World/Tiago/tiago_dual_functional") -> Optional[str]:
    """Rigid body prim for FixedJoint to can — prefer grasping frame / fingers, not forearm."""
    want = [n.strip() for n in ATTACH_LINK_PRIORITY if n.strip()]
    candidates: List[Tuple[int, str]] = []
    for prim in stage.Traverse():
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        path = prim.GetPath().pathString
        if tiago_prefix not in path:
            continue
        name = prim.GetName()
        for pri, link_name in enumerate(want):
            if name == link_name or path.endswith("/" + link_name):
                candidates.append((pri, path))
                break
    if not candidates:
        log_to_file("WARNING: No attach rigid body under TIAGo; tried: " + ",".join(want))
        return None
    candidates.sort(key=lambda x: x[0])
    chosen = candidates[0][1]
    log_to_file(f"Attach body (rigid): {chosen}")
    return chosen


def combine_arm_and_gripper_action(
    tiago: Robot,
    ik_action: ArticulationAction,
    gripper_joint_indices: List[int],
    close_gripper: bool,
) -> Tuple[ArticulationAction, Optional[np.ndarray]]:
    q = tiago.get_joint_positions()
    if q is None:
        return ik_action, None
    q_all = np.array(q, dtype=float)
    if ik_action.joint_positions is not None:
        ik_q = np.array(ik_action.joint_positions, dtype=float)
        if hasattr(ik_action, "joint_indices") and ik_action.joint_indices is not None:
            idx = np.array(ik_action.joint_indices, dtype=int)
            for i, j in enumerate(idx):
                if i < len(ik_q) and j < len(q_all):
                    q_all[j] = ik_q[i]
        else:
            q_all[: len(ik_q)] = ik_q
    g = GRIPPER_CLOSED_POSITION if close_gripper else GRIPPER_OPEN_POSITION
    for gi in gripper_joint_indices:
        if gi < len(q_all):
            q_all[gi] = g
    if "torso_lift_joint" in tiago.dof_names:
        q_all[tiago.dof_names.index("torso_lift_joint")] = TORSO_LIFT_START
    return ArticulationAction(joint_positions=q_all), q_all


def compute_waypoints(can_pos: np.ndarray, approach_dir: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """align -> approach -> pregrasp (wrist short of can; no contact) -> lift -> retreat.

    Returns waypoints and horizontal unit vector d (EE -> can in the approach plane). IK targets are
    shifted by +nudge*d to move the wrist toward the fixed can without moving the can.

    align_dist / approach_dist / grasp_dist are horizontal standoffs from the can along d (larger =
    farther from the object). Defaults favor a wide margin; reduce via env only for a tighter demo.
    """
    d = np.array(approach_dir, dtype=float)
    d[2] = 0.0
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        d = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        d = d / n
    z_up = np.array([0.0, 0.0, 1.0], dtype=float)
    align_dist = float(os.environ.get("TIAGO_WAYPOINT_ALIGN_M", "0.42"))
    approach_dist = float(os.environ.get("TIAGO_WAYPOINT_APPROACH_M", "0.28"))
    grasp_dist = float(os.environ.get("TIAGO_WAYPOINT_GRASP_OFFSET_M", "0.20"))
    align_z = float(os.environ.get("TIAGO_WAYPOINT_ALIGN_Z", "0.15"))
    approach_z = float(os.environ.get("TIAGO_WAYPOINT_APPROACH_Z", "0.08"))
    grasp_z = float(os.environ.get("TIAGO_WAYPOINT_GRASP_Z", "0.04"))
    lift_dz = float(os.environ.get("TIAGO_LIFT_DZ", "0.12"))
    align = can_pos - align_dist * d + align_z * z_up
    approach_pt = can_pos - approach_dist * d + approach_z * z_up
    grasp = can_pos - grasp_dist * d + grasp_z * z_up
    lift = grasp + lift_dz * z_up
    retreat = align.copy()
    return (
        {
            "align": align,
            "approach": approach_pt,
            "grasp": grasp,
            "lift": lift,
            "retreat": retreat,
        },
        d,
    )


def world_to_solver_frame(p_world: np.ndarray, robot_xyz: Tuple[float, float, float], robot_yaw_deg: float) -> np.ndarray:
    # ArticulationKinematicsSolver in this setup reports/solves in robot base-like frame.
    # Convert world points to that frame using configured TIAGo spawn pose.
    dx = np.array(p_world, dtype=float) - np.array(robot_xyz, dtype=float)
    yaw = np.deg2rad(robot_yaw_deg)
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    r_wb = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    # world -> base
    return r_wb.T @ dx


def wait_for_space_to_start() -> None:
    if not WAIT_FOR_SPACE:
        log_to_file("Space gate disabled (TIAGO_WAIT_FOR_SPACE=0). Starting immediately.")
        return
    if not sys.stdin.isatty():
        log_to_file("STDIN is not a TTY; SPACE gate skipped automatically.")
        return
    log_to_file("Paused before episode loop. Press SPACE to start (or 'q' then Enter to quit).")
    start_t = None
    if SPACE_WAIT_TIMEOUT_S > 0:
        start_t = datetime.now().timestamp()
    while True:
        if start_t is not None and (datetime.now().timestamp() - start_t) > SPACE_WAIT_TIMEOUT_S:
            log_to_file(f"SPACE wait timeout ({SPACE_WAIT_TIMEOUT_S}s). Starting automatically.")
            return
        ready, _, _ = select.select([sys.stdin], [], [], 0.2)
        if not ready:
            continue
        ch = sys.stdin.read(1)
        if ch == " ":
            log_to_file("SPACE received. Starting scripted episodes.")
            return
        if ch.lower() == "q":
            raise KeyboardInterrupt("User requested quit before start.")


def run_self_test(ik_solver: ArticulationKinematicsSolver, can_pos: np.ndarray, ee_start: np.ndarray) -> Tuple[Dict[str, np.ndarray], bool]:
    ee = np.array(ee_start, dtype=float)
    can = np.array(can_pos, dtype=float)
    wp, _ = compute_waypoints(can, can - ee)
    d = float(np.linalg.norm(can - ee))
    log_to_file(f"Self-test context: ee_start={ee.tolist()} can_pos={can.tolist()} dist={d:.4f}m")
    log_to_file("Self-test: IK reachability for align/approach/pregrasp/lift/retreat.")
    all_ok = True
    for name in ("align", "approach", "grasp", "lift", "retreat"):
        _, ok = ik_solver.compute_inverse_kinematics(target_position=wp[name])
        log_to_file(f"Self-test IK {name}: {'OK' if ok else 'FAIL'} target={wp[name].tolist()}")
        if not ok:
            all_ok = False
    if not all_ok:
        log_to_file(
            "Self-test failed for one or more waypoints. "
            "Tune robot/can poses or waypoint offsets before collecting data."
        )
    else:
        log_to_file("Self-test passed for all waypoints.")
    return wp, all_ok


def build_ik_solver_with_fallback_frames(tiago: Robot):
    yaml_candidates: List[str] = []
    for y in [
        TIAGO_YAML_PATH,
        str(tiago_repo_root() / "config" / "OBOT_DESCRIPTION_YAML.yaml"),
        str(tiago_repo_root() / "OBOT_DESCRIPTION_YAML.yaml"),
    ]:
        if y and y not in yaml_candidates:
            yaml_candidates.append(y)

    lula = None
    last_yaml_err = None
    chosen_yaml = None
    for yaml_path in yaml_candidates:
        try:
            lula = LulaKinematicsSolver(robot_description_path=yaml_path, urdf_path=TIAGO_URDF_PATH)
            chosen_yaml = yaml_path
            log_to_file(f"Lula YAML accepted: {yaml_path}")
            break
        except Exception as e:
            last_yaml_err = e
            log_to_file(f"Lula YAML rejected: {yaml_path}, reason={e}")
    if lula is None:
        raise RuntimeError(f"Failed to initialize Lula with any YAML candidate. last_error={last_yaml_err}")

    preferred = RIGHT_EE_FRAME_NAME
    candidates = []
    for name in [
        preferred,
        "arm_right_tool_link",
        "gripper_right_grasping_frame",
        "arm_tool_link",
        "gripper_right_left_finger_link",
    ]:
        if name not in candidates:
            candidates.append(name)

    last_err = None
    for frame in candidates:
        try:
            solver = ArticulationKinematicsSolver(tiago, lula, frame)
            ee_pos, _ = solver.compute_end_effector_pose()
            log_to_file(f"Lula IK EE frame selected: {frame}, ee_start={np.array(ee_pos, dtype=float).tolist()}")
            return solver, frame, chosen_yaml
        except Exception as e:
            last_err = e
            log_to_file(f"Lula IK EE frame rejected: {frame}, reason={e}")
    raise RuntimeError(f"Failed to build IK solver for all EE frame candidates. last_error={last_err}")


def main() -> None:
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()

    if not Path(HOUSE_USD_PATH).exists():
        raise FileNotFoundError(f"Scene USD not found: {HOUSE_USD_PATH}")

    add_reference_to_stage(HOUSE_USD_PATH, "/World/HouseScene")
    can_path = find_prim_path_by_name(stage, "_05_tomato_soup_can")
    if can_path is None:
        can_path = find_first_path_with_any_token(stage, ["tomato", "soup", "can"])
    if can_path is None:
        raise RuntimeError("Could not locate can prim in house scene.")
    maybe_apply_can_pose_override(stage, can_path)

    add_reference_to_stage(TIAGO_USD_PATH, "/World/Tiago")
    tiago_root = stage.GetPrimAtPath("/World/Tiago")
    if tiago_root.IsValid():
        xf = UsdGeom.Xformable(tiago_root)
        xf.AddTranslateOp().Set(Gf.Vec3d(*ROBOT_TRANSLATE))
        xf.AddRotateZOp().Set(ROBOT_YAW_DEG)

    tiago = world.scene.add(Robot(prim_path="/World/Tiago/tiago_dual_functional", name="tiago"))

    world.reset()
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
    for _ in range(50):
        tiago.apply_action(ArticulationAction(joint_positions=q0))
        world.step(render=False)

    gripper_joint_indices = [tiago.dof_names.index(n) for n in GRIPPER_JOINT_NAMES if n in tiago.dof_names]
    log_to_file(
        f"Lula config: urdf={TIAGO_URDF_PATH}, yaml_preferred={TIAGO_YAML_PATH}, preferred_ee={RIGHT_EE_FRAME_NAME}"
    )
    ik_solver, chosen_ee_frame, chosen_yaml = build_ik_solver_with_fallback_frames(tiago)
    log_to_file(f"Lula active YAML: {chosen_yaml}")
    log_to_file(f"Lula active EE frame: {chosen_ee_frame}")
    log_to_file(
        f"torso_lift_joint held at {TORSO_LIFT_START} (TIAGO_TORSO_LIFT_START); "
        f"IK re-solve only on target change: {IK_HOLD_WITHOUT_RESOLVE} (TIAGO_IK_HOLD_WITHOUT_RESOLVE=0 to disable)."
    )
    gripper_body_path = find_gripper_attach_rigid_body(stage)
    if gripper_body_path is None and PHYSX_ATTACH:
        raise RuntimeError("Could not locate gripper/grasp rigid body prim for attach (needs TIAGO_PHYSX_ATTACH=1).")
    if gripper_body_path is None:
        log_to_file("No attach rigid body found; TIAGO_PHYSX_ATTACH=0 so continuing without attach path.")

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()

    key_state = SimpleNamespace(h_edge=False, toward_can_nudge_m=0.0)
    app_window = omni.appwindow.get_default_app_window()
    keyboard_dev = app_window.get_keyboard() if app_window else None
    input_iface = carb.input.acquire_input_interface()

    def _on_keyboard(event: carb.input.KeyboardEvent, *_args: Any) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.H:
                key_state.h_edge = True
        return True

    if keyboard_dev is not None:
        _ = input_iface.subscribe_to_keyboard_events(keyboard_dev, _on_keyboard)
        log_to_file(
            f"Hotkey: H advances IK target toward can (+{H_FORWARD_STEP_M}m/step, max {H_FORWARD_MAX_M}m); can pose unchanged."
        )
    else:
        log_to_file("No app keyboard: H hotkey unavailable.")

    os.makedirs(RECORD_OUT_DIR, exist_ok=True)
    episode_index = next_episode_index_from_npz_files()
    log_to_file(f"Start scripted Lula recording: episodes={NUM_EPISODES}, first_episode={episode_index}")
    log_to_file(
        f"PhysX attach: {'on' if PHYSX_ATTACH else 'off'} (TIAGO_PHYSX_ATTACH=1 to enable FixedJoint)."
    )
    log_to_file(f"Robot spawn xyz={ROBOT_TRANSLATE}, yaw_deg={ROBOT_YAW_DEG}")
    if OVERRIDE_CAN_POSE:
        log_to_file(f"Can path: {can_path}, TIAGO_OVERRIDE_CAN_POSE=1 -> xyz={CAN_TRANSLATE}, rpy_deg={CAN_ROTATE_XYZ_DEG}")
    else:
        _cw = get_world_pos(stage, can_path)
        log_to_file(
            f"Can path: {can_path}, pose left as-authored in USD (set TIAGO_OVERRIDE_CAN_POSE=1 to force env pose). "
            f"world_xyz={_cw.tolist() if _cw is not None else None}"
        )

    joint_prim_path: Optional[str] = None
    attached_object: Optional[str] = None

    APPROACH_STEPS = int(os.environ.get("TIAGO_APPROACH_HOLD_STEPS", "48"))
    RESET_AFTER_EPISODE = os.environ.get("TIAGO_RESET_AFTER_EPISODE", "1").strip() not in ("0", "false", "False")

    def reset_episode_state() -> None:
        nonlocal joint_prim_path, attached_object
        if joint_prim_path is not None:
            try:
                stage.RemovePrim(joint_prim_path)
            except Exception:
                pass
            joint_prim_path = None
            attached_object = None
        maybe_apply_can_pose_override(stage, can_path)
        if RESET_AFTER_EPISODE:
            world.reset()
            for _ in range(20):
                world.step(render=False)
            qp = tiago.get_joint_positions()
            if qp is None:
                log_to_file("WARNING: get_joint_positions None after reset; extra settle steps")
                for _ in range(40):
                    world.step(render=True)
                qp = tiago.get_joint_positions()
            if qp is not None:
                q_reset = np.array(qp, dtype=float)
                for jn, v in DEFAULT_JOINT_POSITIONS.items():
                    if jn in tiago.dof_names:
                        q_reset[tiago.dof_names.index(jn)] = v
                for _ in range(60):
                    tiago.apply_action(ArticulationAction(joint_positions=q_reset))
                    world.step(render=True)
        log_to_file(
            "Episode reset: joint removed, "
            + ("can pose re-applied from env (TIAGO_OVERRIDE_CAN_POSE=1). " if OVERRIDE_CAN_POSE else "can pose unchanged (USD). ")
            + "robot joints re-commanded."
        )

    def process_h_forward_nudge() -> None:
        if not key_state.h_edge:
            return
        key_state.h_edge = False
        prev = float(key_state.toward_can_nudge_m)
        key_state.toward_can_nudge_m = min(prev + H_FORWARD_STEP_M, H_FORWARD_MAX_M)
        if key_state.toward_can_nudge_m > prev:
            log_to_file(
                f"H: IK target offset toward can = {key_state.toward_can_nudge_m:.4f}m (can prim untouched)."
            )
        else:
            log_to_file(f"H: forward offset already at max ({H_FORWARD_MAX_M}m).")

    def record_frame(frames: List[Dict[str, Any]], target_position: np.ndarray, action_q: np.ndarray, close_gripper: bool) -> None:
        q_now = tiago.get_joint_positions()
        if q_now is None:
            return
        frames.append(
            {
                "t": float(world.current_time),
                "vr_pos": np.array(target_position, dtype=float),  # retained key for schema compatibility
                "vr_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
                "joint_pos": np.array(q_now, dtype=float),
                "action_joint_pos": np.array(action_q, dtype=float),
                "trigger": float(1.0 if close_gripper else 0.0),
                "attached": bool(joint_prim_path is not None),
                "picked_object_path": attached_object or "",
            }
        )

    def drive_to_target(
        base_target: np.ndarray,
        d_unit: np.ndarray,
        hold_steps: int,
        close_gripper: bool,
        frames: List[Dict[str, Any]],
        tag: str,
    ) -> bool:
        ok = False
        fail_count = 0
        last_q = tiago.get_joint_positions()
        du = np.array(d_unit, dtype=float)
        target_position = np.array(base_target, dtype=float)
        prev_target: Optional[np.ndarray] = None
        stable_q: Optional[np.ndarray] = None
        for _ in range(hold_steps):
            if not simulation_app.is_running():
                return ok
            process_h_forward_nudge()
            target_position = np.array(base_target, dtype=float) + float(key_state.toward_can_nudge_m) * du
            world.step(render=True)
            target_changed = prev_target is None or float(np.linalg.norm(target_position - prev_target)) > IK_RETARGET_POS_EPS
            prev_target = np.array(target_position, copy=True)

            recompute_ik = stable_q is None or target_changed or not IK_HOLD_WITHOUT_RESOLVE
            if recompute_ik:
                ik_action, ik_ok = ik_solver.compute_inverse_kinematics(target_position=target_position)
                if ik_ok:
                    act, q_cmd = combine_arm_and_gripper_action(tiago, ik_action, gripper_joint_indices, close_gripper)
                    tiago.apply_action(act)
                    ok = True
                    if q_cmd is not None:
                        stable_q = np.array(q_cmd, dtype=float)
                        last_q = stable_q
                else:
                    fail_count += 1
                    stable_q = None
            else:
                act = ArticulationAction(joint_positions=np.array(stable_q, dtype=float))
                tiago.apply_action(act)
                ok = True
                last_q = stable_q
            if last_q is not None:
                record_frame(frames, target_position, np.array(last_q, dtype=float), close_gripper)
        if not ok:
            log_to_file(f"[IK] {tag}: no valid solution. target={target_position.tolist()}")
        elif fail_count > 0:
            log_to_file(f"[IK] {tag}: partial failures={fail_count}/{hold_steps}")
        return ok

    try:
        can_pos0_world = get_world_pos(stage, can_path)
        ee_start0, _ = ik_solver.compute_end_effector_pose()
        if can_pos0_world is None:
            raise RuntimeError(f"Could not read can world pose for self-test: {can_path}")
        can_pos0_solver = world_to_solver_frame(np.array(can_pos0_world, dtype=float), ROBOT_TRANSLATE, ROBOT_YAW_DEG)
        log_to_file(
            f"Frame check: can_world={np.array(can_pos0_world, dtype=float).tolist()} "
            f"-> can_solver={can_pos0_solver.tolist()} (using robot spawn transform)"
        )
        _wp0, self_test_ok = run_self_test(ik_solver, can_pos0_solver, np.array(ee_start0, dtype=float))
        if (not self_test_ok) and ABORT_ON_SELF_TEST_FAIL:
            log_to_file("Aborting before episode loop because self-test failed (TIAGO_ABORT_ON_SELF_TEST_FAIL=1).")
            return
        wait_for_space_to_start()

        for ep in range(NUM_EPISODES):
            if not simulation_app.is_running():
                break

            if ep > 0:
                reset_episode_state()
            else:
                maybe_apply_can_pose_override(stage, can_path)
                if joint_prim_path is not None:
                    try:
                        stage.RemovePrim(joint_prim_path)
                    except Exception:
                        pass
                    joint_prim_path = None
                    attached_object = None

            key_state.toward_can_nudge_m = 0.0

            for _ in range(SETTLE_STEPS):
                process_h_forward_nudge()
                world.step(render=True)

            ee_start, _ = ik_solver.compute_end_effector_pose()
            can_pos_world = get_world_pos(stage, can_path)
            if can_pos_world is None:
                raise RuntimeError(f"Could not read can world pose: {can_path}")
            can_pos_solver = world_to_solver_frame(np.array(can_pos_world, dtype=float), ROBOT_TRANSLATE, ROBOT_YAW_DEG)
            wp, d_unit = compute_waypoints(can_pos_solver, can_pos_solver - np.array(ee_start, dtype=float))

            frames: List[Dict[str, Any]] = []
            log_to_file(f"Episode {episode_index} start ({ep + 1}/{NUM_EPISODES})")

            align_ok = drive_to_target(wp["align"], d_unit, WP_HOLD_STEPS, False, frames, "align")
            approach_ok = drive_to_target(wp["approach"], d_unit, APPROACH_STEPS, False, frames, "approach")
            grasp_ok = drive_to_target(wp["grasp"], d_unit, WP_HOLD_STEPS, False, frames, "pregrasp")

            ee_pos, _ = ik_solver.compute_end_effector_pose()
            dist = float(np.linalg.norm(np.array(ee_pos, dtype=float) - can_pos_solver))
            if (
                PHYSX_ATTACH
                and align_ok
                and approach_ok
                and grasp_ok
                and dist <= ATTACH_DISTANCE_MAX
                and joint_prim_path is None
            ):
                try:
                    gripper_prim = stage.GetPrimAtPath(gripper_body_path)
                    obj_prim = stage.GetPrimAtPath(can_path)
                    joint_prim = physx_utils.createJoint(stage, "Fixed", gripper_prim, obj_prim)
                    joint_prim_path = joint_prim.GetPath().pathString
                    attached_object = can_path
                    log_to_file(f"Attach success: dist={dist:.4f} path={attached_object} body={gripper_body_path}")
                except Exception as e:
                    log_to_file(f"Attach error: {e}")
            else:
                if not PHYSX_ATTACH:
                    log_to_file("Attach skipped: TIAGO_PHYSX_ATTACH=0 (no-contact scripted run).")
                else:
                    log_to_file(
                        f"Attach skipped: align_ok={align_ok} approach_ok={approach_ok} grasp_ok={grasp_ok} "
                        f"dist={dist:.4f} thresh={ATTACH_DISTANCE_MAX:.3f}"
                    )

            drive_to_target(wp["grasp"], d_unit, CLOSE_HOLD_STEPS, True, frames, "close_hold")
            lift_ok = drive_to_target(wp["lift"], d_unit, WP_HOLD_STEPS, True, frames, "lift")
            if not lift_ok:
                lift_alt = wp["grasp"] + np.array([0.0, 0.0, 0.06], dtype=float)
                log_to_file(f"[IK] lift fallback: trying smaller lift delta -> {lift_alt.tolist()}")
                drive_to_target(lift_alt, d_unit, WP_HOLD_STEPS, True, frames, "lift_fallback")
            drive_to_target(wp["retreat"], d_unit, WP_HOLD_STEPS, True, frames, "retreat_hold")
            drive_to_target(wp["retreat"], d_unit, CLOSE_HOLD_STEPS, False, frames, "retreat_open")

            if joint_prim_path is not None:
                try:
                    stage.RemovePrim(joint_prim_path)
                except Exception:
                    pass
                joint_prim_path = None
                attached_object = None
                log_to_file("Detach success")

            save_episode(episode_index, frames)
            episode_index += 1

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

