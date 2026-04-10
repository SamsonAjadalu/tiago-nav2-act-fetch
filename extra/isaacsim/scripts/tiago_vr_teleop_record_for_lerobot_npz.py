"""
Record TIAGo VR teleop episodes into NPZ tensors for later conversion to LeRobot.

Why NPZ:
- LeRobot Dataset v3.x schema + recording API depends on installing `lerobot`,
  which often fails inside Isaac environments due to extra system deps.
- NPZ stores state, action, and VR pose for offline conversion (LeRobot in a separate env).

What this script records (per frame during an episode):
- `t`: sim time (seconds, float)
- `vr_pos`: right VR hand/controller position, shape (3,)
- `vr_quat`: controller quaternion (w,x,y,z), shape (4,)
- `joint_pos`: all articulated joint positions from Isaac, shape (num_dofs,)
- `action_joint_pos`: commanded joint positions that frame, shape (num_dofs,)
- `trigger`: trigger value [0,1]
- `attached`: whether a FixedJoint between gripper and object exists
- `picked_object_path`: USD prim path if attached else empty string

Episode segmentation (Quest / OpenXR):
- **Left Y** or **right B** use **`touch` rising edge** (finger down = one edge). Release the button,
  then press again to stop — `click` is often **stuck at 1.0** in Isaac logs and cannot be used alone.
- A short debounce after each toggle reduces duplicate edges from runtime glitches.
- **Audio cues**: one **high** beep = recording on; **two low** beeps = saved & off.

If `xrCreateInstance failed` appears, fix SteamVR / Monado / Quest Link OpenXR runtime — controller clicks won't work until XR initializes.

Trigger still controls gripper open/close and PhysX attach/detach; recording is independent of attach.

Run:
  extra/isaacsim/run_python.sh extra/isaacsim/scripts/tiago_vr_teleop_record_for_lerobot_npz.py \\
    --enable omni.kit.xr.profile.vr --enable isaacsim.xr.openxr
"""

from __future__ import annotations

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "enable_ros2": True,
    }
)

import math
import os
import sys
import re
import struct
import json
import tempfile
import subprocess
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
from pathlib import Path
import tempfile

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tiago_repo_paths import (
    default_isaacsim_data_root,
    default_table_objects_usd_path,
    default_tiago_usd_path,
    default_urdf_path,
    default_yaml_path,
)

import carb
import numpy as np
import threading
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.kit.xr.core import XRCore
from pxr import Gf, Usd, UsdGeom, UsdPhysics
import omni.usd
import omni.timeline

import omni.graph.core as og
import usdrt
from isaacsim.core.utils import extensions

from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver

from omni.physx.scripts import utils as physx_utils

# ROS2 (for Gemini2 RGB frames)
ROS2_IMPORT_ERROR = ""
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image as RosImage

    ROS2_AVAILABLE = True
except Exception as e:
    ROS2_AVAILABLE = False
    ROS2_IMPORT_ERROR = repr(e)
    Node = None  # type: ignore[misc, assignment]
    RosImage = None  # type: ignore[misc, assignment]
    rclpy = None  # type: ignore[misc, assignment]

try:
    import imageio  # type: ignore

    IMAGEIO_AVAILABLE = True
except Exception:
    IMAGEIO_AVAILABLE = False

# ============================================================================
# Configuration (aligned with teleop IK / scene)
# ============================================================================

TIAGO_USD_PATH = default_tiago_usd_path()
TABLE_OBJECTS_USD_PATH = default_table_objects_usd_path()
TIAGO_URDF_PATH = default_urdf_path()
TIAGO_YAML_PATH = default_yaml_path()

RIGHT_EE_FRAME_NAME = "gripper_right_grasping_frame"

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

HAND_POSE_OFFSET = np.array([0.0, 0.0, 0.0], dtype=float)
HAND_POSE_SCALE = 1.0

IK_SOLVE_INTERVAL = 1

ATTACH_DISTANCE_MAX = 0.12
ATTACH_TRIGGER_THRESHOLD = 0.9
DETACH_TRIGGER_THRESHOLD = 0.3

PICKABLE_OBJECT_PATHS: List[str] = [
    "/World/Table/_05_tomato_soup_can/_05_tomato_soup_can",
    "/World/Table/_06_mustard_bottle/_06_mustard_bottle",
    "/World/PickBox",
]

# Output
_DEFAULT_DATA = str(default_isaacsim_data_root())
_ISAAC_DATA = os.environ.get("TIAGO_ISAACSIM_DATA_ROOT", _DEFAULT_DATA)
RECORD_OUT_DIR = os.environ.get(
    "TIAGO_LEROBOT_NPZ_DIR", os.path.join(_ISAAC_DATA, "tiago_lerobot_npz")
)
RECORD_EVERY_N_FRAMES = 1
MAX_EPISODE_SECONDS = 120.0  # safety cap while recording (Y stops sooner)

# Gemini2 RGB recording
GEMINI2_RGB_TOPIC = "/gemini2/color/image_raw"
ROS_CAMERA_GRAPH_PATH = "/ROS2_Gemini2_RGB_Pub"
ROS_CAMERA_FRAME_ID = "gemini2_rgb_frame"
ROS_CAMERA_INFO_TOPIC = "/gemini2/camera_info"
# Offscreen render product (VR often makes viewport 0 invalid → SDG "Width 0 or height 0").
ROS_CAMERA_WIDTH = 1280
ROS_CAMERA_HEIGHT = 720
VIDEO_FPS = 30
# Ignore record-toggle edges for this many sim frames after a successful toggle (OpenXR sometimes
# glitches click/touch; also prevents double-fire on one physical press).
RECORD_TOGGLE_DEBOUNCE_FRAMES = 30
VIDEOS_DIR = os.path.join(RECORD_OUT_DIR, "videos")

LOG_FILE_PATH = os.path.join(RECORD_OUT_DIR, "tiago_record_for_lerobot_log.txt")

# Video marker for external MP4 writer (doesn't require Isaac `rclpy`)
VIDEO_EPISODE_MARKER_PATH = os.path.join(RECORD_OUT_DIR, "video_episode_marker.json")

_EPISODE_NPZ_RE = re.compile(r"^episode_(\d{4})\.npz$")


def next_episode_index_from_npz_files() -> int:
    """
    Next episode index: max(N)+1 for ``episode_NNNN.npz`` under RECORD_OUT_DIR, else 0.
    """
    best = -1
    try:
        if not os.path.isdir(RECORD_OUT_DIR):
            return 0
        for name in os.listdir(RECORD_OUT_DIR):
            m = _EPISODE_NPZ_RE.match(name)
            if m:
                best = max(best, int(m.group(1)))
    except Exception:
        pass
    return max(0, best + 1)


# Recording feedback: short WAV tones played via paplay (PipeWire/Pulse) or aplay (ALSA)
RECORDING_SOUND_SAMPLE_RATE = 44100
_sound_backend_missing_logged = False


def _pcm_sine_mono16(freq_hz: float, duration_s: float, sample_rate: int, volume: float = 0.22) -> bytes:
    n = max(1, int(sample_rate * duration_s))
    fade = max(1, int(sample_rate * 0.004))
    out = bytearray()
    for i in range(n):
        edge = 1.0
        if i < fade:
            edge = i / fade
        if i > n - fade:
            edge = min(edge, (n - i) / fade)
        sample = int(32767 * volume * edge * math.sin(2 * math.pi * freq_hz * i / sample_rate))
        out.extend(struct.pack("<h", sample))
    return bytes(out)


def _wav_mono16(pcm_s16le: bytes, sample_rate: int) -> bytes:
    n = len(pcm_s16le)
    chunk_size = 36 + n
    return (
        struct.pack("<4sI4s", b"RIFF", chunk_size, b"WAVE")
        + struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + struct.pack("<4sI", b"data", n)
        + pcm_s16le
    )


def _play_wav_bytes_async(wav_data: bytes) -> None:
    """Non-blocking: plays on a daemon thread so the sim loop is not stalled."""

    def _run() -> None:
        global _sound_backend_missing_logged
        path: Optional[str] = None
        fd = -1
        try:
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.write(fd, wav_data)
            os.close(fd)
            fd = -1
            for cmd in (["paplay", path], ["aplay", "-q", path]):
                try:
                    subprocess.run(
                        cmd,
                        check=False,
                        timeout=3.0,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return
                except FileNotFoundError:
                    continue
                except Exception:
                    return
            if not _sound_backend_missing_logged:
                _sound_backend_missing_logged = True
                log_to_file(
                    "WARNING: Neither `paplay` nor `aplay` found — recording start/stop sounds disabled."
                )
        except Exception as e:
            if not _sound_backend_missing_logged:
                _sound_backend_missing_logged = True
                log_to_file(f"WARNING: Recording cue sound failed: {e}")
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except Exception:
                    pass
            if path:
                try:
                    os.unlink(path)
                except Exception:
                    pass

    threading.Thread(target=_run, daemon=True).start()


def play_recording_started_sound() -> None:
    """Single higher-pitched beep = now recording."""
    sr = RECORDING_SOUND_SAMPLE_RATE
    pcm = _pcm_sine_mono16(880.0, 0.11, sr)
    _play_wav_bytes_async(_wav_mono16(pcm, sr))


def play_recording_stopped_sound() -> None:
    """Two lower beeps = saved and not recording."""
    sr = RECORDING_SOUND_SAMPLE_RATE
    gap = b"\x00\x00" * int(sr * 0.04)
    one = _pcm_sine_mono16(330.0, 0.09, sr, volume=0.2)
    pcm = one + gap + _pcm_sine_mono16(330.0, 0.09, sr, volume=0.2)
    _play_wav_bytes_async(_wav_mono16(pcm, sr))


def log_to_file(message: str) -> None:
    try:
        os.makedirs(RECORD_OUT_DIR, exist_ok=True)
        if not os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, "w") as f:
                f.write("TIAGo VR record (state/action/VR pose) log\n")
                f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n")
        with open(LOG_FILE_PATH, "a") as f:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            f.write(f"[{timestamp}] {message}\n")
            f.flush()
        carb.log_info(message)
    except Exception as e:
        carb.log_warn(f"[TIAGO-LEROBOT-NPZ] log_to_file failed: {e}")


def update_video_episode_marker(state: str, episode_index: int, t_sim_s: float) -> None:
    """
    Used by an external ROS2->MP4 recorder.
    state: 'recording' | 'idle'

    Writes via temp file + os.replace so readers never see a half-written JSON (which used to
    look like ``idle`` and closed the MP4 instantly).
    """
    try:
        payload = {"state": state, "episode_index": int(episode_index), "t_sim_s": float(t_sim_s)}
        out_dir = os.path.dirname(VIDEO_EPISODE_MARKER_PATH) or "."
        os.makedirs(out_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=out_dir, prefix="video_episode_marker_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, VIDEO_EPISODE_MARKER_PATH)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        carb.log_warn(f"[TIAGO-LEROBOT-NPZ] update_video_episode_marker failed: {e}")


def resolve_ros_camera_prim_path(stage, stream_or_camera_path: str) -> str:
    """
    ROS2 offscreen graphs need a USD Camera prim. Gemini setups often point at *Stream_rgb*,
    which may not be a UsdGeom.Camera — walk ancestors to find one.
    """
    try:
        prim = stage.GetPrimAtPath(stream_or_camera_path)
        cur = prim
        for _ in range(24):
            if not cur or not cur.IsValid():
                break
            if cur.IsA(UsdGeom.Camera):
                return str(cur.GetPath())
            cur = cur.GetParent()
    except Exception:
        pass
    return stream_or_camera_path


def find_gemini2_rgb_camera_prim_path(stage) -> Optional[str]:
    """
    Best-effort: find the USD prim path that the existing Gemini2 pipeline uses
    as the RGB stream render source (often ends with `Stream_rgb`).
    """
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
        # Fallback: any prim ending in Stream_rgb
        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            if path.lower().endswith("stream_rgb"):
                return path
    except Exception:
        return None
    return None


def ensure_ros2_camera_graph(stage, camera_prim_path: str) -> bool:
    """
    Create a standalone ROS2 camera publishing Action Graph (RGB + CameraInfo).

    Uses **IsaacCreateRenderProduct** (fixed WxH) instead of viewport 0. In VR, the main viewport
    is often 0×0 and breaks SDG / ROS2CameraHelper with "Width 0 or height 0 is not valid".
    """
    try:
        extensions.enable_extension("isaacsim.ros2.bridge")
    except Exception as e:
        log_to_file(f"WARNING: enable isaacsim.ros2.bridge failed: {e}")
        return False

    cam_path = resolve_ros_camera_prim_path(stage, camera_prim_path)
    if cam_path != camera_prim_path:
        log_to_file(f"ROS camera: using Camera prim {cam_path} (resolved from {camera_prim_path})")

    try:
        # Remove stale viewport graph before rebuilding offscreen ROS camera graph.
        try:
            old = stage.GetPrimAtPath(ROS_CAMERA_GRAPH_PATH)
            if old.IsValid():
                omni.usd.get_context().delete_prim(ROS_CAMERA_GRAPH_PATH)
                log_to_file(f"Removed prior OmniGraph at {ROS_CAMERA_GRAPH_PATH} for offscreen rebuild")
        except Exception as e:
            log_to_file(f"WARNING: could not delete old ROS graph prim: {e}")

        keys = og.Controller.Keys
        graph_cfg = {
            "graph_path": ROS_CAMERA_GRAPH_PATH,
            "evaluator_name": "push",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
        }
        graph_actions = {
            keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnTick"),
                ("createRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("cameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("cameraHelperInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ],
            keys.CONNECT: [
                ("OnTick.outputs:tick", "createRenderProduct.inputs:execIn"),
                ("createRenderProduct.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                ("createRenderProduct.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
                ("createRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
                ("createRenderProduct.outputs:renderProductPath", "cameraHelperInfo.inputs:renderProductPath"),
            ],
            keys.SET_VALUES: [
                ("createRenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path(cam_path)]),
                ("createRenderProduct.inputs:width", int(ROS_CAMERA_WIDTH)),
                ("createRenderProduct.inputs:height", int(ROS_CAMERA_HEIGHT)),
                ("cameraHelperRgb.inputs:frameId", ROS_CAMERA_FRAME_ID),
                ("cameraHelperRgb.inputs:topicName", GEMINI2_RGB_TOPIC),
                ("cameraHelperRgb.inputs:type", "rgb"),
                ("cameraHelperInfo.inputs:frameId", ROS_CAMERA_FRAME_ID),
                ("cameraHelperInfo.inputs:topicName", ROS_CAMERA_INFO_TOPIC),
            ],
        }

        (graph_handle, _, _, _) = og.Controller.edit(graph_cfg, graph_actions)
        og.Controller.evaluate_sync(graph_handle)
        log_to_file(
            f"Created/updated ROS2 camera publisher graph at {ROS_CAMERA_GRAPH_PATH} "
            f"cameraPrim={cam_path} resolution={ROS_CAMERA_WIDTH}x{ROS_CAMERA_HEIGHT} topic={GEMINI2_RGB_TOPIC}"
        )
        return True
    except Exception as e:
        log_to_file(f"WARNING: Failed creating ROS2 camera graph: {e}")
        return False


def _decode_ros_image(msg: "RosImage") -> Optional[np.ndarray]:
    """Decode ROS Image encodings into uint8 HxWx3 RGB."""
    try:
        h = int(msg.height)
        w = int(msg.width)
        enc = (msg.encoding or "").lower()
        data = msg.data
        if h <= 0 or w <= 0 or data is None:
            return None

        if enc in ("rgb8", "bgr8"):
            arr = np.frombuffer(data, dtype=np.uint8)
            expected = h * w * 3
            if arr.size < expected:
                return None
            arr = arr[:expected].reshape((h, w, 3))
            if enc == "bgr8":
                arr = arr[..., ::-1]
            return arr

        if enc in ("rgba8", "bgra8"):
            arr = np.frombuffer(data, dtype=np.uint8)
            expected = h * w * 4
            if arr.size < expected:
                return None
            arr = arr[:expected].reshape((h, w, 4))
            if enc == "bgra8":
                arr = arr[..., [2, 1, 0, 3]]
            return arr[..., :3]

        # Fallback assume packed 3 channel
        arr = np.frombuffer(data, dtype=np.uint8)
        expected = h * w * 3
        if arr.size >= expected:
            return arr[:expected].reshape((h, w, 3))
        return None
    except Exception:
        return None


if ROS2_AVAILABLE:

    class Gemini2RgbSubscriber(Node):
        def __init__(self, topic: str):
            super().__init__("tiago_gemini2_rgb_recorder")
            self._lock = threading.Lock()
            self._latest_rgb: Optional[np.ndarray] = None
            self._latest_stamp_s: float = 0.0
            self._sub = self.create_subscription(RosImage, topic, self._cb, 10)

        def _cb(self, msg: "RosImage") -> None:
            rgb = _decode_ros_image(msg)
            if rgb is None:
                return
            stamp_s = 0.0
            try:
                stamp_s = float(msg.header.stamp.sec) + 1e-9 * float(msg.header.stamp.nanosec)
            except Exception:
                stamp_s = 0.0
            with self._lock:
                self._latest_rgb = rgb
                self._latest_stamp_s = stamp_s

        def get_latest(self) -> Tuple[Optional[np.ndarray], float]:
            with self._lock:
                if self._latest_rgb is None:
                    return None, 0.0
                return self._latest_rgb.copy(), float(self._latest_stamp_s)


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


def setup_robot_and_ik(world: World):
    carb.log_info("[TIAGO-LEROBOT-NPZ] Setting up robot and IK solver...")

    add_reference_to_stage(TIAGO_USD_PATH, "/World/Tiago")

    tiago = world.scene.add(Robot(prim_path="/World/Tiago/tiago_dual_functional", name="tiago"))

    world.reset()
    world.reset()
    for _ in range(30):
        world.step(render=False)

    joint_positions = tiago.get_joint_positions()
    if joint_positions is None:
        raise RuntimeError("[TIAGO-LEROBOT-NPZ] Articulation not initialized!")

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

    test_pos, _ = art_kin_solver.compute_end_effector_pose()
    carb.log_info(f"[TIAGO-LEROBOT-NPZ] EE frame: {RIGHT_EE_FRAME_NAME}, test_pos={test_pos}")
    log_to_file(f"EE frame: {RIGHT_EE_FRAME_NAME}, test_pos={test_pos}")
    log_to_file(f"Gripper joints found: {gripper_joint_indices}")

    return tiago, art_kin_solver, gripper_joint_indices


def setup_xr():
    xr_core = XRCore.get_singleton()
    if xr_core is None:
        carb.log_error("[TIAGO-LEROBOT-NPZ] XRCore not available")
        return None
    carb.log_info("[TIAGO-LEROBOT-NPZ] XRCore OK")
    return xr_core


def vr_pose_to_target(vr_pose):
    pos = vr_pose.ExtractTranslation()
    vr_pos_raw = np.array([pos[0], pos[1], pos[2]], dtype=float)
    target_position = (vr_pos_raw * HAND_POSE_SCALE) + HAND_POSE_OFFSET

    quat = vr_pose.ExtractRotationQuat()
    q_w = quat.GetReal()
    q_xyz = quat.GetImaginary()
    target_orientation = np.array([q_w, q_xyz[0], q_xyz[1], q_xyz[2]], dtype=float)
    return target_position, target_orientation


def _blend_quat(q_target: np.ndarray, q_easy: np.ndarray, alpha: float) -> np.ndarray:
    q = (1.0 - alpha) * q_target + alpha * q_easy
    norm = np.linalg.norm(q)
    if norm < 1e-6:
        return q_easy.copy()
    return q / norm


def read_trigger_button(vr_device) -> float:
    try:
        trigger_value = vr_device.get_input_gesture_value("trigger", "value")
        if trigger_value is not None:
            return float(np.clip(trigger_value, 0.0, 1.0))
    except Exception:
        pass
    return 0.0


def _read_digital_button_max(device, pairs: List[Tuple[str, str]]) -> float:
    """
    Isaac XRCore: first arg is **input component** (e.g. `y`, `trigger`), second is **gesture**
    (e.g. `click`, `value`). Quest Touch face buttons use `click` / `touch`, not `value`.
    """
    if device is None or not pairs:
        return 0.0
    best = 0.0
    for input_name, gesture_name in pairs:
        try:
            v = device.get_input_gesture_value(input_name, gesture_name)
            if v is None:
                continue
            fv = float(v)
            if math.isnan(fv):
                continue
            best = max(best, float(np.clip(fv, 0.0, 1.0)))
        except Exception:
            continue
    return best


# Quest 2/3 left: OpenXR input `y` — gestures `click`, `touch` (see meta_touch_controller_quest_2_bindings.json)
_LEFT_RECORD_BUTTON_PAIRS: List[Tuple[str, str]] = [
    ("y", "click"),
    ("y", "touch"),
    ("Y", "click"),
    ("secondary_button", "click"),
    ("secondary_button", "value"),
    ("menu", "click"),
]
# Right-hand alias (common mix-up): B = `b` + `click`
_RIGHT_RECORD_ALIAS_PAIRS: List[Tuple[str, str]] = [
    ("b", "click"),
    ("b", "touch"),
    ("B", "click"),
    ("secondary_button", "click"),
    ("secondary_button", "value"),
]

# Recording toggle: **touch only**. User logs show `y:click` stuck at 1.0 for entire sessions,
# so click-based edges cannot reliably start/stop. Touch: press → release → press = two edges.
_LEFT_RECORD_TOGGLE_PAIRS: List[Tuple[str, str]] = [
    ("y", "touch"),
    ("Y", "touch"),
]
_RIGHT_RECORD_TOGGLE_PAIRS: List[Tuple[str, str]] = [
    ("b", "touch"),
    ("B", "touch"),
]


def read_recording_toggle_input(vr_left_device, vr_right_device) -> float:
    """Max of left Y-style and right B-style **touch** inputs only (0..1)."""
    a = _read_digital_button_max(vr_left_device, _LEFT_RECORD_TOGGLE_PAIRS)
    b = _read_digital_button_max(vr_right_device, _RIGHT_RECORD_TOGGLE_PAIRS)
    return max(a, b)


def _read_pairs_debug(device, pairs: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
    """
    Debug helper: return each (input, gesture)->value for the given XR device.
    Example entry name: "y:click".
    """
    if device is None:
        return [(f"{inp}:{gest}", 0.0) for inp, gest in pairs]
    out: List[Tuple[str, float]] = []
    for input_name, gesture_name in pairs:
        v = 0.0
        try:
            vv = device.get_input_gesture_value(input_name, gesture_name)
            if vv is not None:
                v = float(np.clip(vv, 0.0, 1.0))
        except Exception:
            v = 0.0
        out.append((f"{input_name}:{gesture_name}", v))
    return out


def log_xr_controller_inputs_once(vr_left, vr_right, log_fn) -> None:
    """One-shot XR input name dump for debugging binding mismatches."""
    for label, dev in (("left", vr_left), ("right", vr_right)):
        if dev is None:
            log_fn(f"XR inputs dump [{label}]: device is None")
            continue
        try:
            inames = dev.get_input_names()
            log_fn(f"XR inputs dump [{label}]: components={len(inames)} -> {[str(x) for x in inames]}")
            for comp in inames[:24]:
                try:
                    gnames = dev.get_input_gesture_names(comp)
                    log_fn(f"  [{label}] {comp!s} -> {[str(g) for g in gnames]}")
                except Exception as e:
                    log_fn(f"  [{label}] {comp!s} gestures: (error {e})")
        except Exception as e:
            log_fn(f"XR inputs dump [{label}] failed: {e}")


def _save_episode(
    episode_index: int,
    episode_frames: List[Dict[str, Any]],
    episode_video_frames: List[np.ndarray],
) -> None:
    out_path = os.path.join(RECORD_OUT_DIR, f"episode_{episode_index:04d}.npz")
    out_mp4 = os.path.join(VIDEOS_DIR, f"episode_{episode_index:04d}.mp4")
    log_to_file(f"Episode {episode_index} SAVE, frames={len(episode_frames)} -> {out_path}")
    if len(episode_frames) > 0:
        stacked = {k: [] for k in episode_frames[0].keys()}
        for fr in episode_frames:
            for k, v in fr.items():
                stacked[k].append(v)
        np_save = {k: np.array(seq) for k, seq in stacked.items()}
        np.savez_compressed(out_path, **np_save)
    if IMAGEIO_AVAILABLE and len(episode_video_frames) > 0:
        try:
            with imageio.get_writer(out_mp4, fps=VIDEO_FPS) as w:
                for fr in episode_video_frames:
                    w.append_data(fr)
            log_to_file(f"Saved MP4: {out_mp4} frames={len(episode_video_frames)} fps={VIDEO_FPS}")
        except Exception as e:
            log_to_file(f"WARNING: Failed to write MP4 {out_mp4}: {e}")


def combine_arm_and_gripper_action(tiago: Robot, ik_action: ArticulationAction, gripper_joint_indices: list, trigger_value: float):
    current_joint_positions = tiago.get_joint_positions()
    if current_joint_positions is None:
        return ik_action, None

    all_joint_positions = np.array(current_joint_positions, dtype=float)

    if ik_action.joint_positions is not None:
        ik_positions = np.array(ik_action.joint_positions, dtype=float)
        if hasattr(ik_action, "joint_indices") and ik_action.joint_indices is not None:
            ik_indices = np.array(ik_action.joint_indices, dtype=int)
            for i, idx in enumerate(ik_indices):
                if idx < len(all_joint_positions) and i < len(ik_positions):
                    all_joint_positions[idx] = ik_positions[i]
        else:
            num_ik_joints = len(ik_positions)
            if num_ik_joints <= len(all_joint_positions):
                all_joint_positions[:num_ik_joints] = ik_positions

    gripper_position = GRIPPER_OPEN_POSITION - (trigger_value * (GRIPPER_OPEN_POSITION - GRIPPER_CLOSED_POSITION))
    for gripper_idx in gripper_joint_indices:
        if gripper_idx < len(all_joint_positions):
            all_joint_positions[gripper_idx] = gripper_position

    combined_action = ArticulationAction(joint_positions=all_joint_positions)
    return combined_action, all_joint_positions


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
    mat = xform.ComputeLocalToWorldTransform(0.0)
    t = mat.ExtractTranslation()
    return np.array([t[0], t[1], t[2]], dtype=float)


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


def main():
    carb.log_info("[TIAGO-LEROBOT-NPZ] Starting episode recorder...")

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
    _setup_stage(world)

    try:
        tiago, art_kin_solver, gripper_joint_indices = setup_robot_and_ik(world)
    except Exception as e:
        carb.log_error(f"[TIAGO-LEROBOT-NPZ] setup_robot_and_ik failed: {e}")
        return

    xr_core = setup_xr()
    if xr_core is None:
        return

    # Gemini2 RGB subscriber (ROS2)
    cam_node: Optional[Any] = None
    if ROS2_AVAILABLE:
        try:
            if not rclpy.ok():
                rclpy.init(args=None)
            cam_node = Gemini2RgbSubscriber(GEMINI2_RGB_TOPIC)
            log_to_file(f"Subscribed Gemini2 RGB: {GEMINI2_RGB_TOPIC}")
        except Exception as e:
            cam_node = None
            log_to_file(f"WARNING: ROS2 Gemini2 subscribe failed: {e}")
    else:
        log_to_file(f"WARNING: ROS2 not available; Gemini2 video disabled. import_error={ROS2_IMPORT_ERROR}")

    if not IMAGEIO_AVAILABLE:
        log_to_file("WARNING: imageio not available; MP4 writing disabled.")
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    stage = omni.usd.get_context().get_stage()
    # Ensure a ROS2 publisher graph exists even if UI Action Graph editing is confusing.
    camera_prim_path = find_gemini2_rgb_camera_prim_path(stage)
    if camera_prim_path is not None:
        ensure_ros2_camera_graph(stage, camera_prim_path)
    else:
        log_to_file("WARNING: Could not find Gemini2 RGB stream prim; ROS2 camera graph not created.")
    gripper_body_path = find_gripper_rigidbody_prim(stage)
    if gripper_body_path is None:
        carb.log_error("[TIAGO-LEROBOT-NPZ] Could not find gripper rigid body prim")
        return

    log_to_file(f"Using gripper rigid body prim: {gripper_body_path}")

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()

    frame_count = 0
    prev_trigger = 0.0
    prev_record_toggle = 0.0
    left_device_warned = False
    right_device_warned = False
    last_right_missing_warn_frame = -10_000
    xr_inputs_logged = False

    joint_prim_path: Optional[str] = None
    attached_object: Optional[str] = None

    episode_active = False
    episode_start_t = 0.0
    episode_frames: List[Dict[str, Any]] = []
    episode_index = next_episode_index_from_npz_files()
    applied_joint_positions: Optional[np.ndarray] = None
    episode_video_frames: List[np.ndarray] = []
    last_cam_stamp_s: float = -1.0
    record_toggle_cooldown_until = 0

    # Prevent accidental "start on startup" if the record button is already active.
    try:
        vr_left0 = xr_core.get_input_device("/user/hand/left")
        vr_right0 = xr_core.get_input_device("/user/hand/right")
        prev_record_toggle = read_recording_toggle_input(vr_left0, vr_right0)
        log_to_file(f"Initial toggle_value={prev_record_toggle:.3f} (prev_record_toggle initialized)")
    except Exception as e:
        log_to_file(f"WARNING: Failed to init prev_record_toggle: {e}")

    # External ROS2->MP4 recorder uses this marker.
    update_video_episode_marker("idle", -1, float(world.current_time))
    log_to_file(
        f"Next episode index = {episode_index} (auto from episode_*.npz in {RECORD_OUT_DIR}; "
        f"MP4 still needs the separate ROS2 script + /gemini2/color/image_raw)"
    )

    try:
        while simulation_app.is_running():
            if not timeline.is_playing():
                timeline.play()
            world.step(render=True)
            frame_count += 1

            if cam_node is not None:
                try:
                    rclpy.spin_once(cam_node, timeout_sec=0.0)
                except Exception:
                    pass

            vr_left = xr_core.get_input_device("/user/hand/left")
            vr_device = xr_core.get_input_device("/user/hand/right")

            if vr_left is None and not left_device_warned:
                log_to_file(
                    "INFO: No /user/hand/left — toggle recording with **left Y** or **right B** (touch: "
                    "press to start, release, press again to stop). If neither works, see XR inputs dump after ~4s."
                )
                left_device_warned = True

            if vr_device is None:
                # Repeated warning so XR failures are obvious during long runs.
                if (not right_device_warned) or (frame_count - last_right_missing_warn_frame >= 120):
                    log_to_file(
                        "WARNING: No /user/hand/right controller detected. "
                        "Robot will not follow VR hand pose. "
                        "Check OpenXR runtime (xrCreateInstance), headset/controller tracking, and restart XR."
                    )
                    right_device_warned = True
                    last_right_missing_warn_frame = frame_count

            toggle_value = read_recording_toggle_input(vr_left, vr_device)

            if not xr_inputs_logged and frame_count >= 240:
                xr_inputs_logged = True
                log_xr_controller_inputs_once(vr_left, vr_device, log_to_file)

            # Digital button rising edge: toggle recording (start / stop + save).
            toggle_thresh = 0.5
            toggle_edge = toggle_value > toggle_thresh and prev_record_toggle <= toggle_thresh
            if toggle_edge and frame_count >= record_toggle_cooldown_until:
                record_toggle_cooldown_until = frame_count + RECORD_TOGGLE_DEBOUNCE_FRAMES
                if not episode_active:
                    episode_active = True
                    episode_start_t = float(world.current_time)
                    episode_frames = []
                    episode_video_frames = []
                    last_cam_stamp_s = -1.0
                    left_dbg = _read_digital_button_max(vr_left, _LEFT_RECORD_BUTTON_PAIRS)
                    right_dbg = _read_digital_button_max(vr_device, _RIGHT_RECORD_ALIAS_PAIRS)
                    left_pairs = _read_pairs_debug(vr_left, _LEFT_RECORD_BUTTON_PAIRS)
                    right_pairs = _read_pairs_debug(vr_device, _RIGHT_RECORD_ALIAS_PAIRS)
                    msg = (
                        f"Episode {episode_index} REC START (toggle). "
                        f"toggle_value={toggle_value:.3f} left_dbg={left_dbg:.3f} right_dbg={right_dbg:.3f} | "
                        f"left=[{', '.join([f'{k}={v:.2f}' for k,v in left_pairs])}] | "
                        f"right=[{', '.join([f'{k}={v:.2f}' for k,v in right_pairs])}]"
                    )
                    log_to_file(msg)
                    carb.log_info(msg)
                    print(msg, flush=True)
                    play_recording_started_sound()
                    update_video_episode_marker("recording", episode_index, float(world.current_time))
                else:
                    _save_episode(episode_index, episode_frames, episode_video_frames)
                    play_recording_stopped_sound()
                    update_video_episode_marker("idle", episode_index, float(world.current_time))
                    episode_index += 1
                    episode_active = False
                    episode_frames = []
                    episode_video_frames = []
                    left_dbg = _read_digital_button_max(vr_left, _LEFT_RECORD_BUTTON_PAIRS)
                    right_dbg = _read_digital_button_max(vr_device, _RIGHT_RECORD_ALIAS_PAIRS)
                    left_pairs = _read_pairs_debug(vr_left, _LEFT_RECORD_BUTTON_PAIRS)
                    right_pairs = _read_pairs_debug(vr_device, _RIGHT_RECORD_ALIAS_PAIRS)
                    msg = (
                        f"Episode {episode_index - 1} REC STOP (saved). "
                        f"toggle_value={toggle_value:.3f} left_dbg={left_dbg:.3f} right_dbg={right_dbg:.3f} | "
                        f"left=[{', '.join([f'{k}={v:.2f}' for k,v in left_pairs])}] | "
                        f"right=[{', '.join([f'{k}={v:.2f}' for k,v in right_pairs])}]"
                    )
                    log_to_file(msg)
                    carb.log_info(msg)
                    print(msg, flush=True)

            if vr_device is None:
                prev_record_toggle = toggle_value
                continue

            trigger_value = read_trigger_button(vr_device)
            vr_pose = vr_device.get_virtual_world_pose()
            if vr_pose is None:
                prev_record_toggle = toggle_value
                continue

            # Attach/detach triggers (same as working teleop) — does not start/stop recording
            if trigger_value > ATTACH_TRIGGER_THRESHOLD and prev_trigger <= ATTACH_TRIGGER_THRESHOLD:
                if joint_prim_path is None:
                    try:
                        ee_pos, _ = art_kin_solver.compute_end_effector_pose()
                        pick = find_closest_pickable(stage, ee_pos)
                        if pick is not None:
                            obj_path, _dist = pick
                            gripper_prim = stage.GetPrimAtPath(gripper_body_path)
                            obj_prim = stage.GetPrimAtPath(obj_path)
                            joint_prim = physx_utils.createJoint(stage, "Fixed", gripper_prim, obj_prim)
                            joint_prim_path = joint_prim.GetPath().pathString
                            attached_object = obj_path
                            log_to_file(f"PhysX attach: {attached_object}")
                    except Exception as e:
                        log_to_file(f"ATTACH error: {e}")

            if trigger_value < DETACH_TRIGGER_THRESHOLD and joint_prim_path is not None:
                try:
                    stage.RemovePrim(joint_prim_path)
                except Exception:
                    pass
                joint_prim_path = None
                attached_object = None
                log_to_file("PhysX detach")

            # Control + recording
            if frame_count % IK_SOLVE_INTERVAL == 0:
                target_position, target_orientation = vr_pose_to_target(vr_pose)

                # Keep it stable: position-only (like simple teleop)
                ik_action, ik_success = art_kin_solver.compute_inverse_kinematics(target_position=target_position)

                applied_joint_positions = None
                if ik_success:
                    combined_action, applied_joint_positions = combine_arm_and_gripper_action(
                        tiago, ik_action, gripper_joint_indices, trigger_value
                    )
                    tiago.apply_action(combined_action)

            # Record per frame (while episode_active)
            if episode_active and (frame_count % RECORD_EVERY_N_FRAMES == 0):
                if cam_node is not None:
                    rgb, stamp_s = cam_node.get_latest()
                    if rgb is not None and stamp_s > last_cam_stamp_s:
                        episode_video_frames.append(rgb)
                        last_cam_stamp_s = stamp_s

                current_joint_positions = tiago.get_joint_positions()
                if current_joint_positions is not None:
                    # If IK skipped this frame, applied_joint_positions may still be last command; else use sensed q.
                    action_joint_pos = (
                        applied_joint_positions
                        if applied_joint_positions is not None
                        else np.array(current_joint_positions, dtype=float)
                    )

                    episode_frames.append(
                        {
                            "t": float(world.current_time),
                            "vr_pos": np.array([vr_pose.ExtractTranslation()[0], vr_pose.ExtractTranslation()[1], vr_pose.ExtractTranslation()[2]], dtype=float),
                            "vr_quat": np.array(
                                [
                                    vr_pose.ExtractRotationQuat().GetReal(),
                                    vr_pose.ExtractRotationQuat().GetImaginary()[0],
                                    vr_pose.ExtractRotationQuat().GetImaginary()[1],
                                    vr_pose.ExtractRotationQuat().GetImaginary()[2],
                                ],
                                dtype=float,
                            ),
                            "joint_pos": np.array(current_joint_positions, dtype=float),
                            "action_joint_pos": np.array(action_joint_pos, dtype=float),
                            "trigger": float(trigger_value),
                            "attached": bool(joint_prim_path is not None),
                            "picked_object_path": attached_object or "",
                        }
                    )

                # Periodic log while recording
                if frame_count % 60 == 0:
                    msg = (
                        f"[REC] episode={episode_index} t={float(world.current_time):.2f} "
                        f"frames={len(episode_frames)} video_frames={len(episode_video_frames)}"
                    )
                    carb.log_info(msg)
                    print(msg, flush=True)

                # Safety cap: auto-save and stop (does not detach object)
                if float(world.current_time) - episode_start_t > MAX_EPISODE_SECONDS:
                    log_to_file(f"Episode {episode_index} timed out; forcing save.")
                    _save_episode(episode_index, episode_frames, episode_video_frames)
                    play_recording_stopped_sound()
                    update_video_episode_marker("idle", episode_index, float(world.current_time))
                    episode_index += 1
                    episode_active = False
                    episode_frames = []
                    episode_video_frames = []

            prev_trigger = trigger_value
            prev_record_toggle = toggle_value

    finally:
        if cam_node is not None:
            try:
                cam_node.destroy_node()
            except Exception:
                pass
        if ROS2_AVAILABLE:
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass
        simulation_app.close()


if __name__ == "__main__":
    main()

