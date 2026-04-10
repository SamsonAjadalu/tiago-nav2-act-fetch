"""
House-scene TIAGo VR teleop NPZ recording with automatic episode boundaries.

- Recording starts automatically after each reset.
- The moment a PhysX FixedJoint attach is created, the episode is saved and recording stops.
- Robot + can are reset, then the next episode starts automatically.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tiago_repo_paths import house_usd_v1_path


def _load_base_module():
    base_path = Path(__file__).with_name("tiago_vr_teleop_record_for_lerobot_npz.py")
    spec = importlib.util.spec_from_file_location("tiago_record_npz_house_base_auto", str(base_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load base recorder: {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_prim_path_by_name(stage, prim_name: str) -> Optional[str]:
    for prim in stage.Traverse():
        if prim.IsValid() and prim.GetName() == prim_name:
            return str(prim.GetPath())
    return None


def _find_first_path_with_any_token(stage, tokens: List[str]) -> Optional[str]:
    toks = [t.lower() for t in tokens]
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        p = str(prim.GetPath())
        p_l = p.lower()
        if any(t in p_l for t in toks):
            return p
    return None


def main() -> None:
    base = _load_base_module()

    house_usd = house_usd_v1_path()

    # Defaults match the discovered "perfect point".
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

    auto_episodes = int(os.environ.get("TIAGO_AUTO_EPISODES", "999999"))
    reset_settle_steps = int(os.environ.get("TIAGO_AUTO_RESET_SETTLE_STEPS", "20"))
    reset_cmd_steps = int(os.environ.get("TIAGO_AUTO_RESET_CMD_STEPS", "60"))

    # Keep house defaults used in inference.
    base.DEFAULT_JOINT_POSITIONS["torso_lift_joint"] = 0.35

    def _legacy_robot_world_matrix(tx: float, ty: float, tz: float, yaw_deg: float):
        Mt = base.Gf.Matrix4d(1.0)
        Mt.SetTranslate(base.Gf.Vec3d(tx, ty, tz))
        Mr = base.Gf.Matrix4d(1.0)
        Mr.SetRotate(base.Gf.Rotation(base.Gf.Vec3d.ZAxis(), yaw_deg))
        return Mt * Mr

    def _setup_stage_house(world) -> Tuple[str, str]:
        world.scene.add_default_ground_plane()
        stage = base.omni.usd.get_context().get_stage()
        if not Path(house_usd).exists():
            raise FileNotFoundError(f"Scene USD not found: {house_usd}")

        base.add_reference_to_stage(house_usd, "/World/HouseScene")

        if place_env_not_robot:
            house_prim = stage.GetPrimAtPath("/World/HouseScene")
            if not house_prim.IsValid():
                raise RuntimeError("HouseScene prim missing after reference.")
            M = _legacy_robot_world_matrix(legacy_tx, legacy_ty, legacy_tz, legacy_yaw_deg)
            M_inv = M.GetInverse()
            xf = base.UsdGeom.Xformable(house_prim)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(M_inv)
            base.log_to_file(
                f"[HOUSE-AUTO] House root = inverse(M) tx,ty,tz=({legacy_tx},{legacy_ty},{legacy_tz}) yaw_deg={legacy_yaw_deg}; "
                "TIAGo root at origin."
            )
        else:
            base.log_to_file(
                "[HOUSE-AUTO] TIAGO_HOUSE_ENV_CARRIES_LEGACY_ROBOT_POSE=0: house unchanged; robot uses legacy translate/yaw."
            )

        can_path = _find_prim_path_by_name(stage, "_05_tomato_soup_can")
        if can_path is None:
            can_path = _find_first_path_with_any_token(stage, ["tomato", "soup", "can"])
        if can_path is None:
            raise RuntimeError("Could not locate can prim in house scene.")
        base.PICKABLE_OBJECT_PATHS = [can_path]
        base.log_to_file(f"[HOUSE-AUTO] Pickable can path: {can_path}")
        return "/World/HouseScene", can_path

    def _setup_robot_and_ik_house(world):
        base.carb.log_info("[HOUSE-AUTO] Setting up TIAGo and IK for house scene...")
        stage = base.omni.usd.get_context().get_stage()

        base.add_reference_to_stage(base.TIAGO_USD_PATH, "/World/Tiago")
        tiago_root = stage.GetPrimAtPath("/World/Tiago")
        if tiago_root.IsValid():
            xform = base.UsdGeom.Xformable(tiago_root)
            xform.ClearXformOpOrder()
            if place_env_not_robot:
                xform.AddTranslateOp().Set(base.Gf.Vec3d(0.0, 0.0, 0.0))
                xform.AddRotateZOp().Set(0.0)
                base.carb.log_info("[HOUSE-AUTO] TIAGo root translate (0,0,0) yaw 0.")
            else:
                xform.AddTranslateOp().Set(base.Gf.Vec3d(legacy_tx, legacy_ty, legacy_tz))
                xform.AddRotateZOp().Set(legacy_yaw_deg)
                base.carb.log_info(
                    f"[HOUSE-AUTO] TIAGo root tx,ty,tz=({legacy_tx},{legacy_ty},{legacy_tz}) yaw_deg={legacy_yaw_deg}."
                )

        tiago = world.scene.add(base.Robot(prim_path="/World/Tiago/tiago_dual_functional", name="tiago"))

        world.reset()
        world.reset()
        for _ in range(30):
            world.step(render=False)

        joint_positions = tiago.get_joint_positions()
        if joint_positions is None:
            raise RuntimeError("[HOUSE-AUTO] Articulation not initialized!")

        all_joint_names = tiago.dof_names
        restored_positions = base.np.array(joint_positions, dtype=float)
        for joint_name, target_position in base.DEFAULT_JOINT_POSITIONS.items():
            if joint_name in all_joint_names:
                restored_positions[all_joint_names.index(joint_name)] = target_position

        for _ in range(50):
            tiago.apply_action(base.ArticulationAction(joint_positions=restored_positions))
            world.step(render=False)

        gripper_joint_indices = [all_joint_names.index(n) for n in base.GRIPPER_JOINT_NAMES if n in all_joint_names]

        lula_kin = base.LulaKinematicsSolver(
            robot_description_path=base.TIAGO_YAML_PATH,
            urdf_path=base.TIAGO_URDF_PATH,
        )
        art_kin_solver = base.ArticulationKinematicsSolver(tiago, lula_kin, base.RIGHT_EE_FRAME_NAME)
        ee_pos, _ = art_kin_solver.compute_end_effector_pose()
        base.log_to_file(f"[HOUSE-AUTO] EE frame: {base.RIGHT_EE_FRAME_NAME}, test_pos={ee_pos}")

        return tiago, art_kin_solver, gripper_joint_indices

    # --- Boot base systems (XR, ROS2 camera, etc.) using base helpers where possible.
    world = base.World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
    house_root_path, can_path = _setup_stage_house(world)
    tiago, art_kin_solver, gripper_joint_indices = _setup_robot_and_ik_house(world)

    stage = base.omni.usd.get_context().get_stage()

    # Save initial can LOCAL transform for resets.
    _can_prim0 = stage.GetPrimAtPath(can_path)
    _can_xf0 = base.UsdGeom.Xformable(_can_prim0)
    can_local0, _ = _can_xf0.GetLocalTransformation()

    gripper_body_path = base.find_gripper_rigidbody_prim(stage)
    if gripper_body_path is None:
        base.carb.log_error("[HOUSE-AUTO] Could not find gripper rigid body prim")
        return

    base.log_to_file(f"[HOUSE-AUTO] Using gripper rigid body prim: {gripper_body_path}")

    # Ensure ROS2 camera graph exists (for external video recorder).
    cam_node = None
    try:
        camera_prim_path = base.find_gemini2_rgb_camera_prim_path(stage)
        if camera_prim_path is not None:
            base.ensure_ros2_camera_graph(stage, camera_prim_path)
    except Exception:
        pass

    timeline = base.omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()

    episode_index = base.next_episode_index_from_npz_files()
    base.update_video_episode_marker("idle", -1, float(world.current_time))
    base.log_to_file(f"[HOUSE-AUTO] Auto episodes enabled. first_episode={episode_index}")

    def _reset_can_pose() -> None:
        prim = stage.GetPrimAtPath(can_path)
        if not prim.IsValid():
            return
        xf = base.UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(can_local0)

    def _reset_robot_joints() -> None:
        qp = tiago.get_joint_positions()
        if qp is None:
            return
        q_reset = base.np.array(qp, dtype=float)
        for jn, v in base.DEFAULT_JOINT_POSITIONS.items():
            if jn in tiago.dof_names:
                q_reset[tiago.dof_names.index(jn)] = v
        for _ in range(reset_cmd_steps):
            tiago.apply_action(base.ArticulationAction(joint_positions=q_reset))
            world.step(render=True)

    def _reset_episode(joint_prim_path: Optional[str]) -> None:
        if joint_prim_path is not None:
            try:
                stage.RemovePrim(joint_prim_path)
            except Exception:
                pass
        world.reset()
        for _ in range(reset_settle_steps):
            world.step(render=True)
        _reset_can_pose()
        for _ in range(reset_settle_steps):
            world.step(render=True)
        _reset_robot_joints()

    frame_count = 0
    prev_trigger = 0.0
    joint_prim_path: Optional[str] = None
    attached_object: Optional[str] = None

    episode_frames: List[Dict[str, Any]] = []
    episode_video_frames: List[base.np.ndarray] = []
    last_cam_stamp_s: float = -1.0
    episode_start_t = float(world.current_time)
    base.play_recording_started_sound()
    base.update_video_episode_marker("recording", episode_index, float(world.current_time))

    attach_stable_frames = int(os.environ.get("TIAGO_AUTO_ATTACH_STABLE_FRAMES", "1"))
    attach_seen = 0
    xr_core = base.XRCore()

    try:
        for _ep in range(auto_episodes):
            episode_frames = []
            episode_video_frames = []
            last_cam_stamp_s = -1.0
            episode_start_t = float(world.current_time)
            attach_seen = 0
            base.log_to_file(f"[HOUSE-AUTO] Episode {episode_index} REC START (auto)")
            base.play_recording_started_sound()
            base.update_video_episode_marker("recording", episode_index, float(world.current_time))

            while base.simulation_app.is_running():
                if not timeline.is_playing():
                    timeline.play()
                world.step(render=True)
                frame_count += 1

                vr_device = xr_core.get_input_device("/user/hand/right")
                if vr_device is None:
                    continue
                vr_pose = vr_device.get_virtual_world_pose()
                if vr_pose is None:
                    continue

                trigger_value = base.read_trigger_button(vr_device)

                # Attach/detach (same as base teleop). Stop episode as soon as attach is created.
                if trigger_value > base.ATTACH_TRIGGER_THRESHOLD and prev_trigger <= base.ATTACH_TRIGGER_THRESHOLD:
                    if joint_prim_path is None:
                        try:
                            ee_pos, _ = art_kin_solver.compute_end_effector_pose()
                            pick = base.find_closest_pickable(stage, ee_pos)
                            if pick is not None:
                                obj_path, _dist = pick
                                gripper_prim = stage.GetPrimAtPath(gripper_body_path)
                                obj_prim = stage.GetPrimAtPath(obj_path)
                                joint_prim = base.physx_utils.createJoint(stage, "Fixed", gripper_prim, obj_prim)
                                joint_prim_path = joint_prim.GetPath().pathString
                                attached_object = obj_path
                                base.log_to_file(f"[HOUSE-AUTO] PhysX attach: {attached_object}")
                        except Exception as e:
                            base.log_to_file(f"[HOUSE-AUTO] ATTACH error: {e}")

                if trigger_value < base.DETACH_TRIGGER_THRESHOLD and joint_prim_path is not None:
                    try:
                        stage.RemovePrim(joint_prim_path)
                    except Exception:
                        pass
                    joint_prim_path = None
                    attached_object = None
                    base.log_to_file("[HOUSE-AUTO] PhysX detach")

                # Control
                if frame_count % base.IK_SOLVE_INTERVAL == 0:
                    target_position, _target_orientation = base.vr_pose_to_target(vr_pose)
                    ik_action, ik_success = art_kin_solver.compute_inverse_kinematics(target_position=target_position)
                    if ik_success:
                        combined_action, applied_joint_positions = base.combine_arm_and_gripper_action(
                            tiago, ik_action, gripper_joint_indices, trigger_value
                        )
                        tiago.apply_action(combined_action)
                    else:
                        applied_joint_positions = None

                # Record
                if frame_count % base.RECORD_EVERY_N_FRAMES == 0:
                    if cam_node is not None and base.ROS2_AVAILABLE:
                        try:
                            base.rclpy.spin_once(cam_node, timeout_sec=0.0)
                            rgb, stamp_s = cam_node.get_latest()
                            if rgb is not None and stamp_s > last_cam_stamp_s:
                                episode_video_frames.append(rgb)
                                last_cam_stamp_s = stamp_s
                        except Exception:
                            pass

                    current_joint_positions = tiago.get_joint_positions()
                    if current_joint_positions is not None:
                        action_joint_pos = (
                            applied_joint_positions
                            if applied_joint_positions is not None
                            else base.np.array(current_joint_positions, dtype=float)
                        )
                        episode_frames.append(
                            {
                                "t": float(world.current_time),
                                "vr_pos": base.np.array(
                                    [
                                        vr_pose.ExtractTranslation()[0],
                                        vr_pose.ExtractTranslation()[1],
                                        vr_pose.ExtractTranslation()[2],
                                    ],
                                    dtype=float,
                                ),
                                "vr_quat": base.np.array(
                                    [
                                        vr_pose.ExtractRotationQuat().GetReal(),
                                        vr_pose.ExtractRotationQuat().GetImaginary()[0],
                                        vr_pose.ExtractRotationQuat().GetImaginary()[1],
                                        vr_pose.ExtractRotationQuat().GetImaginary()[2],
                                    ],
                                    dtype=float,
                                ),
                                "joint_pos": base.np.array(current_joint_positions, dtype=float),
                                "action_joint_pos": base.np.array(action_joint_pos, dtype=float),
                                "trigger": float(trigger_value),
                                "attached": bool(joint_prim_path is not None),
                                "picked_object_path": attached_object or "",
                            }
                        )

                # Stop on attach
                if joint_prim_path is not None:
                    attach_seen += 1
                else:
                    attach_seen = 0
                if attach_seen >= attach_stable_frames:
                    base.log_to_file(f"[HOUSE-AUTO] Episode {episode_index} STOP on attach (stable_frames={attach_seen})")
                    base._save_episode(episode_index, episode_frames, episode_video_frames)
                    base.play_recording_stopped_sound()
                    base.update_video_episode_marker("idle", episode_index, float(world.current_time))
                    episode_index += 1

                    _reset_episode(joint_prim_path)
                    joint_prim_path = None
                    attached_object = None
                    prev_trigger = 0.0
                    break

                prev_trigger = trigger_value

    finally:
        try:
            base.update_video_episode_marker("idle", -1, float(world.current_time))
        except Exception:
            pass
        base.simulation_app.close()


if __name__ == "__main__":
    main()

