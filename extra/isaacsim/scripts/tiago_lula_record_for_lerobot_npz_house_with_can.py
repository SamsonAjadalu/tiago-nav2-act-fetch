"""
Record house-scene TIAGo pick episodes into NPZ using scripted Lula IK waypoints.

This path removes VR/OpenXR from data collection while keeping the same NPZ schema
used by the existing recorder.

Run:
  extra/isaacsim/run_python.sh extra/isaacsim/scripts/tiago_lula_record_for_lerobot_npz_house_with_can.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tiago_repo_paths import house_usd_v1_path


def _install_xr_stub_if_missing() -> None:
    """Allow loading the base recorder in Isaac runs without XR extension."""
    if "omni.kit.xr.core" in sys.modules:
        return
    try:
        import omni.kit.xr.core  # type: ignore # noqa: F401
        return
    except Exception:
        pass

    # Build a tiny module chain: omni -> omni.kit -> omni.kit.xr -> omni.kit.xr.core
    omni_mod = sys.modules.setdefault("omni", types.ModuleType("omni"))
    kit_mod = sys.modules.setdefault("omni.kit", types.ModuleType("omni.kit"))
    xr_mod = sys.modules.setdefault("omni.kit.xr", types.ModuleType("omni.kit.xr"))
    core_mod = types.ModuleType("omni.kit.xr.core")

    class _XRCoreStub:  # pragma: no cover - runtime shim only
        @staticmethod
        def get_singleton():
            return None

    core_mod.XRCore = _XRCoreStub
    sys.modules["omni.kit.xr.core"] = core_mod
    setattr(xr_mod, "core", core_mod)
    setattr(kit_mod, "xr", xr_mod)
    setattr(omni_mod, "kit", kit_mod)


def _load_base_module():
    _install_xr_stub_if_missing()
    base_path = Path(__file__).with_name("tiago_vr_teleop_record_for_lerobot_npz.py")
    spec = importlib.util.spec_from_file_location("tiago_record_npz_lula_house_base", str(base_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load base recorder: {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_prim_path_by_name(stage, prim_name: str) -> str | None:
    for prim in stage.Traverse():
        if prim.IsValid() and prim.GetName() == prim_name:
            return str(prim.GetPath())
    return None


def _find_first_path_with_any_token(stage, tokens: List[str]) -> str | None:
    toks = [t.lower() for t in tokens]
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        p = str(prim.GetPath())
        p_l = p.lower()
        if any(t in p_l for t in toks):
            return p
    return None


def _upsert_translate_and_rotate_xyz(base, prim_path: str, translate_xyz: Tuple[float, float, float], rotate_xyz_deg: Tuple[float, float, float]) -> None:
    stage = base.omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found for transform update: {prim_path}")
    xf = base.UsdGeom.Xformable(prim)
    t_op = None
    r_op = None
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == base.UsdGeom.XformOp.TypeTranslate and t_op is None:
            t_op = op
        elif op.GetOpType() == base.UsdGeom.XformOp.TypeRotateXYZ and r_op is None:
            r_op = op
    if t_op is None:
        t_op = xf.AddTranslateOp()
    if r_op is None:
        r_op = xf.AddRotateXYZOp()
    t_op.Set(base.Gf.Vec3d(*translate_xyz))
    r_op.Set(base.Gf.Vec3f(*rotate_xyz_deg))


def _compute_scripted_waypoints(base, ee_start: Any, can_pos: Any) -> Dict[str, Any]:
    ee = base.np.array(ee_start, dtype=float)
    can = base.np.array(can_pos, dtype=float)

    to_can = can - ee
    to_can[2] = 0.0
    norm = float(base.np.linalg.norm(to_can))
    if norm < 1e-6:
        approach_dir = base.np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        approach_dir = to_can / norm

    pre_grasp = can - (0.18 * approach_dir) + base.np.array([0.0, 0.0, 0.10], dtype=float)
    grasp = can - (0.03 * approach_dir) + base.np.array([0.0, 0.0, 0.02], dtype=float)
    lift = grasp + base.np.array([0.0, 0.0, 0.18], dtype=float)
    retreat = pre_grasp.copy()

    return {
        "pre_grasp": pre_grasp,
        "grasp": grasp,
        "lift": lift,
        "retreat": retreat,
    }


def main() -> None:
    base = _load_base_module()

    house_usd = house_usd_v1_path()
    robot_translate = (-0.05, 1.41377, 0.0)
    robot_yaw_deg = 0.0

    can_translate = (
        float(os.environ.get("TIAGO_HOUSE_CAN_X", "-0.63146")),
        float(os.environ.get("TIAGO_HOUSE_CAN_Y", "2.05573")),
        float(os.environ.get("TIAGO_HOUSE_CAN_Z", "0.81218")),
    )
    can_rotate_xyz_deg = (
        float(os.environ.get("TIAGO_HOUSE_CAN_ROLL_DEG", "-90.0")),
        float(os.environ.get("TIAGO_HOUSE_CAN_PITCH_DEG", "0.0")),
        float(os.environ.get("TIAGO_HOUSE_CAN_YAW_DEG", "0.0")),
    )

    num_episodes = int(os.environ.get("TIAGO_SCRIPTED_EPISODES", "10"))
    wp_hold_steps = int(os.environ.get("TIAGO_SCRIPTED_WP_HOLD_STEPS", "40"))
    close_hold_steps = int(os.environ.get("TIAGO_SCRIPTED_CLOSE_HOLD_STEPS", "24"))
    settle_steps = int(os.environ.get("TIAGO_SCRIPTED_SETTLE_STEPS", "20"))

    base.DEFAULT_JOINT_POSITIONS["torso_lift_joint"] = 0.35

    resolved_can_path: Dict[str, str] = {"path": ""}

    def _setup_stage_house(world) -> None:
        world.scene.add_default_ground_plane()
        stage = base.omni.usd.get_context().get_stage()
        if not Path(house_usd).exists():
            raise FileNotFoundError(f"Scene USD not found: {house_usd}")

        base.add_reference_to_stage(house_usd, "/World/HouseScene")
        can_path = _find_prim_path_by_name(stage, "_05_tomato_soup_can")
        if can_path is None:
            can_path = _find_first_path_with_any_token(stage, ["tomato", "soup", "can"])
        if can_path is None:
            raise RuntimeError("Could not locate can prim in house scene.")

        _upsert_translate_and_rotate_xyz(base, can_path, can_translate, can_rotate_xyz_deg)

        base.PICKABLE_OBJECT_PATHS = [can_path]
        resolved_can_path["path"] = can_path
        base.log_to_file(f"[HOUSE-LULA-REC] Pickable can path: {can_path}")
        base.log_to_file(f"[HOUSE-LULA-REC] Can pose set to xyz={can_translate}, rpy_deg={can_rotate_xyz_deg}")

    def _setup_robot_and_ik_house(world):
        base.carb.log_info("[HOUSE-LULA-REC] Setting up TIAGo and IK for house scene...")
        stage = base.omni.usd.get_context().get_stage()

        base.add_reference_to_stage(base.TIAGO_USD_PATH, "/World/Tiago")
        tiago_root = stage.GetPrimAtPath("/World/Tiago")
        if tiago_root.IsValid():
            xf = base.UsdGeom.Xformable(tiago_root)
            xf.AddTranslateOp().Set(base.Gf.Vec3d(*robot_translate))
            xf.AddRotateZOp().Set(robot_yaw_deg)

        tiago = world.scene.add(base.Robot(prim_path="/World/Tiago/tiago_dual_functional", name="tiago"))

        world.reset()
        world.reset()
        for _ in range(30):
            world.step(render=False)

        joint_positions = tiago.get_joint_positions()
        if joint_positions is None:
            raise RuntimeError("[HOUSE-LULA-REC] Articulation not initialized.")

        all_joint_names = tiago.dof_names
        restored_positions = base.np.array(joint_positions, dtype=float)
        for joint_name, target_position in base.DEFAULT_JOINT_POSITIONS.items():
            if joint_name in all_joint_names:
                restored_positions[all_joint_names.index(joint_name)] = target_position

        for _ in range(50):
            tiago.apply_action(base.ArticulationAction(joint_positions=restored_positions))
            world.step(render=False)

        gripper_joint_indices = []
        for gripper_name in base.GRIPPER_JOINT_NAMES:
            if gripper_name in all_joint_names:
                gripper_joint_indices.append(all_joint_names.index(gripper_name))

        lula_kin = base.LulaKinematicsSolver(
            robot_description_path=base.TIAGO_YAML_PATH,
            urdf_path=base.TIAGO_URDF_PATH,
        )
        art_kin_solver = base.ArticulationKinematicsSolver(tiago, lula_kin, base.RIGHT_EE_FRAME_NAME)
        ee_pos, _ = art_kin_solver.compute_end_effector_pose()
        base.log_to_file(f"[HOUSE-LULA-REC] EE frame: {base.RIGHT_EE_FRAME_NAME}, start_pos={ee_pos}")
        return tiago, art_kin_solver, gripper_joint_indices

    base._setup_stage = _setup_stage_house
    base.setup_robot_and_ik = _setup_robot_and_ik_house

    world = base.World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
    base._setup_stage(world)
    tiago, art_kin_solver, gripper_joint_indices = base.setup_robot_and_ik(world)

    stage = base.omni.usd.get_context().get_stage()
    gripper_body_path = base.find_gripper_rigidbody_prim(stage)
    if gripper_body_path is None:
        raise RuntimeError("[HOUSE-LULA-REC] Could not locate right gripper rigid body prim.")

    timeline = base.omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()

    base.update_video_episode_marker("idle", -1, float(world.current_time))
    episode_index = base.next_episode_index_from_npz_files()
    base.log_to_file(
        f"[HOUSE-LULA-REC] Starting scripted recording: episodes={num_episodes}, "
        f"next_episode_index={episode_index}"
    )

    joint_prim_path: Optional[str] = None
    attached_object: Optional[str] = None

    def _record_frame(
        episode_frames: List[Dict[str, Any]],
        trigger_value: float,
        target_position: Any,
        action_joint_pos: Any,
    ) -> None:
        current_joint_positions = tiago.get_joint_positions()
        if current_joint_positions is None:
            return
        episode_frames.append(
            {
                "t": float(world.current_time),
                "vr_pos": base.np.array(target_position, dtype=float),
                "vr_quat": base.np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
                "joint_pos": base.np.array(current_joint_positions, dtype=float),
                "action_joint_pos": base.np.array(action_joint_pos, dtype=float),
                "trigger": float(trigger_value),
                "attached": bool(joint_prim_path is not None),
                "picked_object_path": attached_object or "",
            }
        )

    def _drive_to_target(
        target_position: Any,
        hold_steps: int,
        trigger_value: float,
        episode_frames: List[Dict[str, Any]],
    ) -> bool:
        success_any = False
        last_action = tiago.get_joint_positions()
        for _ in range(hold_steps):
            if not base.simulation_app.is_running():
                return success_any
            world.step(render=True)
            ik_action, ik_success = art_kin_solver.compute_inverse_kinematics(target_position=target_position)
            if ik_success:
                combined_action, applied_joint_positions = base.combine_arm_and_gripper_action(
                    tiago, ik_action, gripper_joint_indices, trigger_value
                )
                tiago.apply_action(combined_action)
                success_any = True
                if applied_joint_positions is not None:
                    last_action = applied_joint_positions
            if last_action is not None:
                _record_frame(episode_frames, trigger_value, target_position, last_action)
        return success_any

    try:
        for ep in range(num_episodes):
            if not base.simulation_app.is_running():
                break

            if resolved_can_path["path"]:
                _upsert_translate_and_rotate_xyz(base, resolved_can_path["path"], can_translate, can_rotate_xyz_deg)
            if joint_prim_path is not None:
                try:
                    stage.RemovePrim(joint_prim_path)
                except Exception:
                    pass
                joint_prim_path = None
                attached_object = None

            for _ in range(settle_steps):
                world.step(render=True)

            ee_start, _ = art_kin_solver.compute_end_effector_pose()
            can_pos = base.get_world_pos(stage, resolved_can_path["path"])
            if can_pos is None:
                raise RuntimeError(f"[HOUSE-LULA-REC] Could not read can world pose: {resolved_can_path['path']}")
            wp = _compute_scripted_waypoints(base, ee_start, can_pos)

            episode_frames: List[Dict[str, Any]] = []
            episode_video_frames: List[Any] = []
            base.update_video_episode_marker("recording", episode_index, float(world.current_time))
            base.log_to_file(
                f"[HOUSE-LULA-REC] Episode {episode_index} START ({ep + 1}/{num_episodes}); "
                f"can={can_pos.tolist()}"
            )
            base.play_recording_started_sound()

            _drive_to_target(wp["pre_grasp"], wp_hold_steps, 0.0, episode_frames)
            _drive_to_target(wp["grasp"], wp_hold_steps, 0.0, episode_frames)

            ee_pos, _ = art_kin_solver.compute_end_effector_pose()
            pick = base.find_closest_pickable(stage, ee_pos)
            if pick is not None and joint_prim_path is None:
                obj_path, _ = pick
                try:
                    gripper_prim = stage.GetPrimAtPath(gripper_body_path)
                    obj_prim = stage.GetPrimAtPath(obj_path)
                    joint_prim = base.physx_utils.createJoint(stage, "Fixed", gripper_prim, obj_prim)
                    joint_prim_path = joint_prim.GetPath().pathString
                    attached_object = obj_path
                    base.log_to_file(f"[HOUSE-LULA-REC] PhysX attach: {attached_object}")
                except Exception as e:
                    base.log_to_file(f"[HOUSE-LULA-REC] ATTACH error: {e}")

            _drive_to_target(wp["grasp"], close_hold_steps, 1.0, episode_frames)
            _drive_to_target(wp["lift"], wp_hold_steps, 1.0, episode_frames)
            _drive_to_target(wp["retreat"], wp_hold_steps, 1.0, episode_frames)
            _drive_to_target(wp["retreat"], close_hold_steps, 0.0, episode_frames)

            if joint_prim_path is not None:
                try:
                    stage.RemovePrim(joint_prim_path)
                except Exception:
                    pass
                joint_prim_path = None
                attached_object = None
                base.log_to_file("[HOUSE-LULA-REC] PhysX detach")

            base._save_episode(episode_index, episode_frames, episode_video_frames)
            base.play_recording_stopped_sound()
            base.update_video_episode_marker("idle", episode_index, float(world.current_time))
            base.log_to_file(
                f"[HOUSE-LULA-REC] Episode {episode_index} STOP (saved). frames={len(episode_frames)}"
            )
            episode_index += 1

    finally:
        base.simulation_app.close()


if __name__ == "__main__":
    main()
