"""
House inference (ACT) + post-attach keyboard jog mode.

Behavior:
- SPACE toggles normal ACT inference.
- On attach (FixedJoint created), inference turns off and manual jog mode turns on.
- In manual mode, keyboard controls wrist/EE/gripper for demo positioning.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Set

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tiago_repo_paths import house_usd_v2_path


def _load_base_module():
    base_path = Path(__file__).with_name("tiago_inference_act_safe.py")
    spec = importlib.util.spec_from_file_location("tiago_inference_act_safe_base_jog", str(base_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load base script: {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_prim_path_by_name(stage, prim_name: str) -> str | None:
    for prim in stage.Traverse():
        if prim.IsValid() and prim.GetName() == prim_name:
            return str(prim.GetPath())
    return None


def _find_first_path_with_any_token(stage, tokens: list[str]) -> str | None:
    toks = [t.lower() for t in tokens]
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        p = str(prim.GetPath())
        p_l = p.lower()
        if any(t in p_l for t in toks):
            return p
    return None


def _sample_paths_containing(stage, token: str, limit: int = 12) -> list[str]:
    token_l = token.lower()
    matches: list[str] = []
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        p = str(prim.GetPath())
        if token_l in p.lower():
            matches.append(p)
            if len(matches) >= limit:
                break
    return matches


def _sample_all_paths(stage, limit: int = 30) -> list[str]:
    out: list[str] = []
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        out.append(str(prim.GetPath()))
        if len(out) >= limit:
            break
    return out


def main() -> None:
    base = _load_base_module()

    house_usd = house_usd_v2_path()
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

    # Jog settings (demo-safe small steps).
    jog_joint_step = float(os.environ.get("TIAGO_DEMO_JOG_JOINT_STEP_RAD", "0.01"))
    jog_gripper_step = float(os.environ.get("TIAGO_DEMO_JOG_GRIPPER_STEP", "0.003"))

    def _legacy_robot_world_matrix(tx: float, ty: float, tz: float, yaw_deg: float):
        mt = base.Gf.Matrix4d(1.0)
        mt.SetTranslate(base.Gf.Vec3d(tx, ty, tz))
        mr = base.Gf.Matrix4d(1.0)
        mr.SetRotate(base.Gf.Rotation(base.Gf.Vec3d.ZAxis(), yaw_deg))
        return mt * mr

    def _setup_stage_house(world):
        stage = base.omni.usd.get_context().get_stage()
        base.world = world
        world.scene.add_default_ground_plane()
        if not Path(house_usd).exists():
            raise FileNotFoundError(f"Scene USD not found: {house_usd}")
        base.add_reference_to_stage(house_usd, "/World/HouseScene")

        if place_env_not_robot:
            house_prim = stage.GetPrimAtPath("/World/HouseScene")
            if not house_prim.IsValid():
                raise RuntimeError("HouseScene prim missing after reference.")
            m = _legacy_robot_world_matrix(legacy_tx, legacy_ty, legacy_tz, legacy_yaw_deg)
            m_inv = m.GetInverse()
            xf = base.UsdGeom.Xformable(house_prim)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(m_inv)

        robot_path = _find_prim_path_by_name(stage, "tiago_dual_functional")
        can_path = _find_prim_path_by_name(stage, "_05_tomato_soup_can")
        if robot_path is None:
            base.add_reference_to_stage(base.TIAGO_USD_PATH, "/World/Tiago")
            robot_path = "/World/Tiago/tiago_dual_functional"
            tiago_root = stage.GetPrimAtPath("/World/Tiago")
            if tiago_root.IsValid():
                xform = base.UsdGeom.Xformable(tiago_root)
                xform.ClearXformOpOrder()
                if place_env_not_robot:
                    xform.AddTranslateOp().Set(base.Gf.Vec3d(0.0, 0.0, 0.0))
                    xform.AddRotateZOp().Set(0.0)
                else:
                    xform.AddTranslateOp().Set(base.Gf.Vec3d(legacy_tx, legacy_ty, legacy_tz))
                    xform.AddRotateZOp().Set(legacy_yaw_deg)
            tiago_hint = _sample_paths_containing(stage, "tiago")
            if tiago_hint:
                print("[HOUSE-SAFE-JOG] TIAGo not found in house USD; spawned fallback /World/Tiago.", flush=True)
            else:
                print(
                    "[HOUSE-SAFE-JOG] TIAGo not found in house USD; spawned fallback /World/Tiago. "
                    "House first paths:\n" + "\n".join(_sample_all_paths(stage)),
                    flush=True,
                )
        if can_path is None:
            can_path = _find_first_path_with_any_token(stage, ["tomato", "soup", "can"])
        if can_path is None:
            raise RuntimeError("Could not locate can object in loaded stage.")
        base.ROBOT_PRIM_PATH = robot_path
        base.PICKABLE_OBJECT_PATHS = [can_path]
        base.DEFAULT_JOINT_POSITIONS["torso_lift_joint"] = 0.35

    base._setup_stage = _setup_stage_house

    world = base.World(stage_units_in_meters=1.0, physics_dt=base.PHYSICS_DT_ASSUMED)
    base._setup_stage(world)
    try:
        tiago, art_kin_solver, gripper_joint_indices = base.setup_robot_and_ik(world)
    except Exception as e:
        base.carb.log_error(f"[TIAGO-INFERENCE-SAFE-JOG] setup failed: {e}")
        base.simulation_app.close()
        return

    all_joint_names = list(tiago.dof_names)
    arm_right_indices = base.arm_right_dof_indices(all_joint_names)

    stage = base.omni.usd.get_context().get_stage()
    camera_prim_path = base.find_gemini2_rgb_camera_prim_path(stage)
    if camera_prim_path is None:
        base.carb.log_error("[TIAGO-INFERENCE-SAFE-JOG] No Gemini2 Stream_rgb prim.")
        base.simulation_app.close()
        return
    rgb_annot = base.setup_rgb_annotator(camera_prim_path)

    gripper_body_path = base.find_gripper_rigidbody_prim(stage)
    if gripper_body_path is None:
        base.carb.log_error("[TIAGO-INFERENCE-SAFE-JOG] No gripper rigid body prim")
        base.simulation_app.close()
        return

    policy_cfg = base.PreTrainedConfig.from_pretrained(base.POLICY_DIR)
    preprocessor, postprocessor = base.make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=base.POLICY_DIR,
        preprocessor_overrides={"device_processor": {"device": base.DEVICE}},
    )
    policy = base.ACTPolicy.from_pretrained(base.POLICY_DIR, device=base.DEVICE)
    policy.eval()

    app_window = base.omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard() if app_window else None
    input_iface = base.carb.input.acquire_input_interface()

    class _Key:
        space_edge = False

    key_state = _Key()
    keys_down: Set[Any] = set()

    def _on_keyboard(event: base.carb.input.KeyboardEvent, *_args: Any) -> bool:
        if event.type == base.carb.input.KeyboardEventType.KEY_PRESS:
            keys_down.add(event.input)
            if event.input == base.carb.input.KeyboardInput.SPACE:
                key_state.space_edge = True
        elif event.type == base.carb.input.KeyboardEventType.KEY_RELEASE:
            keys_down.discard(event.input)
        return True

    sub = None
    if keyboard is not None:
        sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_keyboard)

    timeline = base.omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()

    frame_count = 0
    inference_active = False
    manual_mode = False
    cooldown_until = 0
    prev_min_finger_rad = float(base.GRIPPER_OPEN_POSITION)
    joint_prim_path: Optional[str] = None
    attached_object: Optional[str] = None
    last_applied_joint: Optional[np.ndarray] = None
    print(
        "[SAFE-JOG] SPACE toggle inference. After attach => manual mode ON.\n"
        "[SAFE-JOG] Manual keys: E/D(j3), R/F(j4), T/G(j5), Y/H(j6), U/J(j7), Z/X(gripper).",
        flush=True,
    )

    try:
        while base.simulation_app.is_running():
            if not timeline.is_playing():
                timeline.play()
            world.step(render=True)
            frame_count += 1

            if key_state.space_edge:
                key_state.space_edge = False
                if frame_count >= cooldown_until:
                    cooldown_until = frame_count + base.TOGGLE_DEBOUNCE_FRAMES
                    if not manual_mode:
                        inference_active = not inference_active
                        if inference_active:
                            policy.reset()
                            last_applied_joint = None
                        print(f"[SAFE-JOG] inference={'ON' if inference_active else 'OFF'}", flush=True)
                    else:
                        print("[SAFE-JOG] SPACE ignored in manual mode.", flush=True)

            jp_after = tiago.get_joint_positions()
            if jp_after is None:
                continue
            joint_np = np.array(jp_after, dtype=float)
            min_finger_rad = base.min_gripper_finger_rad(joint_np, gripper_joint_indices)
            trigger_value = base.virtual_trigger_from_gripper_joints(joint_np, gripper_joint_indices)

            rgb = base.get_rgb_hwc_uint8(rgb_annot)

            if manual_mode:
                q_cmd = joint_np.copy()
                # Arm joint jog (3..7)
                for jn, key_plus, key_minus in [
                    ("arm_right_3_joint", base.carb.input.KeyboardInput.E, base.carb.input.KeyboardInput.D),
                    ("arm_right_4_joint", base.carb.input.KeyboardInput.R, base.carb.input.KeyboardInput.F),
                    ("arm_right_5_joint", base.carb.input.KeyboardInput.T, base.carb.input.KeyboardInput.G),
                    ("arm_right_6_joint", base.carb.input.KeyboardInput.Y, base.carb.input.KeyboardInput.H),
                    ("arm_right_7_joint", base.carb.input.KeyboardInput.U, base.carb.input.KeyboardInput.J),
                ]:
                    if jn in all_joint_names:
                        j_idx = all_joint_names.index(jn)
                        if key_plus in keys_down:
                            q_cmd[j_idx] += jog_joint_step
                        if key_minus in keys_down:
                            q_cmd[j_idx] -= jog_joint_step
                # Gripper
                for gi in gripper_joint_indices:
                    if base.carb.input.KeyboardInput.Z in keys_down:
                        q_cmd[gi] = min(base.GRIPPER_OPEN_POSITION, q_cmd[gi] + jog_gripper_step)
                    if base.carb.input.KeyboardInput.X in keys_down:
                        q_cmd[gi] = max(base.GRIPPER_CLOSED_POSITION, q_cmd[gi] - jog_gripper_step)
                tiago.apply_action(base.ArticulationAction(joint_positions=q_cmd))
                last_applied_joint = q_cmd.copy()
            elif inference_active and rgb is not None:
                obs = base.preprocess_observation(
                    {"pixels": {"gemini2": rgb}, "agent_pos": joint_np.astype(np.float32)}
                )
                obs = preprocessor(obs)
                with base.torch.inference_mode():
                    action_t = policy.select_action(obs)
                action_t = postprocessor(action_t)
                action_np = action_t.squeeze(0).cpu().numpy().astype(float)
                if last_applied_joint is None or last_applied_joint.shape != action_np.shape:
                    last_applied_joint = joint_np.copy()
                near_dist, _ = base.nearest_pickable_distance_m(stage, np.array(art_kin_solver.compute_end_effector_pose()[0], dtype=float))
                in_near_guard = near_dist < base.NEAR_GUARD_DISTANCE_M
                action_np = base.clamp_action_vs_previous(
                    action_np,
                    last_applied_joint,
                    arm_right_indices,
                    base.MAX_ARM_DELTA_NEAR_RAD,
                    base.MAX_ARM_DELTA_FAR_RAD,
                    in_near_guard,
                )
                tiago.apply_action(base.ArticulationAction(joint_positions=action_np))
                last_applied_joint = action_np.copy()

            # Attach on close edge
            if (
                prev_min_finger_rad >= base.ATTACH_FINGER_MIN_RAD
                and min_finger_rad < base.ATTACH_FINGER_MIN_RAD
                and joint_prim_path is None
            ):
                try:
                    ee_p, _ = art_kin_solver.compute_end_effector_pose()
                    ee_arr = np.array([ee_p[0], ee_p[1], ee_p[2]], dtype=float)
                    pick = base.find_closest_pickable(stage, ee_arr)
                    if pick is not None:
                        obj_path, dist_pick = pick
                        gripper_prim = stage.GetPrimAtPath(gripper_body_path)
                        obj_prim = stage.GetPrimAtPath(obj_path)
                        joint_prim = base.physx_utils.createJoint(stage, "Fixed", gripper_prim, obj_prim)
                        joint_prim_path = joint_prim.GetPath().pathString
                        attached_object = obj_path
                        inference_active = False
                        manual_mode = True
                        try:
                            policy.reset()
                        except Exception:
                            pass
                        print(
                            f"[SAFE-JOG] ATTACH OK object={obj_path} dist={dist_pick:.4f} -> MANUAL MODE ON",
                            flush=True,
                        )
                except Exception as e:
                    base.carb.log_error(f"ATTACH error: {e}")

            # Optional detach if opened again
            if base.AUTO_DETACH_ENABLED and joint_prim_path is not None and min_finger_rad > base.DETACH_FINGER_MIN_RAD:
                try:
                    stage.RemovePrim(joint_prim_path)
                except Exception:
                    pass
                print(f"[SAFE-JOG] DETACH object={attached_object}", flush=True)
                joint_prim_path = None
                attached_object = None

            prev_min_finger_rad = min_finger_rad
    finally:
        if sub is not None:
            try:
                sub = None
            except Exception:
                pass
        base.simulation_app.close()


if __name__ == "__main__":
    main()

