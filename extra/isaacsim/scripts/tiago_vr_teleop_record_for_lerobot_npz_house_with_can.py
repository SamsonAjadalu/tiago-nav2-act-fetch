"""
House-scene wrapper for TIAGo VR teleop NPZ recording.

Minimal wrapper: keep the base recorder's working VR/IK behavior and only override
scene loading, can path resolution, and layout for the house scene.

Layout: TIAGo stays at the origin (no root translate / yaw). The former robot placement is applied
inversely to /World/HouseScene so the room and can stay in the same pose relative to the base as
when the robot prim was offset. Tune TIAGO_HOUSE_LEGACY_* if a saved layout used different numbers.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tiago_repo_paths import house_usd_v1_path


def _load_base_module():
    base_path = Path(__file__).with_name("tiago_vr_teleop_record_for_lerobot_npz.py")
    spec = importlib.util.spec_from_file_location("tiago_record_npz_house_base", str(base_path))
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


def main() -> None:
    base = _load_base_module()

    house_usd = house_usd_v1_path()
    # Former robot-in-world pose (used only to compute the house inverse xform when env-as-origin is on).
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

    def _legacy_robot_world_matrix(tx: float, ty: float, tz: float, yaw_deg: float):
        """Same composition as AddTranslateOp then AddRotateZOp on the TIAGo root (p' = R * p + t)."""
        Mt = base.Gf.Matrix4d(1.0)
        Mt.SetTranslate(base.Gf.Vec3d(tx, ty, tz))
        Mr = base.Gf.Matrix4d(1.0)
        Mr.SetRotate(base.Gf.Rotation(base.Gf.Vec3d.ZAxis(), yaw_deg))
        return Mt * Mr

    # Keep house defaults used in inference.
    base.DEFAULT_JOINT_POSITIONS["torso_lift_joint"] = 0.35

    def _setup_stage_house(world) -> None:
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
                f"[HOUSE-REC] House root = inverse(legacy robot pose tx,ty,tz=({legacy_tx},{legacy_ty},{legacy_tz}) "
                f"yaw_deg={legacy_yaw_deg}); TIAGo root at origin."
            )
        else:
            base.log_to_file("[HOUSE-REC] TIAGO_HOUSE_ENV_CARRIES_LEGACY_ROBOT_POSE=0: house identity; move robot with TIAGO_* on robot prim (legacy).")

        can_path = _find_prim_path_by_name(stage, "_05_tomato_soup_can")
        if can_path is None:
            can_path = _find_first_path_with_any_token(stage, ["tomato", "soup", "can"])
        if can_path is None:
            raise RuntimeError("Could not locate can prim in house scene.")
        base.PICKABLE_OBJECT_PATHS = [can_path]
        base.log_to_file(f"[HOUSE-REC] Pickable can path: {can_path}")

    def _setup_robot_and_ik_house(world):
        base.carb.log_info("[HOUSE-REC] Setting up TIAGo and IK for house scene...")
        stage = base.omni.usd.get_context().get_stage()

        base.add_reference_to_stage(base.TIAGO_USD_PATH, "/World/Tiago")
        tiago_root = stage.GetPrimAtPath("/World/Tiago")
        if tiago_root.IsValid():
            xform = base.UsdGeom.Xformable(tiago_root)
            if place_env_not_robot:
                xform.ClearXformOpOrder()
                xform.AddTranslateOp().Set(base.Gf.Vec3d(0.0, 0.0, 0.0))
                xform.AddRotateZOp().Set(0.0)
                base.carb.log_info("[HOUSE-REC] TIAGo root translate (0,0,0) yaw 0.")
            else:
                xform.AddTranslateOp().Set(base.Gf.Vec3d(legacy_tx, legacy_ty, legacy_tz))
                xform.AddRotateZOp().Set(legacy_yaw_deg)
                base.carb.log_info(f"[HOUSE-REC] TIAGo root legacy pose tx,ty,tz,yaw=({legacy_tx},{legacy_ty},{legacy_tz},{legacy_yaw_deg}).")

        tiago = world.scene.add(base.Robot(prim_path="/World/Tiago/tiago_dual_functional", name="tiago"))

        world.reset()
        world.reset()
        for _ in range(30):
            world.step(render=False)

        joint_positions = tiago.get_joint_positions()
        if joint_positions is None:
            raise RuntimeError("[HOUSE-REC] Articulation not initialized!")

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
        base.log_to_file(f"[HOUSE-REC] EE frame: {base.RIGHT_EE_FRAME_NAME}, test_pos={ee_pos}")

        return tiago, art_kin_solver, gripper_joint_indices

    base._setup_stage = _setup_stage_house
    base.setup_robot_and_ik = _setup_robot_and_ik_house
    base.main()


if __name__ == "__main__":
    main()

