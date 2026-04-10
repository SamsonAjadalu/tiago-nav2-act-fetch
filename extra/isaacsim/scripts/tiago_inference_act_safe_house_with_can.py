"""
Safe ACT inference launcher for a custom scene USD that already contains TIAGo + can.

This script keeps the original safe inference script untouched and only overrides:
- scene loading
- robot prim path
- pickable object paths

Target scene (default: ``<repo>/usd/house_with_can2.usd``; override ``TIAGO_HOUSE_USD_PATH``).

Expected source prims in that USD:
  /Root/tiago_dual_functional
  /Root/_05_tomato_soup_can
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tiago_repo_paths import house_usd_v2_path


def _load_base_module():
    base_path = Path(__file__).with_name("tiago_inference_act_safe.py")
    spec = importlib.util.spec_from_file_location("tiago_inference_act_safe_base", str(base_path))
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

    house_usd = house_usd_v2_path()

    # Default guesses; these will be replaced by auto-detected paths if found.
    base.ROBOT_PRIM_PATH = "/World/HouseScene/Root/tiago_dual_functional"
    base.PICKABLE_OBJECT_PATHS = ["/World/HouseScene/Root/_05_tomato_soup_can"]
    # Use the torso max from URDF (upper=0.35) for this scene.
    base.DEFAULT_JOINT_POSITIONS["torso_lift_joint"] = 0.35
    spawn_robot_translate = (-0.05, 1.41377, 0.0)
    spawn_robot_yaw_deg = 162.0

    def _setup_stage_house(world):
        stage = base.omni.usd.get_context().get_stage()
        base.world = world
        world.scene.add_default_ground_plane()
        if not Path(house_usd).exists():
            raise FileNotFoundError(f"Scene USD not found: {house_usd}")
        base.add_reference_to_stage(house_usd, "/World/HouseScene")
        robot_path = _find_prim_path_by_name(stage, "tiago_dual_functional")
        can_path = _find_prim_path_by_name(stage, "_05_tomato_soup_can")
        if robot_path is None:
            # Fallback: spawn the same TIAGo USD used by the known-working safe script.
            base.add_reference_to_stage(base.TIAGO_USD_PATH, "/World/Tiago")
            robot_path = "/World/Tiago/tiago_dual_functional"
            tiago_root = stage.GetPrimAtPath("/World/Tiago")
            if tiago_root.IsValid():
                xform = base.UsdGeom.Xformable(tiago_root)
                xform.AddTranslateOp().Set(base.Gf.Vec3d(*spawn_robot_translate))
                xform.AddRotateZOp().Set(spawn_robot_yaw_deg)
            tiago_hint = _sample_paths_containing(stage, "tiago")
            if tiago_hint:
                print("[HOUSE-SAFE] TIAGo not found in house USD; spawned fallback /World/Tiago.", flush=True)
            else:
                print(
                    "[HOUSE-SAFE] TIAGo not found in house USD; spawned fallback /World/Tiago. "
                    "House first paths:\n" + "\n".join(_sample_all_paths(stage)),
                    flush=True,
                )
        if can_path is None:
            can_path = _find_first_path_with_any_token(stage, ["tomato", "soup", "can"])
        if can_path is None:
            hints = _sample_paths_containing(stage, "can") + _sample_paths_containing(stage, "tomato")
            hint_msg = "\n".join(hints) if hints else "<no paths containing 'can' or 'tomato'>"
            raise RuntimeError(
                "Could not locate can object in loaded stage.\n"
                f"Sample matching paths:\n{hint_msg}"
            )
        base.ROBOT_PRIM_PATH = robot_path
        base.PICKABLE_OBJECT_PATHS = [can_path]

    base._setup_stage = _setup_stage_house
    base.main()


if __name__ == "__main__":
    main()
