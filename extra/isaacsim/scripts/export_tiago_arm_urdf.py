"""
Export TIAGo arm-only (wheels hidden) to URDF via isaacsim.asset.exporter.urdf.

1. Load TIAGo USD
2. Hide wheel / suspension links
3. Write temp USD, convert with UsdToUrdf (requires extension enabled before import)

Launcher: extra/isaacsim/run_python.sh this_script.py
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tiago_repo_paths import default_tiago_usd_path, tiago_repo_root

from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.asset.exporter.urdf")

from nvidia.srl.from_usd.to_urdf import UsdToUrdf
from pxr import Usd, UsdGeom

TIAGO_USD_PATH = os.environ.get("TIAGO_EXPORT_USD_IN", "").strip() or default_tiago_usd_path()
OUTPUT_URDF_PATH = os.environ.get("TIAGO_EXPORT_URDF_OUT", "").strip() or str(
    tiago_repo_root() / "tiago_arm_only.urdf"
)

stage = Usd.Stage.Open(TIAGO_USD_PATH)
if not stage:
    print(f"Failed to open USD: {TIAGO_USD_PATH}")
    simulation_app.close()
    raise SystemExit(1)

print(f"Loaded USD: {TIAGO_USD_PATH}")

wheel_links = [
    "/Root/tiago_dual_functional/wheel_front_left_link",
    "/Root/tiago_dual_functional/wheel_front_right_link",
    "/Root/tiago_dual_functional/wheel_rear_left_link",
    "/Root/tiago_dual_functional/wheel_rear_right_link",
    "/Root/tiago_dual_functional/suspension_front_left_link",
    "/Root/tiago_dual_functional/suspension_front_right_link",
    "/Root/tiago_dual_functional/suspension_rear_right_link",
    "/Root/tiago_dual_functional/suspension_rear_left_link",
]

print("Hiding wheel links...")
for wheel_path in wheel_links:
    prim = stage.GetPrimAtPath(wheel_path)
    if prim:
        UsdGeom.Imageable(prim).MakeInvisible()
        print(f"  Hidden: {wheel_path}")

temp_usd_path = "/tmp/tiago_arm_only.usd"
stage.Export(temp_usd_path)
print(f"Saved modified USD: {temp_usd_path}")

out_path = Path(OUTPUT_URDF_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)
mesh_dir = out_path.parent / f"{out_path.stem}_meshes"
mesh_dir.mkdir(parents=True, exist_ok=True)

usd_to_urdf = UsdToUrdf.init_from_file(temp_usd_path, log_level=logging.ERROR)
usd_to_urdf.save_to_file(
    urdf_output_path=str(out_path),
    visualize_collision_meshes=False,
    mesh_dir=str(mesh_dir),
    mesh_path_prefix="",
)

# Converter may emit inf for unlimited effort; replace for valid URDF parsers
if out_path.exists():
    text = out_path.read_text()
    text = text.replace("inf", "0.")
    out_path.write_text(text)

print(f"URDF written: {OUTPUT_URDF_PATH}")
print(f"Meshes: {mesh_dir}")

if out_path.exists():
    content = out_path.read_text()
    if "arm_right_1_joint" in content:
        joints = re.findall(r'joint name="arm_right_\d+_joint"', content)
        print(f"arm_right_* joints: {len(joints)}")
    else:
        joints = re.findall(r'joint name="([^"]+)"', content)
        arm_joints = [j for j in joints if "arm" in j.lower()]
        print(f"Arm-related joints (sample): {arm_joints[:10]}")

simulation_app.close()
