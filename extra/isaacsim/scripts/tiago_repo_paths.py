"""Repository-relative defaults for URDF, YAML, and USD paths.

Override with ``TIAGO_REPO_ROOT``, ``TIAGO_URDF_PATH``, ``TIAGO_YAML_PATH``,
``TIAGO_USD_DIR``, or per-asset env vars where noted in call sites.
"""

from __future__ import annotations

import os
from pathlib import Path


def tiago_repo_root() -> Path:
    root = os.environ.get("TIAGO_REPO_ROOT", "").strip()
    if root:
        return Path(root).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def _path_from_env(name: str, default: Path) -> str:
    v = os.environ.get(name, "").strip()
    return str(Path(v).expanduser().resolve()) if v else str(default)


def default_urdf_path() -> str:
    return _path_from_env(
        "TIAGO_URDF_PATH",
        tiago_repo_root() / "tiago_urdf" / "tiago_dual_simple.urdf",
    )


def default_yaml_path() -> str:
    return _path_from_env(
        "TIAGO_YAML_PATH",
        tiago_repo_root() / "config" / "OBOT_DESCRIPTION_YAML.yaml",
    )


def usd_dir() -> Path:
    d = os.environ.get("TIAGO_USD_DIR", "").strip()
    return Path(d).expanduser().resolve() if d else tiago_repo_root() / "usd"


def usd_file(filename: str) -> str:
    override = os.environ.get(f"TIAGO_USD_{filename.replace('.', '_').upper()}", "").strip()
    if override:
        return str(Path(override).expanduser().resolve())
    return str(usd_dir() / filename)


def default_tiago_usd_path() -> str:
    return os.environ.get("TIAGO_USD_PATH", "").strip() or usd_file("tiago_dual_functional.usd")


def default_table_objects_usd_path() -> str:
    return (
        os.environ.get("TABLE_OBJECTS_USD_PATH", "").strip()
        or usd_file("table_with_objects.usd")
    )


def default_isaacsim_data_root() -> Path:
    v = os.environ.get("TIAGO_ISAACSIM_DATA_ROOT", "").strip()
    if v:
        return Path(v).expanduser().resolve()
    return Path.home() / "isaacsim_data"


def default_act_policy_dir() -> str:
    explicit = os.environ.get("TIAGO_ACT_POLICY_DIR", "").strip()
    if explicit:
        return explicit
    base = os.environ.get("TIAGO_ISAACSIM_DATA_ROOT", "").strip()
    if base:
        return str(
            Path(base).expanduser()
            / "tiago_act_policy"
            / "checkpoints"
            / "last"
            / "pretrained_model"
        )
    return str(
        default_isaacsim_data_root()
        / "tiago_act_policy"
        / "checkpoints"
        / "last"
        / "pretrained_model"
    )


def default_lerobot_npz_parent() -> str:
    v = os.environ.get("TIAGO_LEROBOT_NPZ_PARENT", "").strip()
    if v:
        return v
    base = os.environ.get("TIAGO_ISAACSIM_DATA_ROOT", "").strip()
    if base:
        return str(Path(base).expanduser())
    return str(default_isaacsim_data_root())


def house_usd_v2_path() -> str:
    v = os.environ.get("TIAGO_HOUSE_USD_PATH", "").strip()
    return v if v else usd_file("house_with_can2.usd")


def house_usd_v1_path() -> str:
    v = os.environ.get("TIAGO_HOUSE_SCENE_USD_PATH", "").strip()
    return v if v else usd_file("house_with_can.usd")
