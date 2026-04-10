# TIAGo Isaac Sim / LeRobot (demo slice)

This branch is a **minimal tree** for Isaac Sim ACT inference, NPZ teleop recording, and LeRobot dataset conversion and training. It is intended for external review and handoff; the full upstream project may include additional packages (Isaac Lab extension, ROS 2 workspaces, and assets not listed here).

## License

This software is licensed under the **GNU Affero General Public License v3.0**. See [LICENSE](LICENSE).

## Contents

| Path | Role |
|------|------|
| [config/](config/) | Lula robot description YAML (`OBOT_DESCRIPTION_YAML.yaml`) |
| [extra/isaacsim/](extra/isaacsim/) | Bash launchers, Python entrypoints, and [extra/isaacsim/scripts/tiago_repo_paths.py](extra/isaacsim/scripts/tiago_repo_paths.py) for repository-relative defaults |
| [extra/lerobot/](extra/lerobot/) | NPZ → LeRobot conversion scripts and training config |
| [tiago_urdf/](tiago_urdf/) | URDF and meshes referenced by Lula |

## Prerequisites

- **NVIDIA Isaac Sim** (version compatible with the APIs used in `extra/isaacsim/scripts/`)
- **LeRobot** and **PyTorch** available to the same Python used for ACT scripts (often Isaac Sim’s `python.sh`; optional setup notes live under `extra/isaacsim/`)
- **Scene USD files** are not committed (see `.gitignore`). Place assets under `<repository>/usd/` or set `TIAGO_USD_DIR`, `TIAGO_HOUSE_USD_PATH`, and related variables described in `tiago_repo_paths.py`.

Optional: add `extra/local_paths.sh` defining `ISAACSIM_ROOT`; several launchers source it when the file exists.

## Environment variables (quick reference)

| Variable | Role |
|----------|------|
| `TIAGO_REPO_ROOT` | Root of this repository (scripts infer it when unset) |
| `TIAGO_ISAACSIM_DATA_ROOT` | NPZ output, datasets, and policy checkpoints (default in helpers: `~/isaacsim_data`) |
| `TIAGO_ACT_POLICY_DIR` | Directory containing LeRobot `config.json` and `model.safetensors` |
| `ISAACSIM_ROOT` | Isaac Sim installation root (used by launchers with `python.sh`) |

Further detail: [extra/isaacsim/COMMANDS.md](extra/isaacsim/COMMANDS.md).

## Branch relationship

Branch **`share/tiago-demo`** is an orphan history containing only the paths above plus this README. Development continues on **`tiago-hil-direct-v0`** (and related branches) in the full repository.
