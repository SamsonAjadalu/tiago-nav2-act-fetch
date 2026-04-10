# Long-Horizon Mobile Manipulation in Isaac Sim

This repository implements a long-horizon mobile manipulation workflow for the PAL Robotics TIAGo in NVIDIA Isaac Sim. ROS 2 Nav2 handles room-scale navigation, while LeRobot ACT provides visuomotor control for local approach and grasping.

## License

This software is licensed under the **GNU Affero General Public License v3.0**. See [LICENSE](LICENSE).

## Contents

| Path | Role |
|------|------|
| [config/](config/) | Lula robot description YAML (`OBOT_DESCRIPTION_YAML.yaml`) |
| [extra/isaacsim/](extra/isaacsim/) | Bash launchers, Python entrypoints, and [extra/isaacsim/scripts/tiago_repo_paths.py](extra/isaacsim/scripts/tiago_repo_paths.py) for repository-relative defaults |
| [extra/lerobot/](extra/lerobot/) | NPZ → LeRobot conversion scripts and training config |
| [tiago_urdf/](tiago_urdf/) | URDF and meshes referenced by Lula |
| [ros2_ws/](ros2_ws/) | Nav2 bringup, maps, MoveIt2 / RViz helpers (build with `colcon`; see [ros2_ws/README.md](ros2_ws/README.md)) |

## Prerequisites

- **NVIDIA Isaac Sim** (version compatible with the APIs used in `extra/isaacsim/scripts/`)
- **LeRobot** and **PyTorch** available to the same Python used for ACT scripts (often Isaac Sim’s `python.sh`; optional setup notes live under `extra/isaacsim/`)
- **Scene assets:** Place scene files under `<repository>/usd/` or configure `TIAGO_USD_DIR`, `TIAGO_HOUSE_USD_PATH`, and the related variables described in `tiago_repo_paths.py`.
- **ROS 2 Humble** and **Nav2** stack (`nav2_bringup` and dependencies from `ros2_ws/src/*/package.xml`). Build and run instructions: [ros2_ws/README.md](ros2_ws/README.md).

Optional: add `extra/local_paths.sh` defining `ISAACSIM_ROOT`; several launchers source it when the file exists.

## Environment variables (quick reference)

| Variable | Role |
|----------|------|
| `TIAGO_REPO_ROOT` | Root of this repository (scripts infer it when unset) |
| `TIAGO_ISAACSIM_DATA_ROOT` | NPZ output, datasets, and policy checkpoints (default in helpers: `~/isaacsim_data`) |
| `TIAGO_ACT_POLICY_DIR` | Directory containing LeRobot `config.json` and `model.safetensors` |
| `ISAACSIM_ROOT` | Isaac Sim installation root (used by launchers with `python.sh`) |

Further detail: [extra/isaacsim/COMMANDS.md](extra/isaacsim/COMMANDS.md). Nav2 / ROS 2: [ros2_ws/README.md](ros2_ws/README.md).
