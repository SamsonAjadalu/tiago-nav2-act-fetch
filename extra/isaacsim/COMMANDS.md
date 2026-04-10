# Commands

Command reference for the workflows used in this project (old + new), with quick notes on what each command starts.

Paths below use **`TIAGO_REPO_ROOT`** (the git clone of this repository) and **`TIAGO_ISAACSIM_DATA_ROOT`** (where NPZ checkpoints, datasets, and policy exports live; defaults to `~/isaacsim_data` in Python helpers). Set them in the shell before running:

```bash
export TIAGO_REPO_ROOT=/path/to/tiago_hil
export TIAGO_ISAACSIM_DATA_ROOT="${TIAGO_ISAACSIM_DATA_ROOT:-${HOME}/isaacsim_data}"
```

Scene **USD** files are not tracked in git (see `.gitignore`); place them under `"${TIAGO_REPO_ROOT}/usd/"` or set `TIAGO_USD_DIR` / per-scene env vars documented in `extra/isaacsim/scripts/tiago_repo_paths.py`.

---

## 1) Environment and policy setup

Set ACT checkpoint directory (if not already set in local config):

```bash
export TIAGO_ACT_POLICY_DIR="${TIAGO_ISAACSIM_DATA_ROOT}/tiago_act_policy/checkpoints/last/pretrained_model"
```

Install missing package for external ROS2 video recorder (only if needed):

```bash
python3 -m pip install --user imageio
```

---

## 2) ACT inference (original scene)

Start original ACT inference:

```bash
bash "${TIAGO_REPO_ROOT}/extra/isaacsim/run_inference.sh"
```

Start original ACT safe inference:

```bash
bash "${TIAGO_REPO_ROOT}/extra/isaacsim/run_inference_safe.sh"
```

---

## 3) ACT safe inference (house_with_can scene)

Start house-specific ACT safe inference:

```bash
bash "${TIAGO_REPO_ROOT}/extra/isaacsim/run_inference_safe_house_with_can.sh"
```

What this starts:

- loads `house_with_can.usd`,
- resolves can target in house scene,
- spawns TIAGo fallback when robot is not inside the house USD.

---

## 4) Residual PPO training / play

Set **`ISAACLAB_ROOT`** if Isaac Lab is not at `~/IsaacLab`. Train residual grasp-pose PPO (example):

```bash
bash "${TIAGO_REPO_ROOT}/extra/isaacsim/train_residual_grasp_pose_c2.sh" --headless --max_iterations 2000 --num_envs 16
```

Play a trained checkpoint:

```bash
bash "${TIAGO_REPO_ROOT}/extra/isaacsim/play_residual_grasp_pose_c2.sh" "<checkpoint_path>"
```

ACT-only style eval override example (from protocol/chats):

```bash
bash "${TIAGO_REPO_ROOT}/extra/isaacsim/train_residual_grasp_pose_c2.sh" --headless --max_iterations 200 env.action_scale=0.0
```

---

## 5) Data recording (old/original recorder)

Run original NPZ teleop recorder:

```bash
bash "${TIAGO_REPO_ROOT}/extra/isaacsim/run_python.sh" "${TIAGO_REPO_ROOT}/extra/isaacsim/scripts/tiago_vr_teleop_record_for_lerobot_npz.py" --enable omni.kit.xr.profile.vr --enable isaacsim.xr.openxr
```

---

## 6) Data recording (house scene, separate dataset root)

Use two terminals with the same `TIAGO_LEROBOT_NPZ_DIR`.

### Terminal A: external ROS2 video-to-MP4 recorder

```bash
export TIAGO_LEROBOT_NPZ_DIR="${TIAGO_ISAACSIM_DATA_ROOT}/tiago_lerobot_npz_house"
python3 "${TIAGO_REPO_ROOT}/extra/isaacsim/scripts/tiago_ros2_gemini2_video_marker_to_mp4.py" \
  --marker "${TIAGO_LEROBOT_NPZ_DIR}/video_episode_marker.json" \
  --out-dir "${TIAGO_LEROBOT_NPZ_DIR}/videos" \
  --topic /gemini2/color/image_raw \
  --fps 30
```

### Terminal B: house teleop NPZ recorder

```bash
export TIAGO_LEROBOT_NPZ_DIR="${TIAGO_ISAACSIM_DATA_ROOT}/tiago_lerobot_npz_house"
bash "${TIAGO_REPO_ROOT}/extra/isaacsim/run_record_lerobot_npz_house_with_can.sh"
```

Outputs:

- `episode_XXXX.npz`
- `video_episode_marker.json`
- `videos/episode_XXXX.mp4`

---

## 7) Dataset root switching (avoid accidental mixing)

Warehouse/old root:

```bash
export TIAGO_LEROBOT_NPZ_DIR="${TIAGO_ISAACSIM_DATA_ROOT}/tiago_lerobot_npz"
```

House root:

```bash
export TIAGO_LEROBOT_NPZ_DIR="${TIAGO_ISAACSIM_DATA_ROOT}/tiago_lerobot_npz_house"
```

---

## 8) Quick syntax/sanity checks

Shell launchers:

```bash
bash -n "${TIAGO_REPO_ROOT}/extra/isaacsim/run_inference_safe_house_with_can.sh"
bash -n "${TIAGO_REPO_ROOT}/extra/isaacsim/run_record_lerobot_npz_house_with_can.sh"
```

Python wrappers:

```bash
python3 -m py_compile "${TIAGO_REPO_ROOT}/extra/isaacsim/scripts/tiago_inference_act_safe_house_with_can.py"
python3 -m py_compile "${TIAGO_REPO_ROOT}/extra/isaacsim/scripts/tiago_vr_teleop_record_for_lerobot_npz_house_with_can.py"
```

#Nav2 tiago

ros2 launch tiago_nav bringup.launch.py namespace:=tiago \
  initial_pose_x:=-0.84791 initial_pose_y:=-0.79706 initial_pose_yaw:=3.141592653589793
