# ACT vs PPO Evaluation Protocol (Deadline Mode)

Use this protocol for every comparison so results are defensible.

Set `TIAGO_REPO_ROOT` to this repository clone and `ISAACLAB_ROOT` to your Isaac Lab install (default `~/IsaacLab`).

## 1) Fixed setup

- Task: `Template-Tiago-Hil-Residual-GraspPose-C2-Direct-v0`
- Env count: `--num_envs 16` for training, `--num_envs 1` for visual play checks
- Horizon: keep the same `episode_length_s` across compared runs
- Evaluation unit: TensorBoard `Episode/*` metrics from the same run length

## 2) Metrics to compare

Primary:

- `Episode/success_rate` (higher is better)
- `Episode/failure_rate` (lower is better)

Secondary:

- `Episode/mean_d_pos` (lower is better)
- `Episode/mean_span_dist` (lower is better)
- `Episode/mean_centered` (higher is better)
- `episode_lengths/iter` (earlier success should reduce timeout dominance)

Do not select checkpoints by `rewards/iter` only.

## 3) Runs to execute

### A. ACT-only baseline (residual disabled in effect)

Run PPO with zero residual scale so control is ACT-only:

```bash
bash "${TIAGO_REPO_ROOT}/extra/isaacsim/train_residual_grasp_pose_c2.sh" \
  --headless --max_iterations 120 --num_envs 16 \
  env.action_scale=0.0
```

### B. PPO baseline (current full config)

```bash
bash "${TIAGO_REPO_ROOT}/extra/isaacsim/train_residual_grasp_pose_c2.sh" \
  --headless --max_iterations 120 --num_envs 16
```

### C. Residual ladder short runs

1) Gripper-only residual:

```bash
bash "${TIAGO_REPO_ROOT}/extra/isaacsim/train_residual_grasp_pose_c2.sh" \
  --headless --max_iterations 120 --num_envs 16 \
  env.narrow_residual_ctrl_indices=[7,8]
```

2) Wrist-only residual:

```bash
bash "${TIAGO_REPO_ROOT}/extra/isaacsim/train_residual_grasp_pose_c2.sh" \
  --headless --max_iterations 120 --num_envs 16 \
  env.narrow_residual_ctrl_indices=[4,5,6]
```

3) Full configured residual:

```bash
bash "${TIAGO_REPO_ROOT}/extra/isaacsim/train_residual_grasp_pose_c2.sh" \
  --headless --max_iterations 120 --num_envs 16 \
  env.narrow_residual_ctrl_indices=[4,5,6,7,8]
```

## 4) Summarize each run

Use:

```bash
"${ISAACLAB_ROOT}/_isaac_sim/python.sh" \
  "${TIAGO_REPO_ROOT}/extra/isaacsim/scripts/summarize_grasppose_run.py" \
  --run_dir "${ISAACLAB_ROOT}/logs/rl_games/tiago_hil_residual_grasp_pose_c2/<run_dir>"
```

Collect summaries into the changelog/report as the final comparison table.
