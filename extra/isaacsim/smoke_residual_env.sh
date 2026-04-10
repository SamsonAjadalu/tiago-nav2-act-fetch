#!/usr/bin/env bash
# Quick sanity check: residual env + table USD should run without PhysX explosion.
# Usage: bash extra/isaacsim/smoke_residual_env.sh
# Optional: NUM_ENVS=8 bash ...   (default 4 for fastest check)

set -euo pipefail
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${HOME}/IsaacLab}"
NUM_ENVS="${NUM_ENVS:-4}"

cd "${ISAACLAB_ROOT}"
exec ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task Template-Tiago-Hil-Residual-Direct-v0 \
  --headless \
  --num_envs "${NUM_ENVS}" \
  --max_iterations 1 \
  "$@"
