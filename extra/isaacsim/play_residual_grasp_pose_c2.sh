#!/usr/bin/env bash
set -euo pipefail
# Play a trained RL-Games PPO checkpoint for GraspPose C2 (same task / ACT dir as train).
#
# Script location: tiago_hil/extra/isaacsim/play_residual_grasp_pose_c2.sh (not IsaacLab root).
#
# ACT policy path: export TIAGO_ACT_POLICY_DIR=... OR put one absolute path line in
#   extra/isaacsim/.tiago_act_policy_dir (overrides a bad placeholder in your shell).
#
# Usage:
#   ./play_residual_grasp_pose_c2.sh /absolute/path/to/last_....pth
#   ./play_residual_grasp_pose_c2.sh --latest
# Extra args go to play.py, e.g.: ./play_residual_grasp_pose_c2.sh --latest --video --video_length 400
#
# Runs until you close the Isaac Sim window. First GPU load can take several minutes.

_ISA_SIM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${_ISA_SIM_DIR}/include_tiago_act_policy_dir.sh" || exit 1

ISAACLAB_ROOT="${ISAACLAB_ROOT:-${HOME}/IsaacLab}"
LOG_GLOB="${ISAACLAB_ROOT}/logs/rl_games/tiago_hil_residual_grasp_pose_c2/*/nn/last_*.pth"

if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
  echo "Usage: $0 CHECKPOINT.pth   OR   $0 --latest   [extra play.py args]"
  exit 0
fi

if [[ "${1:-}" == "--latest" ]]; then
  shift
  mapfile -t _ckpts < <(ls -t ${LOG_GLOB} 2>/dev/null || true)
  if [[ ${#_ckpts[@]} -eq 0 ]]; then
    echo "ERROR: No checkpoints matched: ${LOG_GLOB}"
    echo "Train first, or pass the full path to a .pth file."
    exit 1
  fi
  CHECKPOINT="${_ckpts[0]}"
  echo "[INFO] Using latest checkpoint: ${CHECKPOINT}"
else
  CHECKPOINT="${1:-}"
  if [[ -z "${CHECKPOINT}" ]]; then
    echo "ERROR: Missing checkpoint path."
    echo "  Example: $0 ${ISAACLAB_ROOT}/logs/rl_games/tiago_hil_residual_grasp_pose_c2/<date>/nn/last_....pth"
    echo "  Or: $0 --latest"
    exit 1
  fi
  shift
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
  exit 1
fi

cd "${ISAACLAB_ROOT}"
# shellcheck disable=SC1091
source "${_ISA_SIM_DIR}/include_pip_sync_tiago_pkg.sh"

exec ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task Template-Tiago-Hil-Residual-GraspPose-C2-Direct-v0 --enable_cameras --num_envs 1 --checkpoint "${CHECKPOINT}" --real-time "$@"
