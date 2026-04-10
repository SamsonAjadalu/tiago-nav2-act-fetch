#!/usr/bin/env bash
set -euo pipefail
# Frozen ACT base + residual PPO. Cameras enabled.
#
# Prereq (once): bash "${TIAGO_REPO_ROOT}/extra/isaacsim/setup_lerobot_for_isaaclab.sh"
#
# export TIAGO_ACT_POLICY_DIR=/path/to/pretrained_model   (config + model.safetensors)
# Editable tiago_hil sync: TIAGO_SKIP_PIP_SYNC=1 to skip.

_ISA_SIM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${TIAGO_ACT_POLICY_DIR:?Set TIAGO_ACT_POLICY_DIR to the LeRobot pretrained_model directory}"

if [[ "${TIAGO_ACT_POLICY_DIR}" == *"..."* ]] || [[ ! -d "${TIAGO_ACT_POLICY_DIR}" ]]; then
  echo "ERROR: TIAGO_ACT_POLICY_DIR must be a real directory (not a placeholder like .../pretrained_model)."
  echo "  Got: ${TIAGO_ACT_POLICY_DIR}"
  echo "  Example: export TIAGO_ACT_POLICY_DIR=/media/you/isaacsim/tiago_act_policy/checkpoints/last/pretrained_model"
  exit 1
fi
if [[ ! -f "${TIAGO_ACT_POLICY_DIR}/config.json" ]] || [[ ! -f "${TIAGO_ACT_POLICY_DIR}/model.safetensors" ]]; then
  echo "ERROR: ${TIAGO_ACT_POLICY_DIR} must contain config.json and model.safetensors (LeRobot export)."
  exit 1
fi

ISAACLAB_ROOT="${ISAACLAB_ROOT:-${HOME}/IsaacLab}"
cd "${ISAACLAB_ROOT}"
# shellcheck disable=SC1091
source "${_ISA_SIM_DIR}/include_pip_sync_tiago_pkg.sh"

exec ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task Template-Tiago-Hil-Residual-C2-Direct-v0 \
  --enable_cameras \
  --num_envs 4 \
  "$@"
