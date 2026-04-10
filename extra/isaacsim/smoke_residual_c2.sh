#!/usr/bin/env bash
# One-epoch smoke: C2 env + cameras + ACT load (no long training).
# Requires: TIAGO_ACT_POLICY_DIR, Isaac Lab, tiago_hil installed in that Python.
#
# Example:
#   source /path/to/tiago_hil/extra/local_paths.sh
#   export TIAGO_ACT_POLICY_DIR="${TIAGO_ISAACSIM_DATA_ROOT:-${HOME}/isaacsim_data}/tiago_act_policy/checkpoints/last/pretrained_model"
#   bash /path/to/tiago_hil/extra/isaacsim/smoke_residual_c2.sh
#
# Pass-through: e.g.  bash smoke_residual_c2.sh --headless

set -euo pipefail
_ISA_SIM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${TIAGO_ACT_POLICY_DIR:?Set TIAGO_ACT_POLICY_DIR}"

if [[ "${TIAGO_ACT_POLICY_DIR}" == *"..."* ]] || [[ ! -d "${TIAGO_ACT_POLICY_DIR}" ]]; then
  echo "ERROR: TIAGO_ACT_POLICY_DIR must be a real pretrained_model directory, not a placeholder path."
  exit 1
fi
if [[ ! -f "${TIAGO_ACT_POLICY_DIR}/config.json" ]] || [[ ! -f "${TIAGO_ACT_POLICY_DIR}/model.safetensors" ]]; then
  echo "ERROR: Missing config.json or model.safetensors under ${TIAGO_ACT_POLICY_DIR}"
  exit 1
fi

ISAACLAB_ROOT="${ISAACLAB_ROOT:-${HOME}/IsaacLab}"
cd "${ISAACLAB_ROOT}"
# shellcheck disable=SC1091
source "${_ISA_SIM_DIR}/include_pip_sync_tiago_pkg.sh"

exec ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task Template-Tiago-Hil-Residual-C2-Direct-v0 \
  --enable_cameras \
  --num_envs 2 \
  --max_iterations 1 \
  "$@"
