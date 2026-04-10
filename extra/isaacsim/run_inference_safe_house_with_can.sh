#!/usr/bin/env bash
set -euo pipefail
_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "${_REPO}/extra/local_paths.sh" ]]; then
  # shellcheck source=/dev/null
  source "${_REPO}/extra/local_paths.sh"
fi
: "${ISAACSIM_ROOT:?Set ISAACSIM_ROOT in extra/local_paths.sh (see local_paths.example.sh)}"
export TIAGO_ISAACSIM_DATA_ROOT="${TIAGO_ISAACSIM_DATA_ROOT:-${HOME}/isaacsim_data}"
export TIAGO_ACT_POLICY_DIR="${TIAGO_ACT_POLICY_DIR:-${TIAGO_ISAACSIM_DATA_ROOT}/tiago_act_policy/checkpoints/160000/pretrained_model}"
cd "${ISAACSIM_ROOT}"
exec "${ISAACSIM_ROOT}/python.sh" "${_REPO}/extra/isaacsim/scripts/tiago_inference_act_safe_house_with_can.py" "$@"
