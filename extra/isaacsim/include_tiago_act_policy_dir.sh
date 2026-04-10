#!/usr/bin/env bash
# Sourced by train_residual_grasp_pose_c2.sh and play_residual_grasp_pose_c2.sh.
# Resolution order:
#   1) .tiago_act_policy_dir (first non-comment line) if present
#   2) TIAGO_ACT_POLICY_DIR from the environment if it validates
#   3) Known workspace default(s) on this machine (so a bad tutorial export is ignored)

_ACT_ISA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_LOCAL_ACT="${_ACT_ISA_DIR}/.tiago_act_policy_dir"

_tiago_act_valid() {
  [[ -n "${1:-}" ]] && [[ -d "${1}" ]] \
    && [[ -f "${1}/config.json" ]] && [[ -f "${1}/model.safetensors" ]]
}

if [[ -f "${_LOCAL_ACT}" ]]; then
  _from_file=""
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ "${line}" =~ ^[[:space:]]*# ]] && continue
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line}" ]] && continue
    _from_file="${line}"
    break
  done < "${_LOCAL_ACT}"
  if [[ -n "${_from_file}" ]]; then
    TIAGO_ACT_POLICY_DIR="${_from_file}"
    export TIAGO_ACT_POLICY_DIR
  fi
fi

if ! _tiago_act_valid "${TIAGO_ACT_POLICY_DIR:-}"; then
  _ISAAC_DATA="${TIAGO_ISAACSIM_DATA_ROOT:-${HOME}/isaacsim_data}"
  _TIAGO_DEFAULTS=(
    "${_ISAAC_DATA}/tiago_act_policy/checkpoints/last/pretrained_model"
    "${_ISAAC_DATA}/tiago_act_policy/checkpoints/160000/pretrained_model"
  )
  TIAGO_ACT_POLICY_DIR=""
  for _c in "${_TIAGO_DEFAULTS[@]}"; do
    if _tiago_act_valid "${_c}"; then
      TIAGO_ACT_POLICY_DIR="${_c}"
      export TIAGO_ACT_POLICY_DIR
      break
    fi
  done
fi

if ! _tiago_act_valid "${TIAGO_ACT_POLICY_DIR:-}"; then
  echo "ERROR: No valid LeRobot pretrained_model (need config.json + model.safetensors)."
  echo "  Put one absolute path line in: ${_LOCAL_ACT}"
  echo "  Or export TIAGO_ACT_POLICY_DIR to that directory."
  return 1 2>/dev/null || exit 1
fi

return 0 2>/dev/null || true
