#!/usr/bin/env bash
set -euo pipefail
_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ -f "${_REPO}/extra/local_paths.sh" ]]; then
  # shellcheck source=/dev/null
  source "${_REPO}/extra/local_paths.sh"
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  conda activate "${TIAGO_LEROBOT_CONDA_ENV:-yor}"
fi

_DEFAULT_DATA="${HOME}/isaacsim_data"
_DATA_ROOT="${TIAGO_ISAACSIM_DATA_ROOT:-$_DEFAULT_DATA}"
OUTPUT_DIR="${TIAGO_ACT_POLICY_OUTPUT_DIR:-${_DATA_ROOT}/tiago_act_policy}"
ARCHIVE_DIR="${TIAGO_ACT_POLICY_ARCHIVE_DIR:-${_DATA_ROOT}/tiago_act_policy_archives}"
_DATASET_ROOT="${TIAGO_LEROBOT_DATASET_DIR:-${_DATA_ROOT}/tiago_lerobot_dataset}"
_CFG="${_REPO}/extra/lerobot/configs/tiago_act.yaml"

if [[ -d "$OUTPUT_DIR" ]]; then
  mkdir -p "$ARCHIVE_DIR"
  TS=$(date +"%Y%m%d_%H%M%S")
  mv "$OUTPUT_DIR" "$ARCHIVE_DIR/tiago_act_policy_$TS"
fi

exec python -m lerobot.scripts.lerobot_train --config_path "$_CFG" \
  --dataset.root="$_DATASET_ROOT" \
  --policy.device=cuda \
  --wandb.enable=false \
  --output_dir="$OUTPUT_DIR" \
  "$@"
