#!/usr/bin/env bash
set -euo pipefail
_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "${_REPO}/extra/local_paths.sh" ]]; then
  # shellcheck source=/dev/null
  source "${_REPO}/extra/local_paths.sh"
fi
: "${ISAACSIM_ROOT:?Set ISAACSIM_ROOT in extra/local_paths.sh (see local_paths.example.sh)}"
cd "${ISAACSIM_ROOT}"
exec "${ISAACSIM_ROOT}/python.sh" "${_REPO}/extra/isaacsim/scripts/tiago_inference_act.py" "$@"
