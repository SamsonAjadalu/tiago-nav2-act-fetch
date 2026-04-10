#!/usr/bin/env bash
set -euo pipefail
_EXTRA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${_EXTRA}/local_paths.sh" ]]; then
  # shellcheck source=/dev/null
  source "${_EXTRA}/local_paths.sh"
fi
: "${ISAACSIM_ROOT:?Set ISAACSIM_ROOT in extra/local_paths.sh (see local_paths.example.sh)}"
exec "${ISAACSIM_ROOT}/python.sh" "$@"
