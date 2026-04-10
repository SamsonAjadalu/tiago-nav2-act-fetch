#!/usr/bin/env bash
set -euo pipefail

_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "${_REPO}/extra/local_paths.sh" ]]; then
  # shellcheck source=/dev/null
  source "${_REPO}/extra/local_paths.sh"
fi

: "${ISAACSIM_ROOT:?Set ISAACSIM_ROOT in extra/local_paths.sh (see local_paths.example.sh)}"

_is_num_re='^-?[0-9]+([.][0-9]+)?$'

_usage() {
  echo "usage: $0 [X [Y [Z]]] [--house-x X] [--house-y Y] [--house-z Z] [isaac python args...]" >&2
}

# Positional: allow X Y Z in order (for quick debugging). If only one number is
# provided, keep backward compatibility: treat it as Y.
if [[ "${1:-}" =~ ${_is_num_re} ]]; then
  if [[ "${2:-}" =~ ${_is_num_re} || -z "${2:-}" ]]; then
    if [[ -z "${2:-}" ]]; then
      export TIAGO_HOUSE_LEGACY_TY="${1}"
      shift 1
    else
      export TIAGO_HOUSE_LEGACY_TX="${1}"
      export TIAGO_HOUSE_LEGACY_TY="${2}"
      if [[ "${3:-}" =~ ${_is_num_re} ]]; then
        export TIAGO_HOUSE_LEGACY_TZ="${3}"
        shift 3
      else
        shift 2
      fi
    fi
  fi
fi

# Flags: override any positional values.
while [[ $# -gt 0 ]]; do
  case "${1}" in
    --house-x)
      [[ -n "${2:-}" ]] || { _usage; exit 2; }
      export TIAGO_HOUSE_LEGACY_TX="${2}"
      shift 2
      ;;
    --house-y)
      [[ -n "${2:-}" ]] || { _usage; exit 2; }
      export TIAGO_HOUSE_LEGACY_TY="${2}"
      shift 2
      ;;
    --house-z)
      [[ -n "${2:-}" ]] || { _usage; exit 2; }
      export TIAGO_HOUSE_LEGACY_TZ="${2}"
      shift 2
      ;;
    --help|-h)
      _usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

cd "${ISAACSIM_ROOT}"
exec "${ISAACSIM_ROOT}/python.sh" \
  "${_REPO}/extra/isaacsim/scripts/tiago_vr_teleop_record_for_lerobot_npz_house_with_can.py" \
  --enable omni.kit.xr.profile.vr \
  --enable isaacsim.xr.openxr \
  "$@"

