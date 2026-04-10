# shellcheck shell=bash
# Sync tiago_hil source next to this tree into Isaac Lab's Python (editable).
# Source from launchers only (not standalone): cwd must be ISAACLAB_ROOT; _ISA_SIM_DIR = this isaacsim/ dir.
# TIAGO_SKIP_PIP_SYNC=1 skips install.

if [[ -n "${TIAGO_SKIP_PIP_SYNC:-}" ]]; then
  return 0
fi

_TIAGO_SRC="$(cd "${_ISA_SIM_DIR}/../source/tiago_hil" && pwd)"
if [[ ! -f "${_TIAGO_SRC}/pyproject.toml" ]]; then
  echo "WARNING: tiago_hil package not found at ${_TIAGO_SRC}; skip pip sync." >&2
  return 0
fi

echo "[tiago_hil] pip install -e ${_TIAGO_SRC} (Isaac Lab python)"
./isaaclab.sh -p -m pip install -q -e "${_TIAGO_SRC}"
