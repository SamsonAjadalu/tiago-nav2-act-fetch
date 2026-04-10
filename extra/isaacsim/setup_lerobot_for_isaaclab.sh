#!/usr/bin/env bash
# Install LeRobot into Isaac Lab's Python while keeping numpy<2 for Isaac.
# Run:  bash extra/isaacsim/setup_lerobot_for_isaaclab.sh
# Env:  ISAACLAB_ROOT (optional)

set -euo pipefail
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${HOME}/IsaacLab}"
cd "${ISAACLAB_ROOT}"

PY=(./isaaclab.sh -p)

echo "[setup] Isaac Lab root: ${ISAACLAB_ROOT}"
echo "[setup] Pin numpy<2 first, then lerobot, then force numpy<2 again…"
"${PY[@]}" -m pip install -U "numpy>=1.26,<2"
"${PY[@]}" -m pip install -U lerobot
"${PY[@]}" -m pip install -U --force-reinstall "numpy>=1.26,<2"

if "${PY[@]}" -m pip show rerun-sdk &>/dev/null; then
  echo "[setup] Removing rerun-sdk (wants numpy>=2; not needed for frozen ACT)…"
  "${PY[@]}" -m pip uninstall -y rerun-sdk || true
fi

echo "[setup] Verify…"
"${PY[@]}" -c 'import numpy as np; import lerobot; from lerobot.policies.act.modeling_act import ACTPolicy; assert int(np.__version__.split(".")[0]) < 2; print("[setup] OK numpy", np.__version__)'

echo "[setup] Done. Pip may still print resolver warnings (packaging / usd-core); if train runs, ignore."
