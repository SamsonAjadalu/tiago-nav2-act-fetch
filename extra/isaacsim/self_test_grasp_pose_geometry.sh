#!/usr/bin/env bash
set -euo pipefail
# Pure torch self-test (no Kit / no tiago_hil package import chain).

_SELF_TEST_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TIAGO_HIL_ROOT="${TIAGO_HIL_ROOT:-$(cd "${_SELF_TEST_DIR}/../.." && pwd)}"
GRASP_PY="${TIAGO_HIL_ROOT}/extra/source/tiago_hil/tiago_hil/tasks/direct/tiago_hil/grasp_pose_geometry.py"
exec python3 "${GRASP_PY}"
