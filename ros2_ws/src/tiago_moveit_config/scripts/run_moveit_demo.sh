#!/usr/bin/env bash
set -euo pipefail
ROS_DISTRO="${ROS_DISTRO:-humble}"
# shellcheck source=/dev/null
source "/opt/ros/${ROS_DISTRO}/setup.bash"
INSTALL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ ! -f "${INSTALL_ROOT}/setup.bash" ]]; then
  echo "Expected colcon install at ${INSTALL_ROOT} (run from install tree after colcon build)."
  exit 1
fi
# shellcheck source=/dev/null
source "${INSTALL_ROOT}/setup.bash"
exec ros2 launch tiago_moveit_config demo.launch.py "$@"
