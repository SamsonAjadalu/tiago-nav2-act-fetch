#!/usr/bin/env bash
# No 'set -u' here: ROS setup.bash reads vars like AMENT_TRACE_SETUP_FILES that may be unset.
set -eo pipefail
#
# Prefer the short launcher (no ellipses): tiago_hil/run_tiago_moveit_py_isaac.sh
# Manual sources must be full paths, e.g.:
#   source /opt/ros/humble/setup.bash
#   source /media/robotswithai/data1/robots/tiago_hil/moveit2_humble_ws/install/setup.bash
#   source /media/robotswithai/data1/robots/tiago_hil/ros2_ws/install/setup.bash
#
# Isaac must already be publishing /tiago/joint_states and /tiago/clock (except --self-test).
ROOT="${TIAGO_HIL_ROOT:-/media/robotswithai/data1/robots/tiago_hil}"

source /opt/ros/humble/setup.bash
source "$ROOT/moveit2_humble_ws/install/setup.bash"
source "$ROOT/ros2_ws/install/setup.bash"

if [[ "${1:-}" == "--self-test" ]]; then
  shift
  exec ros2 run tiago_moveit_config tiago_moveit_py_grasp_near.py --self-test "$@"
fi

LAUNCH_PID=""
cleanup() {
  if [[ -n "${LAUNCH_PID}" ]] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    kill "${LAUNCH_PID}" 2>/dev/null || true
    wait "${LAUNCH_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ros2 launch tiago_moveit_config demo.launch.py \
  start_move_group:=false \
  use_rviz:=false \
  mirror_joint_states:=false \
  start_robot_state_publisher:=false \
  start_odom_static_tf:=false \
  use_sim_time:=true \
  relay_sim_clock:=true &
LAUNCH_PID=$!

wait_sec=90
for ((i = 0; i < wait_sec; i++)); do
  if ros2 topic list 2>/dev/null | grep -qx "/tiago/joint_states"; then
    break
  fi
  sleep 1
done
if ! ros2 topic list 2>/dev/null | grep -qx "/tiago/joint_states"; then
  echo "Timeout: /tiago/joint_states not seen. Is Isaac running?" >&2
  exit 1
fi

# Let /tiago/clock -> /clock relay and Isaac-stamped joint_states settle (avoids PSM "time 0" vs sim clock).
sleep 8

set +e
ros2 run tiago_moveit_config tiago_moveit_py_grasp_near.py --use-sim-time --isaac-tf "$@"
code=$?
set -e
exit "$code"
