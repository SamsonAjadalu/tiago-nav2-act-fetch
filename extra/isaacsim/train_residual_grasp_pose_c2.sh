#!/usr/bin/env bash
set -euo pipefail
# C2 + grasp-pose (52-D obs: +10 ACT ctrl targets). Default cfg uses narrow residual: wrist+gripper deltas
# only inside a near-can zone; override with task.env.cfg.narrow_residual_enable=false for full-arm residual.
# See CHANGELOG_TIAGO_HIL.md and
# canonical_tomato_can_grasp.py (tune targets from Isaac Grasp Editor).
#
# ACT policy path: export TIAGO_ACT_POLICY_DIR=... OR one line in extra/isaacsim/.tiago_act_policy_dir
# Code sync: editable tiago_hil is installed each run (TIAGO_SKIP_PIP_SYNC=1 to skip).

_ISA_SIM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${_ISA_SIM_DIR}/include_tiago_act_policy_dir.sh" || exit 1

ISAACLAB_ROOT="${ISAACLAB_ROOT:-${HOME}/IsaacLab}"
cd "${ISAACLAB_ROOT}"
# shellcheck disable=SC1091
source "${_ISA_SIM_DIR}/include_pip_sync_tiago_pkg.sh"

# Play: extra/isaacsim/play_residual_grasp_pose_c2.sh --latest
# Default env count is env_cfg.scene.num_envs (8) if --num_envs omitted; pass --num_envs 16 explicitly when needed.
# RTX may warn on complex table USD (Midtown meshes); performance mode reduces RTX load. Override: --rendering_mode balanced
exec ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task Template-Tiago-Hil-Residual-GraspPose-C2-Direct-v0 \
  --enable_cameras \
  --rendering_mode performance \
  "$@"
