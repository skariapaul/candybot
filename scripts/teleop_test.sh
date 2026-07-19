#!/bin/bash
# Keyboard teleop smoke test for the follower arm -- useful since only the
# follower (no leader arm yet) is connected to this dev machine. Arrow keys /
# WASD move the arm depending on lerobot's keyboard teleop mapping; press ESC
# to stop. Run scripts/calibrate_arm.sh first.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

PORT="${CANDYBOT_ROBOT_PORT:-/dev/so101_follower}"

lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port="$PORT" \
  --robot.id=candybot_follower \
  --robot.calibration_dir="$(pwd)/configs" \
  --teleop.type=keyboard
