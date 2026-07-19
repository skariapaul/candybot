#!/bin/bash
# Runs lerobot-calibrate for the SO-101 follower, using the id/port convention
# from configs/candybot.yaml. Follow the on-screen prompts (move to mid-range,
# then sweep each joint through its full range of motion).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

PORT="${CANDYBOT_ROBOT_PORT:-/dev/so101_follower}"

lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port="$PORT" \
  --robot.id=candybot_follower \
  --robot.calibration_dir="$(pwd)/configs"

echo "Calibration saved to configs/candybot_follower.json"
