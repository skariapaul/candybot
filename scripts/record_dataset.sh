#!/bin/bash
# Records a demonstration dataset for one bin via lerobot-record.
#
# NOTE: needs the SO-101 *leader* arm for real kinesthetic teleop -- only the
# follower is connected to this dev machine today. Until the leader arm
# arrives, this can still be run with `--teleop.type=keyboard` for a small
# proof dataset, but it won't be demo-quality training data.
#
# Usage: ./scripts/record_dataset.sh <chocolate|candy> [leader-port]
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
[ -f .env ] && set -a && source .env && set +a

if [ $# -lt 1 ] || [[ "$1" != "chocolate" && "$1" != "candy" ]]; then
  echo "Usage: $0 <chocolate|candy> [leader-port]"
  exit 1
fi
BIN_NAME="$1"
LEADER_PORT="${2:-}"
FOLLOWER_PORT="${CANDYBOT_ROBOT_PORT:-/dev/so101_follower}"

if [ -z "${HF_USER:-}" ]; then
  echo "HF_USER not set -- add it to .env (see .env.example)."
  exit 1
fi

TELEOP_ARGS=(--teleop.type=keyboard)
if [ -n "$LEADER_PORT" ]; then
  TELEOP_ARGS=(--teleop.type=so101_leader --teleop.port="$LEADER_PORT")
else
  echo "No leader port given -- falling back to --teleop.type=keyboard (proof dataset only, not demo-quality)."
fi

lerobot-record \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id=candybot_follower \
  --robot.calibration_dir="$(pwd)/configs" \
  "${TELEOP_ARGS[@]}" \
  --dataset.repo_id="${HF_USER}/candybot_${BIN_NAME}" \
  --dataset.single_task="Pick up the ${BIN_NAME} and hand it to the visitor." \
  --dataset.num_episodes=60 \
  --dataset.episode_time_s=20 \
  --dataset.push_to_hub=true
