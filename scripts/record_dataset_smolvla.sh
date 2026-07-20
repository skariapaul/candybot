#!/bin/bash
# Records demonstration episodes for the smolVLA dataset. Unlike
# record_dataset.sh's fixed per-bin task, run this multiple times with
# DIFFERENT instruction text (and physically vary the target cup's identity/
# position between runs) so the trained policy generalizes past one fixed
# scenario -- that variety is the whole point of a language-conditioned
# model. Each invocation appends more episodes to the same
# candybot_smolvla dataset (lerobot-record accumulates episodes under a
# repo_id across separate runs).
#
# NOTE: needs the SO-101 leader arm for real kinesthetic teleop -- falls back
# to keyboard teleop (proof-of-pipeline only, not demo-quality data) if no
# leader port is given, same as record_dataset.sh.
#
# Usage: ./scripts/record_dataset_smolvla.sh "<task description>" [leader-port] [num-episodes]
# Example: ./scripts/record_dataset_smolvla.sh "pick up the gold cup and hand it to the visitor" /dev/ttyACM1 15
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
[ -f .env ] && set -a && source .env && set +a

if [ $# -lt 1 ]; then
  echo 'Usage: ./scripts/record_dataset_smolvla.sh "<task description>" [leader-port] [num-episodes]'
  exit 1
fi
TASK="$1"
LEADER_PORT="${2:-}"
NUM_EPISODES="${3:-15}"
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
  --dataset.repo_id="${HF_USER}/candybot_smolvla" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.episode_time_s=20 \
  --dataset.push_to_hub=true
