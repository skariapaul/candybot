#!/bin/bash
# Pulls a trained ACT checkpoint from the Hugging Face Hub after
# training/train_act.sh has pushed it, and prints the local path to set
# as configs/candybot.yaml's robot.bins.<bin>.policy_checkpoint.
#
# Usage: ./scripts/pull_checkpoint.sh <chocolate|candy>
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
[ -f .env ] && set -a && source .env && set +a

if [ $# -ne 1 ] || [[ "$1" != "chocolate" && "$1" != "candy" ]]; then
  echo "Usage: $0 <chocolate|candy>"
  exit 1
fi
BIN_NAME="$1"

if [ -z "${HF_USER:-}" ]; then
  echo "HF_USER not set (see .env.example)."
  exit 1
fi

REPO_ID="${HF_USER}/candybot_act_${BIN_NAME}"
DEST="outputs/checkpoints/${BIN_NAME}"
mkdir -p "$DEST"

huggingface-cli download "$REPO_ID" --local-dir "$DEST"
echo
echo "Downloaded to $DEST"
echo "Set configs/candybot.yaml: robot.bins.${BIN_NAME}.policy_checkpoint: $(pwd)/${DEST}"
