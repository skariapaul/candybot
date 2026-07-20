#!/bin/bash
# Pulls a trained checkpoint (ACT per-bin, or the unified smolVLA policy)
# from the Hugging Face Hub after training/train_act.sh or
# training/train_smolvla.sh has pushed it, and prints the local path to set
# in configs/candybot.yaml.
#
# Usage: ./scripts/pull_checkpoint.sh <chocolate|candy|smolvla>
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
[ -f .env ] && set -a && source .env && set +a

if [ $# -ne 1 ] || [[ "$1" != "chocolate" && "$1" != "candy" && "$1" != "smolvla" ]]; then
  echo "Usage: $0 <chocolate|candy|smolvla>"
  exit 1
fi
TARGET="$1"

if [ -z "${HF_USER:-}" ]; then
  echo "HF_USER not set (see .env.example)."
  exit 1
fi

if [ "$TARGET" = "smolvla" ]; then
  REPO_ID="${HF_USER}/candybot_smolvla"
else
  REPO_ID="${HF_USER}/candybot_act_${TARGET}"
fi
DEST="outputs/checkpoints/${TARGET}"
mkdir -p "$DEST"

huggingface-cli download "$REPO_ID" --local-dir "$DEST"
echo
echo "Downloaded to $DEST"
if [ "$TARGET" = "smolvla" ]; then
  echo "Set configs/candybot.yaml: robot.smolvla.checkpoint: $(pwd)/${DEST}"
else
  echo "Set configs/candybot.yaml: robot.bins.${TARGET}.policy_checkpoint: $(pwd)/${DEST}"
fi
