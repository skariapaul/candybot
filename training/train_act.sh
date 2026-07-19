#!/bin/bash
# Runs on the remote MI300X host (after remote_env_setup.sh). Fine-tunes an
# ACT policy on a candybot dataset and pushes the checkpoint to the HF Hub,
# where scripts/pull_checkpoint.sh retrieves it back on the edge laptop.
#
# Usage: ./train_act.sh <chocolate|candy> [steps]
set -euo pipefail

if [ $# -lt 1 ] || [[ "$1" != "chocolate" && "$1" != "candy" ]]; then
  echo "Usage: $0 <chocolate|candy> [steps]"
  exit 1
fi
BIN_NAME="$1"
STEPS="${2:-100000}"

if [ -z "${HF_USER:-}" ]; then
  echo "HF_USER not set (see .env.example)."
  exit 1
fi

lerobot-train \
  --dataset.repo_id="${HF_USER}/candybot_${BIN_NAME}" \
  --policy.type=act \
  --policy.push_to_hub=true \
  --policy.repo_id="${HF_USER}/candybot_act_${BIN_NAME}" \
  --steps="$STEPS" \
  --output_dir="outputs/train/act_${BIN_NAME}"

echo "Checkpoint: outputs/train/act_${BIN_NAME}/checkpoints/last/pretrained_model/"
echo "Pushed to: https://huggingface.co/${HF_USER}/candybot_act_${BIN_NAME}"
