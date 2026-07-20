#!/bin/bash
# Runs on the remote MI300X host (after remote_env_setup.sh). Fine-tunes
# smolVLA (450M params, starts from the pretrained lerobot/smolvla_base
# checkpoint rather than training from scratch -- verified against lerobot
# 0.4.1's own TrainPipelineConfig: --policy.path is the fine-tune-from-
# pretrained flag, same mechanism lerobot-eval uses to load a checkpoint)
# on the varied-instruction candybot dataset, and pushes the result to the
# HF Hub, where scripts/pull_checkpoint.sh retrieves it back on the edge
# laptop. Unlike train_act.sh, this is ONE unified policy/dataset, not
# one per bin -- see docs/SMOLVLA.md.
#
# Usage: ./train_smolvla.sh [steps]
set -euo pipefail

STEPS="${1:-20000}"  # smolVLA fine-tunes fast from the pretrained base -- far fewer steps than ACT's from-scratch 100k

if [ -z "${HF_USER:-}" ]; then
  echo "HF_USER not set (see .env.example)."
  exit 1
fi

lerobot-train \
  --dataset.repo_id="${HF_USER}/candybot_smolvla" \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=true \
  --policy.repo_id="${HF_USER}/candybot_smolvla" \
  --steps="$STEPS" \
  --output_dir="outputs/train/smolvla"

echo "Checkpoint: outputs/train/smolvla/checkpoints/last/pretrained_model/"
echo "Pushed to: https://huggingface.co/${HF_USER}/candybot_smolvla"
