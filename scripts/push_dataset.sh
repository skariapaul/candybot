#!/bin/bash
# Pushes a locally-recorded dataset directory to the Hugging Face Hub.
# Only needed if scripts/record_dataset.sh was run with push_to_hub=false;
# by default that script pushes automatically as it records.
#
# Usage: ./scripts/push_dataset.sh <chocolate|candy|smolvla>
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
[ -f .env ] && set -a && source .env && set +a

if [ $# -ne 1 ] || [[ "$1" != "chocolate" && "$1" != "candy" && "$1" != "smolvla" ]]; then
  echo "Usage: $0 <chocolate|candy|smolvla>"
  exit 1
fi
TARGET="$1"

if [ -z "${HF_USER:-}" ] || [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_USER and HF_TOKEN must be set (see .env.example)."
  exit 1
fi

REPO_ID="${HF_USER}/candybot_${TARGET}"
LOCAL_DIR="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}/${REPO_ID}"

huggingface-cli upload "$REPO_ID" "$LOCAL_DIR" --repo-type dataset --token "$HF_TOKEN"
echo "Pushed $LOCAL_DIR -> https://huggingface.co/datasets/$REPO_ID"
