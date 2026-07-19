#!/bin/bash
# Syncs the repo's training/ directory (and configs) to the remote MI300X
# host, gated on REMOTE_HOST/REMOTE_USER/REMOTE_SSH_KEY from .env. Datasets
# and checkpoints themselves move via the Hugging Face Hub (see
# scripts/push_dataset.sh / pull_checkpoint.sh) -- this is just for the code.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && source .env && set +a

: "${REMOTE_HOST:?Set REMOTE_HOST in .env}"
: "${REMOTE_USER:?Set REMOTE_USER in .env}"
SSH_KEY="${REMOTE_SSH_KEY:-~/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_CANDYBOT_DIR:-~/candybot}"

rsync -avz -e "ssh -i $SSH_KEY" \
  --exclude ".venv*" --exclude "outputs" --exclude ".git" \
  training/ pyproject.toml configs/candybot.yaml \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

echo "Synced to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
