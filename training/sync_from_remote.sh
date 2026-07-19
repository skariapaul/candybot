#!/bin/bash
# Fallback for pulling a checkpoint directly off the MI300X host's filesystem
# via rsync, if pushing it through the Hugging Face Hub (scripts/pull_checkpoint.sh)
# isn't desired (e.g. no reliable internet at the venue).
#
# Usage: ./sync_from_remote.sh <chocolate|candy>
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && source .env && set +a

: "${REMOTE_HOST:?Set REMOTE_HOST in .env}"
: "${REMOTE_USER:?Set REMOTE_USER in .env}"
SSH_KEY="${REMOTE_SSH_KEY:-~/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_CANDYBOT_DIR:-~/candybot}"

if [ $# -ne 1 ] || [[ "$1" != "chocolate" && "$1" != "candy" ]]; then
  echo "Usage: $0 <chocolate|candy>"
  exit 1
fi
BIN_NAME="$1"
DEST="outputs/checkpoints/${BIN_NAME}"
mkdir -p "$DEST"

rsync -avz -e "ssh -i $SSH_KEY" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/outputs/train/act_${BIN_NAME}/checkpoints/last/pretrained_model/" \
  "$DEST/"

echo "Synced to $DEST"
echo "Set configs/candybot.yaml: robot.bins.${BIN_NAME}.policy_checkpoint: $(pwd)/${DEST}"
