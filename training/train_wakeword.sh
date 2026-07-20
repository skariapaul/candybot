#!/bin/bash
# Syncs the wake-word training pipeline to the remote MI300X host -- the
# more reliable path if this laptop's gfx902 iGPU turns out too slow for the
# actual augment+train steps (device detection working, confirmed via
# train_wakeword_local.sh, doesn't guarantee good throughput on this old
# iGPU for the heavy compute steps). MI300X is gfx942, officially
# ROCm-supported, no HSA_OVERRIDE_GFX_VERSION workaround needed there.
#
# Like training/sync_to_remote.sh, this only syncs code -- it prints the
# remaining setup/run steps for you to run over SSH, matching how ACT/
# smolVLA training is already handled in this project.
#
# Usage: ./training/train_wakeword.sh
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] && set -a && source .env && set +a

: "${REMOTE_HOST:?Set REMOTE_HOST in .env}"
: "${REMOTE_USER:?Set REMOTE_USER in .env}"
SSH_KEY="${REMOTE_SSH_KEY:-~/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_CANDYBOT_DIR:-~/candybot}"

echo "Syncing training/wakeword/ to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/training/wakeword/ ..."
rsync -avz -e "ssh -i $SSH_KEY" \
  --exclude ".venv-wakeword" --exclude "data" --exclude "output" --exclude "export" \
  wakeword/ "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/training/wakeword/"

cat <<EOF

Synced. Now on the remote host:

  ssh -i $SSH_KEY ${REMOTE_USER}@${REMOTE_HOST}
  cd ${REMOTE_DIR}/training/wakeword/trainer

  # Needs Python 3.11 specifically (piper-phonemize has no Linux 3.12 wheel).
  # If the remote host is also 3.12-only, grab a portable build the same way
  # this laptop did (see training/wakeword/README.md):
  #   curl -sL -o /tmp/py311.tar.gz \\
  #     https://github.com/astral-sh/python-build-standalone/releases/latest/download/cpython-3.11.15+20260718-x86_64-unknown-linux-gnu-install_only.tar.gz
  #   tar -xzf /tmp/py311.tar.gz -C ~/.local/

  ~/.local/python/bin/python3.11 -m venv .venv-wakeword   # or plain python3.11 if already available
  source .venv-wakeword/bin/activate
  CC=gcc CXX=g++ pip install -r requirements.txt          # CC/CXX needed if webrtcvad's build defaults to a missing clang
  pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/rocm6.3

  python train_wakeword.py --config configs/hey_zen.yaml

Then pull the trained model back:

  scp -i $SSH_KEY "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/training/wakeword/trainer/export/hey_zen.onnx*" \\
    training/wakeword/trainer/export/
EOF
