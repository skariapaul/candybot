#!/bin/bash
# Runs the "Hey Zen" wake-word training pipeline locally, targeting this
# laptop's iGPU via ROCm -- confirmed working: torch.cuda.is_available()
# is True here, and the pipeline's own check-env step detects it as
# "AMD Radeon Graphics". Falls back to CPU automatically if the GPU path
# fails for any reason (same torch.cuda.is_available() check the rest of
# candybot uses).
#
# Needs Python 3.11 specifically -- piper-phonemize has no Linux wheel for
# 3.12 as of this writing. See training/wakeword/trainer/.venv-wakeword
# setup notes in training/wakeword/README.md if recreating this venv.
#
# Usage: ./training/train_wakeword_local.sh [--from STEP] [--step STEP] [--list-steps]
set -euo pipefail
cd "$(dirname "$0")/wakeword/trainer"

if [ ! -d .venv-wakeword ]; then
  echo ".venv-wakeword not found -- see training/wakeword/README.md to set it up first."
  exit 1
fi

source .venv-wakeword/bin/activate
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-9.0.0}"

python train_wakeword.py --config configs/hey_zen.yaml "$@"

echo
echo "If this completed the 'export' step, find the model at:"
echo "  training/wakeword/trainer/export/hey_zen.onnx"
echo "  training/wakeword/trainer/export/hey_zen.onnx.data"
echo "Copy both to candybot/models/wakeword/, then set configs/candybot.yaml's"
echo "voice.wake_word.model to the new model name."
