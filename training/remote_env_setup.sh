#!/bin/bash
# Run ON the remote MI300X host. Unlike scripts/setup_env.sh (this dev
# laptop's gfx902 workaround), MI300X (gfx942) is officially ROCm-supported --
# no HSA_OVERRIDE_GFX_VERSION needed here.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV_DIR=".venv-train"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

echo "Installing PyTorch (ROCm 6.3 wheel)..."
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/rocm6.3

echo "Installing lerobot==0.4.1..."
pip install "lerobot==0.4.1"

pip install "huggingface_hub[cli]"

python -c "import torch; print('CUDA/ROCm available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

echo
echo "Done. Activate with: source $VENV_DIR/bin/activate"
echo "Then: huggingface-cli login   (needs HF_TOKEN from .env)"
