#!/bin/bash
# Sets up a venv for candybot, targeting the iGPU (ROCm) by default on this dev machine.
# See docs/SETUP_DEV_MACHINE.md for the gfx902 caveat this script works around.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV_DIR=".venv"
GFX_OVERRIDE="${HSA_OVERRIDE_GFX_VERSION:-9.0.0}"   # gfx900, nearest ROCm-supported neighbor to this laptop's gfx902

# pip buffers large wheel downloads (the ROCm torch wheel is ~4.5GB) in TMPDIR before
# installing. Root ("/") is a small 17G partition that's already tight -- point TMPDIR
# at this repo's own partition (/home, hundreds of GB free) instead of the /tmp default.
export TMPDIR="$(pwd)/.pip-tmp"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating venv at $VENV_DIR (python $(python3 --version))..."
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

echo "Installing PyTorch (ROCm 6.3 wheel, targeting the iGPU)..."
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/rocm6.3

echo "Installing lerobot==0.4.1 [feetech] (SO-101 uses Feetech STS3215 servos)..."
pip install "lerobot[feetech]==0.4.1"

echo "Installing candybot (editable) with voice/dashboard/dev extras..."
pip install -e ".[voice,dashboard,dev]"

# Bake the gfx override into venv activation so it's always set when this env is active,
# without polluting the user's shell profile. Idempotent — skips if already appended.
if ! grep -q "HSA_OVERRIDE_GFX_VERSION" "$VENV_DIR/bin/activate"; then
  {
    echo ""
    echo "# candybot: target this laptop's gfx902 iGPU via the nearest ROCm-supported neighbor"
    echo "export HSA_OVERRIDE_GFX_VERSION=\"$GFX_OVERRIDE\""
  } >> "$VENV_DIR/bin/activate"
fi
export HSA_OVERRIDE_GFX_VERSION="$GFX_OVERRIDE"

echo
echo "Validating GPU access (subprocess + timeout guarded — see candybot/hardware_probe.py)..."
python -m candybot.hardware_probe

echo
echo "Done. Activate with: source $VENV_DIR/bin/activate"
