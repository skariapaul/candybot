# Setup: this dev machine

Hardware: AMD Ryzen 7 PRO 3700U w/ Vega Mobile "Picasso" iGPU (`gfx902`), Ubuntu. **Not** the final booth machine — see `docs/SETUP_BOOTH_MACHINE.md` — but the SO-101 follower arm and its camera are physically attached here, so real hardware bring-up happens on this machine.

## gfx902 / ROCm caveat — read first

ROCm 6.3 officially dropped `gfx900`-family support. `rocminfo` shows this iGPU as a valid KFD compute agent, but that only means the kernel driver (`amdgpu`) sees it — it does **not** guarantee ROCm's userspace math libraries (rocBLAS/MIOpen) work correctly for this arch. Since this iGPU also drives the desktop, a bad kernel launch carries real risk of hanging the session.

Per project direction, **the GPU is the target, not a nice-to-have** — `scripts/setup_env.sh` installs the ROCm PyTorch wheel with `HSA_OVERRIDE_GFX_VERSION=9.0.0` (`gfx900`, nearest officially-supported neighbor) by default. `candybot/hardware_probe.py` validates this with a real tensor op **run in a subprocess with a timeout**, so a hang doesn't take down your shell/desktop — if it fails, everything falls back to CPU automatically, but CPU is the fallback, not the plan.

**Confirmed working as of 2026-07-19**: `HSA_OVERRIDE_GFX_VERSION=9.0.0` lets `torch` run real tensor ops on this iGPU, reporting device "AMD Radeon Graphics" -- no need to try `9.0.6`/`9.0.9`.

## Steps

```bash
# System deps
sudo apt update && sudo apt install -y python3-setuptools python3-wheel cmake build-essential \
  ffmpeg libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev \
  libportaudio2 v4l-utils

./scripts/setup_env.sh          # conda env, ROCm PyTorch wheel, pinned lerobot v0.4.1
./scripts/install_udev_rules.sh # stable /dev/so101_follower symlink, dialout group
python scripts/probe_hardware.py  # GPU/ROCm status, serial ports, cameras, audio devices — run this after every reboot/hardware change
```

Log out/in once after `install_udev_rules.sh` for the `dialout`/`audio` group membership to take effect.

## Voice model download (one-time)

Piper TTS needs its voice model downloaded once (gitignored under `models/`, not vendored):

```bash
mkdir -p models
python -m piper.download_voices en_US-lessac-medium --data-dir models
```

## Physical setup

Place two distinct, fixed bins — one for chocolate, one for candy — within the follower arm's reach, in the same layout `scripted_actions.py`'s waypoints assume. Document the exact positions/measurements here once `scripted_actions.py` waypoints are captured, so the layout can be reproduced at the booth.

## Verifying

- `arecord -l` should list the USB headset as a capture device (look for `USB Audio`).
- `v4l2-ctl --list-formats-ext -d /dev/video4` (or whichever node `probe_hardware.py` reports) to confirm the arm-mounted camera capture node, distinct from the laptop's built-in webcam.
- `ls -la /dev/so101_follower` should resolve to the arm's `ttyACM*` device after the udev rule is installed.
