# Setup: real booth machine (Ryzen AI)

Status: not yet done — this dev laptop (Ryzen 7 PRO 3700U, Vega/Picasso gfx902) is a stand-in for hardware bring-up. This doc will fill in once the actual booth laptop (Ryzen AI 300-series or Strix Halo, officially ROCm-supported) is available.

Known differences to account for when moving over:

- **No gfx override needed** (or a different, officially-supported one) — the booth chip is on ROCm's supported GPU list, unlike this dev machine's gfx902. Re-verify with `rocminfo` and drop/adjust the `HSA_OVERRIDE_GFX_VERSION` workaround in `scripts/setup_env.sh` accordingly.
- **Re-run `scripts/probe_hardware.py`** to re-map serial ports, camera device nodes, and audio devices — `/dev/ttyACM0`/`/dev/video4`/USB Audio card indices are not guaranteed to match; the udev rule (`scripts/udev/99-so101.rules`) matching on `idVendor:idProduct` should still work but confirm.
- **Physically remount the two bins** (chocolate, candy) at the same relative position/reach used when the `scripted_actions.py` waypoints and any trained checkpoints were captured — waypoints are calibrated to a specific physical layout, not derived from vision.
- **Re-validate real-time performance** — the whole point of moving to this hardware is that GPU inference should be faster/more reliable than the dev laptop; treat any regression as a real bug, not expected variance.
