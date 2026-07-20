# What's actually working vs. blocked

Kept up to date as modules land — check here before assuming something works.

## Fully testable on the current dev laptop

- Arm connect / calibrate / keyboard-teleop (`scripts/calibrate_arm.sh`, `scripts/teleop_test.sh`)
- `scripted_actions.py` real pick-and-hand motion for **both** bins (chocolate, candy) — no ML required
- Camera enumeration and streaming
- Full voice stack against the real USB headset: ASR, wake-word, push-to-talk, TTS
- `dialogue.py` name-capture, item-choice, and open-command-capture retry/confirm logic (unit tests with canned transcripts; live tests by speaking) -- `CommandCaptureSession` (for `action_mode: smolvla`) included
- Dashboard: telemetry, camera feed, transcript log, live FSM state, and **Zen** (the CPU-chip mascot avatar) -- fully testable today with zero external assets, no robot/training dependency (see `docs/ARCHITECTURE.md`)
- **The complete orchestrator loop end-to-end** using scripted motion in place of a trained policy — i.e. a full working demo (greet → name → chocolate-or-candy → pick correct item → hand over → thank you) is achievable without any ML training
- GPU inference validation (`hardware_probe.py` against gfx902 via `HSA_OVERRIDE_GFX_VERSION=9.0.0`) — **confirmed working** as of 2026-07-19: `torch.cuda.is_available()` is True and a real tensor op succeeds, reporting device "AMD Radeon Graphics". Unofficial-but-functional on this arch.
- The pure-logic unit test suite (`pytest tests/`) — 20/20 passing (hardware probe fallback logic, dialogue retry/confirm incl. CommandCaptureSession, orchestrator FSM transitions)
- `scripts/probe_hardware.py` confirmed end to end as of 2026-07-19: stable `/dev/so101_follower` symlink present, camera resolves to `/dev/video4`/`5` ("USB2.0_CAM1", distinct from the built-in HP HD/IR cameras at video0-3), GPU report as above. USB headset was unplugged at check time -- `find_device()`'s fallback-to-default logic (not yet exercised live) covers this, but voice testing needs it plugged back in.

## Blocked / deferred

- Demo-quality `lerobot-record` dataset collection (both `scripts/record_dataset.sh` and the smolVLA-specific `scripts/record_dataset_smolvla.sh`) — needs the SO-101 **leader** arm (only the follower is attached today)
- Actual ACT or smolVLA policy training — needs MI300X connection details (not yet specified in `.env`); see `docs/SMOLVLA.md` for the smolVLA-specific workflow
- `policy_runtime.py`'s real inference paths (`_run()` for ACT, `run_command()` for smolVLA) — both built and wired behind `robot.action_mode: scripted|policy|smolvla`, but no trained checkpoint exists yet for either
- True trade-show-floor acoustics (background noise, crowd chatter) — only desk-approximable today
- The real physical push-to-talk button — a keyboard stand-in (`space` key, read from the terminal's own stdin, see `docs/VOICE_MODES.md`) ships now, swappable later since most such buttons enumerate as HID keyboards and will work the same way as long as the terminal has focus
- Performance/behavior on the actual booth hardware — see `docs/SETUP_BOOTH_MACHINE.md`

## One-time physical setup required

Two distinct, fixed bins (chocolate, candy) must be placed within the arm's reach before `scripted_actions.py`'s waypoints will pick the correct item — see `docs/SETUP_DEV_MACHINE.md`.
