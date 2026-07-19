# What's actually working vs. blocked

Kept up to date as modules land — check here before assuming something works.

## Fully testable on the current dev laptop

- Arm connect / calibrate / keyboard-teleop (`scripts/calibrate_arm.sh`, `scripts/teleop_test.sh`)
- `scripted_actions.py` real pick-and-hand motion for **both** bins (chocolate, candy) — no ML required
- Camera enumeration and streaming
- Full voice stack against the real USB headset: ASR, wake-word, push-to-talk, TTS
- `dialogue.py` name-capture and item-choice retry/confirm logic (unit tests with canned transcripts; live tests by speaking)
- Dashboard: telemetry, camera feed, transcript log, live FSM state
- **The complete orchestrator loop end-to-end** using scripted motion in place of a trained policy — i.e. a full working demo (greet → name → chocolate-or-candy → pick correct item → hand over → thank you) is achievable without any ML training
- GPU inference validation itself (`hardware_probe.py` against gfx902 via `HSA_OVERRIDE_GFX_VERSION=9.0.0`) — this is an active target on this machine, not deferred

## Blocked / deferred

- Demo-quality `lerobot-record` dataset collection — needs the SO-101 **leader** arm (only the follower is attached today)
- Actual ACT policy training — needs MI300X connection details (not yet specified in `.env`)
- `policy_runtime.py`'s real inference path — built and wired behind the `scripted|policy` config toggle, but no trained checkpoint exists yet
- True trade-show-floor acoustics (background noise, crowd chatter) — only desk-approximable today
- The real physical push-to-talk button — a keyboard stand-in (`space` key via `pynput`) ships now, swappable later since most such buttons enumerate as HID keyboards
- Performance/behavior on the actual booth hardware — see `docs/SETUP_BOOTH_MACHINE.md`

## One-time physical setup required

Two distinct, fixed bins (chocolate, candy) must be placed within the arm's reach before `scripted_actions.py`'s waypoints will pick the correct item — see `docs/SETUP_DEV_MACHINE.md`.
