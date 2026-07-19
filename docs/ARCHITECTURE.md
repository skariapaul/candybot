# Architecture

## Demo flow

```
IDLE → GREET → CAPTURE_NAME → CONFIRM_NAME → ASK_ITEM_CHOICE → PICK_AND_HAND → THANK_YOU → RESET
```

Implemented as an async state machine in `candybot/orchestrator/fsm.py` (using `transitions.extensions.asyncio.AsyncMachine`). The orchestrator (`candybot/orchestrator/run.py`) wires together the voice, robot, and dashboard subsystems and drives this FSM.

- **CAPTURE_NAME / CONFIRM_NAME**: delegated to `candybot.voice.dialogue.NameCaptureSession` — retries on low-confidence ASR, strips filler phrases ("my name is..."), confirms against a yes/no keyword set (easier ASR target than a name), falls back to addressing the visitor as "friend" after 3 attempts so the demo never stalls.
- **ASK_ITEM_CHOICE**: delegated to `candybot.voice.dialogue.ItemChoiceSession` — classifies the response against two small keyword sets (chocolate vs candy), defaults to candy after repeated failures.
- **PICK_AND_HAND**: dispatches to `candybot.robot.scripted_actions.pick_chocolate()`/`pick_candy()` (hand-authored waypoints, works today) or `candybot.robot.policy_runtime` (trained ACT checkpoint, once available) based on `robot.action_mode` in `configs/candybot.yaml`.
- **THANK_YOU**: TTS closing line, then back to `RESET`/`IDLE`.

## Subsystems

| Package | Responsibility |
|---|---|
| `candybot.voice` | ASR (faster-whisper), TTS (piper), two trigger modes (push-to-talk via a terminal keypress, wake-word via `openwakeword`), dialogue logic |
| `candybot.robot` | LeRobot SO-101 follower wrapper, camera, scripted pick-and-hand motions, safety limits, (later) trained-policy inference |
| `candybot.dashboard` | FastAPI + WebSocket server showing live camera feed, dialogue transcript/state, and this laptop's own CPU/GPU/ROCm telemetry |
| `candybot.orchestrator` | Top-level FSM and entrypoint tying the above together |
| `candybot.hardware_probe` | Safe GPU/ROCm capability detection used by every module that picks a torch device — never hardcodes `"cuda"` |

## Device selection

`hardware_probe.py` runs a real tensor op in a subprocess with a timeout (a bad kernel launch on this iGPU can hang the desktop session) and exposes a single `get_device() -> "cuda" | "cpu"` used everywhere. Default target on this dev laptop is GPU (gfx902 via `HSA_OVERRIDE_GFX_VERSION=9.0.0`), with CPU as the safety fallback, not the other way around — see `docs/SETUP_DEV_MACHINE.md`.

## Edge/cloud split

Data collection and inference happen at the edge (this laptop today, a proper Ryzen AI laptop at the booth). Policy fine-tuning happens on a remote AMD Instinct MI300X system (`training/`) — datasets and checkpoints move between the two via the Hugging Face Hub. See `docs/TRAINING_MI300X.md`.
