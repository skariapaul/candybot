# candybot

A live trade-show demo built for AMD, showcasing **Physical AI** end-to-end: **Ryzen AI** (edge inference), **ROCm** (software stack), and **Instinct MI300X** (training).

A WowRobo SO-101 leader-follower robot arm with an arm-mounted camera greets a visitor, asks their name by voice, asks whether they'd like **chocolate or candy**, picks up the requested item from its bin, hands it over, and thanks them for stopping by.

## How it works

```
IDLE → GREET → CAPTURE_NAME → CONFIRM_NAME → ASK_ITEM_CHOICE → PICK_AND_HAND → THANK_YOU → RESET
```

- **Voice**: local ASR (faster-whisper) + local TTS (piper), triggered either by push-to-talk or an always-listening wake word — switchable depending on how loud the floor is.
- **Robot**: a WowRobo SO-101 follower arm, controlled via [LeRobot](https://github.com/huggingface/lerobot). Ships today with hand-authored pick-and-hand motions for two bins (chocolate, candy); a trained ACT policy is the next step once demonstration data and MI300X training are available.
- **Dashboard**: a local web dashboard on the same laptop, showing live camera feed, dialogue state, and the laptop's own CPU/GPU/ROCm utilization — part of the demo's wow factor.
- **Training**: demonstrations are collected at the edge and fine-tuned on a remote AMD Instinct MI300X system, then the checkpoint is deployed back for local inference.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design and [`docs/SETUP_DEV_MACHINE.md`](docs/SETUP_DEV_MACHINE.md) to get a machine running.

## Quick start (dev machine)

```bash
./scripts/setup_env.sh
./scripts/install_udev_rules.sh
python scripts/probe_hardware.py
./scripts/calibrate_arm.sh
python -m candybot.orchestrator.run
```

## Status

Early build — see [`docs/VERIFICATION.md`](docs/VERIFICATION.md) for what's currently working vs. blocked on hardware/training access.
