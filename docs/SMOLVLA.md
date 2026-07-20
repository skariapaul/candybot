# smolVLA: open-ended language-conditioned picking

Status: code path built and wired (`configs/candybot.yaml`'s `robot.action_mode: smolvla`, `candybot/robot/policy_runtime.py`'s `run_command()`, `candybot/voice/dialogue.py`'s `CommandCaptureSession`), **not yet trained** -- same blockers as ACT training (`docs/TRAINING_MI300X.md`): needs the SO-101 leader arm for real demonstration data, and remote MI300X connection details in `.env`.

## Why smolVLA instead of ACT

ACT (`docs/ARCHITECTURE.md`'s original design) is a fixed-task policy: one checkpoint per bin, trained to always do the same motion. smolVLA is a genuine vision-language-action model -- given the camera image and a natural-language instruction, it predicts actions directly, no separate object-detection stage. That means the demo can move from "chocolate or candy?" (binary, pre-baked) to "what would you like?" (open-ended: "pick up the gold cup", "the one on the left", etc.), with the model actually reasoning about the scene and the words together.

smolVLA is a 450M-parameter model, ~2GB VRAM, <100ms inference latency -- fits this laptop's 2GB shared VRAM budget comfortably, confirmed against public benchmarks (not yet measured on this specific gfx902 iGPU, since no checkpoint exists yet to test with).

## How it's wired

- `robot.action_mode: smolvla` (alongside `scripted`/`policy`) selects this path in `candybot/orchestrator/run.py`: instead of `ItemChoiceSession`'s chocolate/candy classification, it runs `CommandCaptureSession` (capture → confirm → retry → fallback, same shape, but the raw confirmed transcript *is* the command -- no keyword classification).
- `candybot/robot/policy_runtime.py`'s `run_command(controller, config, command)` loads the policy via the same `make_policy`/`make_pre_post_processors`/`predict_action` pipeline as ACT's `_run()` (they share `_load_policy()` and `_run_policy_loop()` -- policy-agnostic, since `PreTrainedConfig.from_pretrained()` reads the real type from the checkpoint), but passes `command` as `predict_action`'s `task=` argument instead of a fixed per-bin string. **This is the actual point of a VLA model** -- ACT's task string is just a label; smolVLA's is the real conditioning signal.
- `configs/candybot.yaml`'s `robot.smolvla` section (`checkpoint`, `dataset_repo_id`) is a single unified policy, not one per bin.

## Training workflow

1. **Collect varied demonstrations** (`scripts/record_dataset_smolvla.sh "<instruction>" <leader-port> [episodes]`), run **multiple times with different instruction text** -- unlike ACT's one-fixed-task-per-bin recording, generalization past the exact two demo cups depends on variety in both phrasing and physical setup (move the cups, use different words) across recording sessions. All sessions accumulate into the same `<hf_user>/candybot_smolvla` dataset.
   ```bash
   ./scripts/record_dataset_smolvla.sh "pick up the gold cup and hand it to the visitor" /dev/ttyACM1 15
   ./scripts/record_dataset_smolvla.sh "grab the white cup" /dev/ttyACM1 15
   ./scripts/record_dataset_smolvla.sh "give me the one on the left" /dev/ttyACM1 15
   # ...repeat with the cups physically moved between sessions
   ```
2. **Fine-tune on the MI300X host** (`training/train_smolvla.sh [steps]`) -- starts from the pretrained `lerobot/smolvla_base` checkpoint (`--policy.path=lerobot/smolvla_base`, verified against lerobot 0.4.1's own `TrainPipelineConfig`) rather than training from scratch, so far fewer steps are needed than ACT's from-scratch 100k (default 20k, adjust based on how training loss looks).
3. **Pull the checkpoint back** (`scripts/pull_checkpoint.sh smolvla`) and set `configs/candybot.yaml`'s `robot.smolvla.checkpoint` (the printed local path) and `robot.smolvla.dataset_repo_id` (`<hf_user>/candybot_smolvla`), then `robot.action_mode: smolvla`.

## Camera setup consideration

Upstream lerobot issues discussing SO-101 + smolVLA fine-tuning note that a second, non-wrist camera view (an overhead or front-facing angle, in addition to the wrist cam) often helps spatial grounding for VLA models -- the wrist camera alone can lose sight of the target object during approach. This is flagged as an **optional future enhancement**, not blocking: the current single-wrist-camera setup (`candybot/robot/camera.py`, `SO101Controller`) is what `run_command()` uses today, and can be extended with a second `CameraConfig` entry later if picking accuracy turns out to need it.

## Stretch: a custom wake phrase

Unrelated to smolVLA itself, but worth noting alongside "make the interaction feel more natural": `docs/VOICE_MODES.md` already flags training a custom "hey candybot" wake-word model (currently using the interim `hey_jarvis` community model) as a stretch goal.
