`trainer/` is vendored from [lgpearson1771/openwakeword-trainer](https://github.com/lgpearson1771/openwakeword-trainer), commit `34e5690` (2026-02-13), MIT license (Luke Pearson).

Chosen over openWakeWord's own official training notebook because that notebook pins `tensorflow-cpu==2.8.1` among other stale deps that predate Python 3.12 wheel support -- this fork avoids TensorFlow entirely (uses `torch.onnx.export` directly) and patches known torchaudio 2.10+/speechbrain/piper breaking changes. See `trainer/docs/training_notes.md` for the fork author's own notes on why.

To update: re-download the same way, diff against `trainer/`, and re-apply candybot-specific changes (`configs/hey_zen.yaml`, anything referenced from `training/train_wakeword*.sh`).
