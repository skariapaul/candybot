# "Hey Zen" wake word training

Trains a custom openWakeWord model so candybot can be activated hands-free by saying "Hey Zen", instead of push-to-talk. See `docs/VOICE_MODES.md` for how trigger modes work at runtime, and `../../candybot/orchestrator/run.py`'s `_prompt_trigger_mode()` for switching between push-to-talk and wake-word at launch.

## Why this is its own isolated setup

`trainer/` (vendored from [lgpearson1771/openwakeword-trainer](https://github.com/lgpearson1771/openwakeword-trainer), see `VENDORED.md`) needs its own pinned dependency stack -- notably a genuinely separate PyTorch install and, critically, **Python 3.11**, not candybot's main Python 3.12 venv. `piper-phonemize` (a training-only dependency) has no Linux wheel for Python 3.12 as of this writing, only up to 3.11.

## One-time setup

```bash
# 1. Get a portable Python 3.11 (no sudo needed, no system Python version conflict)
mkdir -p ~/.local/python-standalone && cd ~/.local/python-standalone
curl -sL -o cpython311.tar.gz \
  "https://github.com/astral-sh/python-build-standalone/releases/latest/download/cpython-3.11.15+20260718-x86_64-unknown-linux-gnu-install_only.tar.gz"
tar -xzf cpython311.tar.gz

# 2. Create the training venv
cd <repo-root>/training/wakeword/trainer
~/.local/python-standalone/python/bin/python3.11 -m venv .venv-wakeword
source .venv-wakeword/bin/activate

# 3. Install requirements -- CC/CXX override needed because the standalone
#    Python build defaults to clang for compiling extensions (webrtcvad has
#    one), and this machine only has gcc.
CC=gcc CXX=g++ pip install -r requirements.txt

# 4. Swap in the ROCm PyTorch build (requirements.txt pulls the default
#    CUDA-oriented PyPI build, which won't see this AMD GPU)
pip uninstall -y torch torchaudio
pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/rocm6.3
```

Verify: `python train_wakeword.py --config configs/hey_zen.yaml --step check-env` should report `CUDA: AMD Radeon Graphics` (not a warning about CPU-only) -- confirmed working on this laptop's gfx902 iGPU via `HSA_OVERRIDE_GFX_VERSION=9.0.0` (same override the main candybot venv uses, see `docs/SETUP_DEV_MACHINE.md`).

## Running the pipeline

Prefer the wrapper scripts at `training/train_wakeword_local.sh` (this laptop, GPU-targeted) or `training/train_wakeword.sh` (syncs to the MI300X, in case this laptop's older iGPU turns out too slow for the actual augment/train compute once it's not just device *detection* being tested):

```bash
./training/train_wakeword_local.sh                  # full 13-step pipeline
./training/train_wakeword_local.sh --list-steps      # see all steps
./training/train_wakeword_local.sh --from augment     # resume from a specific step
./training/train_wakeword_local.sh --step verify-clips  # run just one step
```

Each step is independently verified by the pipeline itself -- if one fails, it tells you exactly how to resume rather than restarting from scratch.

## The config: `trainer/configs/hey_zen.yaml`

Only the identity/phrase fields differ from the fork's example (`hey_echo.yaml`):
- `target_phrase: ["hey zen"]`
- `custom_negative_phrases`: phonetically-similar confusables ("hey ten", "hazen", "amazon", etc.) the model should explicitly learn to reject

Everything else (sample counts, model architecture, training steps) is the fork author's tuned defaults -- see the file itself for the full list, and `trainer/README.md`'s "Configuration Reference" table for what each field does.

## After training

Two files land in `trainer/export/`: `hey_zen.onnx` (model graph) and `hey_zen.onnx.data` (weights) -- **both required together**. Copy both to `candybot/models/wakeword/`, then set `configs/candybot.yaml`'s `voice.wake_word.model` to `"hey_zen"` (matching how `voice.wake_word.model: hey_jarvis` currently resolves to the bundled interim model). No code changes needed -- `candybot/voice/wakeword.py` already loads whatever model name is configured, and the trigger-mode prompt's displayed label updates automatically from that filename.

## Cleanup

`data/` (~15GB downloaded datasets) and `output/` (intermediate training artifacts) can be deleted once `export/` has the final model -- see `trainer/README.md`'s Cleanup section.

## Threshold tuning

If "Hey Zen" triggers on background chatter, or needs to be over-pronounced to trigger at all, see `trainer/README.md`'s "Threshold Tuning" section -- adjust `configs/candybot.yaml`'s `voice.wake_word.threshold` (0.5 default) up or down without retraining, or add more `custom_negative_phrases` and retrain for a real fix.
