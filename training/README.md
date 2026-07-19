# Training (remote MI300X)

Status: scripts written and ready, **not yet executed** -- remote host connection details aren't specified yet. See `docs/TRAINING_MI300X.md` for the full workflow narrative; this is the operational quick-reference.

## Prerequisites

1. Fill in `.env` at the repo root: `REMOTE_HOST`, `REMOTE_USER`, `REMOTE_SSH_KEY`, and optionally `REMOTE_CANDYBOT_DIR` (defaults to `~/candybot`).
2. `HF_USER` / `HF_TOKEN` in `.env` -- datasets and checkpoints move via the Hugging Face Hub, not raw file copy (though `sync_from_remote.sh` is a fallback if that's not viable at the venue).
3. The SO-101 **leader** arm connected, for demo-quality `scripts/record_dataset.sh` recordings. Without it, that script falls back to keyboard teleop -- fine to test the pipeline shape, not for real training data.

## One-time remote setup

```bash
./sync_to_remote.sh                     # push this repo's training/ + configs to the remote host
ssh -i $REMOTE_SSH_KEY $REMOTE_USER@$REMOTE_HOST
cd candybot && ./training/remote_env_setup.sh   # ROCm 6.3 PyTorch + lerobot==0.4.1 -- MI300X is gfx942, officially supported, no override needed
huggingface-cli login                    # paste HF_TOKEN
```

## Per bin (chocolate, candy)

```bash
# On the edge laptop:
./scripts/record_dataset.sh chocolate <leader-port>   # pushes to HF Hub automatically

# On the remote MI300X host:
./training/train_act.sh chocolate   # defaults to 100k steps; pushes checkpoint to HF Hub

# Back on the edge laptop:
./scripts/pull_checkpoint.sh chocolate
# then set configs/candybot.yaml: robot.bins.chocolate.policy_checkpoint to the printed path,
# and robot.action_mode: policy
```

Repeat for `candy`. Each bin is its own fixed-task ACT policy for the MVP (no language conditioning) -- see `docs/ARCHITECTURE.md`.
