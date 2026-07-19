# Training on MI300X

Status: scripts and workflow documented, **not yet executed** — remote MI300X connection details (host/user/access method) aren't specified yet. Fill in `.env` (`REMOTE_HOST`, `REMOTE_USER`, `REMOTE_SSH_KEY`) once available; `training/sync_to_remote.sh` and `train_act.sh` read from there.

## Why edge/cloud split

Demonstration data is collected at the edge (SO-101 leader/follower teleop + camera), but fine-tuning an ACT policy is far faster on a real GPU than on this laptop's iGPU. AMD's own reference pipeline (ROCm Blogs, "Edge-to-Cloud Robotics with AMD ROCm") uses this same split: Ryzen AI edge for data collection + inference, Instinct MI300X for fine-tuning, dataset/checkpoint handoff via the Hugging Face Hub.

## Workflow

1. **Collect demonstrations at the edge** (needs the SO-101 **leader** arm, not yet connected — only the follower is attached today):
   ```bash
   lerobot-record \
     --robot.type=so101_follower --robot.port=/dev/so101_follower \
     --teleop.type=so101_leader --teleop.port=<leader-port> \
     --dataset.repo_id=$HF_USER/candybot_chocolate \
     --dataset.num_episodes=60 --dataset.episode_time_s=20
   ```
   Repeat for a separate `candybot_candy` dataset — each bin is its own fixed-task ACT policy for the MVP (no language conditioning), per `docs/ARCHITECTURE.md`.

2. **Push the dataset** to the Hugging Face Hub (`scripts/push_dataset.sh`, uses `HF_TOKEN`/`HF_USER` from `.env`).

3. **On the MI300X host** (`training/remote_env_setup.sh` — ROCm 6.3 + PyTorch 2.7.1+rocm6.3 + pinned lerobot v0.4.1):
   ```bash
   lerobot-train \
     --dataset.repo_id=$HF_USER/candybot_chocolate \
     --policy.type=act \
     --steps=100000 \
     --output_dir=outputs/train/act_chocolate
   ```
   Checkpoint lands at `outputs/train/act_chocolate/checkpoints/last/pretrained_model/`.

4. **Pull the checkpoint back** (`scripts/pull_checkpoint.sh`) and point `configs/candybot.yaml`'s `robot.bins.chocolate.policy_checkpoint` at it, then set `robot.action_mode: policy`.

## Stretch goal

smolVLA (language-conditioned) fine-tuning, so a single policy could respond to "pick the red one" style requests instead of the current fixed chocolate/candy split — deferred until the ACT baseline is proven live.
