"""ACT policy inference for a bin, using a checkpoint pulled via
scripts/pull_checkpoint.sh. Mirrors lerobot-record's own real-robot policy
inference pattern (make_policy / make_pre_post_processors / predict_action /
build_dataset_frame / make_robot_action) rather than a simplified
reimplementation -- normalization stats live in the training dataset's
metadata, not just the checkpoint weights, so a real LeRobotDatasetMetadata
load is unavoidable.

No trained checkpoint exists yet -- this is the runtime path, wired behind
the same robot.action_mode: scripted|policy config toggle as
scripted_actions.py, ready to use once training/train_act.sh produces one.
"""

from __future__ import annotations

import logging
import time

from candybot.config import CandybotConfig
from candybot.hardware_probe import get_device
from candybot.robot.safety import clamp_action
from candybot.robot.so101_controller import SO101Controller

logger = logging.getLogger(__name__)

_DEFAULT_MAX_STEPS = 300
_DEFAULT_HZ = 15.0


class PolicyNotReadyError(RuntimeError):
    pass


def _load_policy(checkpoint_path: str, dataset_repo_id: str):
    """Loads a trained ACT policy + its pre/post-processors + dataset feature
    schema (needed for normalization and for converting policy output back
    into a robot action dict).
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint_path)
    policy_cfg.pretrained_path = checkpoint_path
    policy_cfg.device = get_device()  # gfx902 falls back to CPU automatically, see hardware_probe.py

    ds_meta = LeRobotDatasetMetadata(dataset_repo_id)
    policy = make_policy(policy_cfg, ds_meta=ds_meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=checkpoint_path,
        device_processor={"device": policy_cfg.device},
    )
    return policy, preprocessor, postprocessor, ds_meta


def _run(
    controller: SO101Controller,
    config: CandybotConfig,
    bin_name: str,
    max_steps: int = _DEFAULT_MAX_STEPS,
    hz: float = _DEFAULT_HZ,
) -> None:
    from lerobot.datasets.utils import build_dataset_frame
    from lerobot.policies.utils import make_robot_action
    from lerobot.utils.constants import OBS_STR
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.utils import get_safe_torch_device

    bin_config = config.robot.bins[bin_name]
    if not bin_config.policy_checkpoint or not bin_config.dataset_repo_id:
        raise PolicyNotReadyError(
            f"Bin '{bin_name}' has no policy_checkpoint/dataset_repo_id configured. "
            f"Run training/train_act.sh then scripts/pull_checkpoint.sh, and set "
            f"configs/candybot.yaml's robot.bins.{bin_name} fields."
        )

    policy, preprocessor, postprocessor, ds_meta = _load_policy(
        bin_config.policy_checkpoint, bin_config.dataset_repo_id
    )
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    task = f"Pick up the {bin_name} and hand it to the visitor."
    period_s = 1.0 / hz

    logger.info(f"Running ACT policy for '{bin_name}' on device={policy.config.device}, {max_steps} steps @ {hz}Hz")

    for step in range(max_steps):
        step_start = time.perf_counter()

        obs = controller.get_observation()
        observation_frame = build_dataset_frame(ds_meta.features, obs, prefix=OBS_STR)

        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=get_safe_torch_device(policy.config.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=task,
            robot_type="so101_follower",
        )
        action = make_robot_action(action_values, ds_meta.features)
        controller.send_action(clamp_action(action))

        elapsed = time.perf_counter() - step_start
        if elapsed < period_s:
            time.sleep(period_s - elapsed)

    logger.info(f"Policy run for '{bin_name}' complete ({max_steps} steps).")


def pick_chocolate(controller: SO101Controller, config: CandybotConfig) -> None:
    _run(controller, config, "chocolate")


def pick_candy(controller: SO101Controller, config: CandybotConfig) -> None:
    _run(controller, config, "candy")


ACTIONS = {"chocolate": pick_chocolate, "candy": pick_candy}
