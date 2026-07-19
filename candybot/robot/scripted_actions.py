"""Hand-authored pick-and-hand motions for the two candy bins.

Works without any trained policy -- waypoints are captured via
scripts/capture_waypoints.py by physically posing the arm (torque disabled)
through each stage: approach, over bin, lower, grip, lift, move to handoff,
release, retreat. This is the MVP action path (configs/candybot.yaml's
robot.action_mode: scripted); candybot.robot.policy_runtime is the trained-ACT
alternative once a checkpoint exists, selected via the same config toggle.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from candybot.config import CandybotConfig
from candybot.robot.safety import move_through_waypoints
from candybot.robot.so101_controller import REST_POSITION, SO101Controller

logger = logging.getLogger(__name__)


class WaypointsNotCapturedError(RuntimeError):
    pass


def _load_waypoints(waypoints_file: str) -> list[dict[str, float]]:
    path = Path(waypoints_file)
    if not path.exists():
        raise WaypointsNotCapturedError(
            f"No waypoints captured yet at {path}. Run: python scripts/capture_waypoints.py <chocolate|candy>"
        )
    return json.loads(path.read_text())


def _pick_and_hand(controller: SO101Controller, config: CandybotConfig, bin_name: str) -> None:
    bin_config = config.robot.bins[bin_name]
    waypoints = _load_waypoints(bin_config.waypoints_file)
    logger.info(f"Running scripted pick-and-hand for '{bin_name}' ({len(waypoints)} waypoints)")
    move_through_waypoints(controller, waypoints)
    # Return to a neutral rest pose so the next visitor's greet/idle animation starts clean.
    move_through_waypoints(controller, [REST_POSITION])


def pick_chocolate(controller: SO101Controller, config: CandybotConfig) -> None:
    _pick_and_hand(controller, config, "chocolate")


def pick_candy(controller: SO101Controller, config: CandybotConfig) -> None:
    _pick_and_hand(controller, config, "candy")


ACTIONS = {"chocolate": pick_chocolate, "candy": pick_candy}
