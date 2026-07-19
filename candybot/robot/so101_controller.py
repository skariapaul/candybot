"""Thin wrapper around lerobot's SO101Follower for candybot.

Keeps the rest of candybot decoupled from lerobot's exact class/import shape,
so a future lerobot version bump only touches this one file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from candybot.config import REPO_ROOT, CandybotConfig
from candybot.robot.camera import make_camera_config, resolve_camera_device

logger = logging.getLogger(__name__)

# Neutral, safe rest position in each motor's normalized range (-100..100 for
# arm joints, 0..100 for the gripper). Arbitrary placeholder until
# scripted_actions.py's waypoint capture pass records a real one for this
# physical mount -- see candybot/robot/safety.py.
REST_POSITION: dict[str, float] = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": 0.0,
    "elbow_flex.pos": 0.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 0.0,  # closed
}


class SO101Controller:
    """connect/observe/act/home wrapper around lerobot.robots.so101_follower.SO101Follower."""

    def __init__(self, config: CandybotConfig):
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

        self._config = config
        camera_device = resolve_camera_device(preferred=config.camera.device)
        camera_cfg = make_camera_config(
            camera_device, config.camera.width, config.camera.height, config.camera.fps
        )

        robot_cfg = SO101FollowerConfig(
            port=config.robot.port,
            id=config.robot.id,
            # Must match scripts/calibrate_arm.sh's --robot.calibration_dir, or connect() will think
            # there's no calibration on file and prompt to recalibrate every time.
            calibration_dir=Path(REPO_ROOT) / "configs",
            cameras={"wrist": camera_cfg},
            max_relative_target=config.robot.max_relative_target,
        )
        self._robot = SO101Follower(robot_cfg)

    def connect(self, calibrate: bool = True) -> None:
        self._robot.connect(calibrate=calibrate)
        logger.info("SO-101 follower connected.")

    def disconnect(self) -> None:
        self._robot.disconnect()
        logger.info("SO-101 follower disconnected.")

    def get_observation(self) -> dict[str, Any]:
        return self._robot.get_observation()

    def send_action(self, action: dict[str, float]) -> dict[str, Any]:
        return self._robot.send_action(action)

    def home(self) -> None:
        logger.info("Homing to rest position.")
        self.send_action(REST_POSITION)

    @property
    def is_connected(self) -> bool:
        return self._robot.is_connected
