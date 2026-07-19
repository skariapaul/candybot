"""Safety helpers shared by scripted_actions.py and (later) policy_runtime.py.

Three layers, deliberately independent of each other so a failure in one
doesn't silently disable another:
  1. Joint limit clamps -- keep any single commanded position within a safe
     normalized range (tighter than the mechanical limit, to avoid collisions
     with the table/bins).
  2. Interpolated motion -- move_through_waypoints() steps between waypoints
     in small increments rather than one big jump, so max_relative_target
     (enforced at the lerobot layer, see so101_controller.py) is naturally
     respected and motion looks deliberate rather than jerky on camera.
  3. A motion watchdog -- aborts (raises) if a motion sequence runs
     unexpectedly long, so a stuck arm doesn't hang a live demo forever.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Conservative placeholder bounds in each motor's normalized range. Arm joints
# use -100..100, the gripper 0..100 (see lerobot's MotorNormMode). TODO: tighten
# these to the real safe range for this physical mount once the bins are placed
# and scripted_actions.py waypoints are captured (see docs/SETUP_DEV_MACHINE.md).
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan.pos": (-100.0, 100.0),
    "shoulder_lift.pos": (-100.0, 100.0),
    "elbow_flex.pos": (-100.0, 100.0),
    "wrist_flex.pos": (-100.0, 100.0),
    "wrist_roll.pos": (-100.0, 100.0),
    "gripper.pos": (0.0, 100.0),
}

DEFAULT_WATCHDOG_S = 20.0


class MotionTimeoutError(RuntimeError):
    """Raised when a motion sequence exceeds its watchdog timeout."""


def clamp_action(action: dict[str, float]) -> dict[str, float]:
    """Clamps every joint in `action` to JOINT_LIMITS, warning if it had to."""
    clamped = {}
    for joint, value in action.items():
        lo, hi = JOINT_LIMITS.get(joint, (-100.0, 100.0))
        new_value = max(lo, min(hi, value))
        if new_value != value:
            logger.warning(f"Clamped {joint}: {value} -> {new_value} (limit {lo}..{hi})")
        clamped[joint] = new_value
    return clamped


def _interpolate(start: dict[str, float], end: dict[str, float], steps: int) -> list[dict[str, float]]:
    joints = end.keys()
    return [
        {j: start.get(j, end[j]) + (end[j] - start.get(j, end[j])) * (i / steps) for j in joints}
        for i in range(1, steps + 1)
    ]


def move_through_waypoints(
    controller: Any,
    waypoints: list[dict[str, float]],
    steps_per_segment: int = 10,
    step_delay_s: float = 0.05,
    watchdog_s: float = DEFAULT_WATCHDOG_S,
) -> None:
    """Moves the arm through `waypoints` in order, interpolating between each pair
    and clamping every intermediate command to JOINT_LIMITS. Raises MotionTimeoutError
    if the whole sequence runs longer than `watchdog_s`.
    """
    if not waypoints:
        return

    start_time = time.monotonic()
    current = {k: v for k, v in controller.get_observation().items() if k.endswith(".pos")}

    for target in waypoints:
        for step in _interpolate(current, target, steps_per_segment):
            if time.monotonic() - start_time > watchdog_s:
                raise MotionTimeoutError(
                    f"Motion sequence exceeded {watchdog_s}s watchdog -- aborting to avoid hanging the demo."
                )
            controller.send_action(clamp_action(step))
            time.sleep(step_delay_s)
        current = target

    logger.info(f"Moved through {len(waypoints)} waypoint(s) in {time.monotonic() - start_time:.1f}s")
