#!/usr/bin/env python
"""Interactive tool to capture pick-and-hand waypoints for a bin.

Usage:
    python scripts/capture_waypoints.py chocolate
    python scripts/capture_waypoints.py candy

Disables torque so you can freely pose the arm by hand (e.g.: approach, over
bin, lower, close gripper, lift, move to handoff, open gripper, retreat).
Press ENTER to record the current pose as the next waypoint. 'undo' drops the
last one, 'done' saves and exits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from candybot.config import load_config
from candybot.robot.so101_controller import SO101Controller


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("chocolate", "candy"):
        print(__doc__)
        sys.exit(1)
    bin_name = sys.argv[1]

    config = load_config()
    controller = SO101Controller(config)
    controller.connect(calibrate=True)
    controller.disable_torque()

    print("Torque disabled -- pose the arm by hand.")
    print("ENTER to record a waypoint, 'undo' to drop the last one, 'done' to save and exit.\n")

    waypoints: list[dict[str, float]] = []
    try:
        while True:
            cmd = input(f"[{len(waypoints)} recorded] > ").strip().lower()
            if cmd == "done":
                break
            if cmd == "undo":
                if waypoints:
                    waypoints.pop()
                    print("Dropped last waypoint.")
                continue
            obs = controller.get_observation()
            waypoint = {k: v for k, v in obs.items() if k.endswith(".pos")}
            waypoints.append(waypoint)
            print(f"Recorded waypoint {len(waypoints)}: {waypoint}")
    finally:
        controller.enable_torque()

    if not waypoints:
        print("No waypoints recorded -- exiting without saving.")
        controller.disconnect()
        return

    out_path = Path(config.robot.bins[bin_name].waypoints_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(waypoints, indent=2))
    print(f"\nSaved {len(waypoints)} waypoints to {out_path}")

    controller.disconnect()


if __name__ == "__main__":
    main()
