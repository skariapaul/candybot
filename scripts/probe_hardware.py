#!/usr/bin/env python
"""Reports GPU/ROCm status, serial ports (arm), cameras, and audio devices.

Run after `setup_env.sh` and after any reboot/hardware change -- device node
names (/dev/ttyACM*, /dev/video*, ALSA card indices) are not guaranteed stable
across reconnects unless the udev rule (scripts/udev/99-so101.rules) is installed.
"""

from __future__ import annotations

import glob
import json
import subprocess
import sys


def gpu_report() -> None:
    print("=== GPU / ROCm ===")
    try:
        from candybot.hardware_probe import probe

        report = probe()
        print(json.dumps(report.__dict__, indent=2))
    except ImportError:
        print("candybot not installed yet -- run scripts/setup_env.sh first")


def serial_ports() -> None:
    print("\n=== Serial ports (robot arm) ===")
    try:
        from serial.tools import list_ports

        ports = list(list_ports.comports())
        if not ports:
            print("No serial ports found.")
        for p in ports:
            print(f"  {p.device}  vid:pid={p.vid:04x}:{p.pid:04x}  {p.description}" if p.vid else f"  {p.device}  {p.description}")
    except ImportError:
        for dev in sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*") + glob.glob("/dev/so101_*")):
            print(f"  {dev}")

    stable = glob.glob("/dev/so101_follower")
    if stable:
        print(f"  Stable symlink present: {stable[0]} -> install_udev_rules.sh has been run")
    else:
        print("  No stable /dev/so101_follower symlink -- run scripts/install_udev_rules.sh")


def cameras() -> None:
    print("\n=== Cameras ===")
    nodes = sorted(glob.glob("/dev/video*"))
    if not nodes:
        print("No /dev/video* nodes found.")
        return
    for node in nodes:
        name = ""
        try:
            out = subprocess.run(["v4l2-ctl", "--device", node, "--info"], capture_output=True, text=True, timeout=3)
            for line in out.stdout.splitlines():
                if "Card type" in line:
                    name = line.split(":", 1)[1].strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            name = "(install v4l-utils for details: sudo apt install v4l-utils)"
        print(f"  {node}  {name}")


def audio_devices() -> None:
    print("\n=== Audio devices ===")
    try:
        import sounddevice as sd

        for i, dev in enumerate(sd.query_devices()):
            kind = []
            if dev["max_input_channels"] > 0:
                kind.append("in")
            if dev["max_output_channels"] > 0:
                kind.append("out")
            print(f"  [{i}] {dev['name']}  ({'/'.join(kind)})")
    except ImportError:
        try:
            out = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=3)
            print(out.stdout or "(no capture devices reported by arecord)")
        except FileNotFoundError:
            print("Install the 'voice' extra (pip install -e .[voice]) or arecord for details.")


if __name__ == "__main__":
    sys.path.insert(0, ".")
    gpu_report()
    serial_ports()
    cameras()
    audio_devices()
