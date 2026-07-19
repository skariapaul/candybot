"""System telemetry for the dashboard: CPU/RAM (psutil) + amdgpu sysfs GPU
stats. Sysfs comes straight from the in-kernel amdgpu driver, so it works
independent of whether the ROCm userspace stack itself is functional on this
laptop's gfx902 -- see candybot/hardware_probe.py for the separate (and more
cautious) GPU *compute* validation used for actual inference.
"""

from __future__ import annotations

import glob
from pathlib import Path

import psutil


def _find_amdgpu_card() -> Path | None:
    for card_dir in sorted(glob.glob("/sys/class/drm/card*/device")):
        if (Path(card_dir) / "gpu_busy_percent").exists():
            return Path(card_dir)
    return None


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def gpu_stats() -> dict:
    card = _find_amdgpu_card()
    if card is None:
        return {"available": False, "busy_percent": None, "temp_c": None, "vram_percent": None, "vram_used_mb": None, "vram_total_mb": None}

    busy = _read_int(card / "gpu_busy_percent")

    temp_c = None
    for hwmon in sorted(glob.glob(str(card / "hwmon/hwmon*"))):
        raw = _read_int(Path(hwmon) / "temp1_input")
        if raw is not None:
            temp_c = raw / 1000.0
            break

    vram_used = _read_int(card / "mem_info_vram_used")
    vram_total = _read_int(card / "mem_info_vram_total")
    vram_percent = round(100 * vram_used / vram_total, 1) if vram_used is not None and vram_total else None

    return {
        "available": True,
        "busy_percent": busy,
        "temp_c": temp_c,
        "vram_percent": vram_percent,
        "vram_used_mb": round(vram_used / 1024 / 1024, 1) if vram_used is not None else None,
        "vram_total_mb": round(vram_total / 1024 / 1024, 1) if vram_total is not None else None,
    }


def cpu_ram_stats() -> dict:
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_percent": psutil.virtual_memory().percent,
    }


def full_report() -> dict:
    from candybot.hardware_probe import probe

    device_report = probe()
    return {
        "cpu": cpu_ram_stats(),
        "gpu": gpu_stats(),
        "device": {
            "device": device_report.device,
            "gpu_name": device_report.gpu_name,
            "gpu_validated": device_report.gpu_validated,
        },
    }
