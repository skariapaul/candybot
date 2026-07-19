"""Camera enumeration and access for candybot.

The arm-mounted USB camera exposes two /dev/video* nodes (common for UVC
cameras -- one is the real capture node, the other metadata-only). Node
numbering also isn't guaranteed stable across reconnects/reboots. So instead
of trusting a fixed index, resolve_camera_device() actually tries to read a
frame from each candidate via OpenCV and picks the first one that works.
"""

from __future__ import annotations

import glob
import logging
import subprocess
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def list_video_devices() -> list[str]:
    return sorted(glob.glob("/dev/video*"))


def resolve_camera_device(preferred: str | None = None) -> str:
    """Returns the first /dev/video* node that actually yields a real frame.

    Tries `preferred` first (e.g. configs/candybot.yaml's camera.device), then
    falls back to probing every other detected node.
    """
    candidates = ([preferred] if preferred else []) + [
        d for d in list_video_devices() if d != preferred
    ]

    tried = []
    for device in candidates:
        tried.append(device)
        cap = cv2.VideoCapture(device)
        try:
            if not cap.isOpened():
                if device == preferred:
                    logger.warning(f"Preferred camera {device} could not be opened (busy/disconnected?) -- falling back to another device.")
                continue
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                if device != preferred and preferred is not None:
                    logger.warning(f"Preferred camera {preferred} unavailable -- using {device} instead. Verify this is really the arm-mounted camera, not e.g. a built-in webcam.")
                else:
                    logger.info(f"Resolved camera device: {device} (frame shape {frame.shape})")
                return device
        finally:
            cap.release()

    raise RuntimeError(f"No working camera found among: {tried}")


def apply_exposure_controls(
    device: str,
    auto_exposure: int | None,
    exposure_time_absolute: int | None,
    gain: int | None,
) -> None:
    """Applies V4L2 exposure/gain controls via v4l2-ctl. These are device-wide,
    kernel-level settings -- they apply live and persist regardless of which
    process (candybot's own OpenCVCamera, or another cv2.VideoCapture) holds
    the capture handle open. Booth lighting varies by venue; this camera's
    "Aperture Priority" auto-exposure badly underexposed this location's
    mixed lighting (near-black frames) -- see configs/candybot.yaml's
    camera.* fields for the tunable values.
    """
    ctrls = []
    if auto_exposure is not None:
        ctrls.append(f"auto_exposure={auto_exposure}")
    if exposure_time_absolute is not None:
        ctrls.append(f"exposure_time_absolute={exposure_time_absolute}")
    if gain is not None:
        ctrls.append(f"gain={gain}")
    if not ctrls:
        return

    result = subprocess.run(
        ["v4l2-ctl", "-d", device, "--set-ctrl", ",".join(ctrls)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning(f"Failed to apply exposure controls to {device}: {result.stderr.strip()}")
    else:
        logger.info(f"Applied exposure controls to {device}: {', '.join(ctrls)}")


def make_camera_config(device: str, width: int, height: int, fps: int):
    """Builds a lerobot OpenCVCameraConfig for use in SO101FollowerConfig(cameras=...)."""
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    return OpenCVCameraConfig(index_or_path=Path(device), width=width, height=height, fps=fps)
