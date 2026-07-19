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
                continue
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                logger.info(f"Resolved camera device: {device} (frame shape {frame.shape})")
                return device
        finally:
            cap.release()

    raise RuntimeError(f"No working camera found among: {tried}")


def make_camera_config(device: str, width: int, height: int, fps: int):
    """Builds a lerobot OpenCVCameraConfig for use in SO101FollowerConfig(cameras=...)."""
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    return OpenCVCameraConfig(index_or_path=Path(device), width=width, height=height, fps=fps)
