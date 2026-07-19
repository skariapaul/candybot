"""Loads configs/candybot.yaml into typed config objects, with env var overrides.

See .env.example for which env vars override which fields.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "candybot.yaml"


class BinConfig(BaseModel):
    waypoints_file: str
    policy_checkpoint: str | None = None


class RobotConfig(BaseModel):
    type: str = "so101_follower"
    port: str
    id: str
    calibration_path: str
    action_mode: str = "scripted"
    max_relative_target: float = 20
    bins: dict[str, BinConfig]


class CameraConfig(BaseModel):
    device: str
    width: int
    height: int
    fps: int


class AudioConfig(BaseModel):
    input_device_name_hint: str
    output_device_name_hint: str
    sample_rate: int


class PushToTalkConfig(BaseModel):
    key: str = "space"


class WakeWordConfig(BaseModel):
    model: str = "hey_jarvis"
    threshold: float = 0.5


class AsrConfig(BaseModel):
    model_size: str = "base.en"
    vad_filter: bool = True


class TtsConfig(BaseModel):
    voice_model: str = "en_US-lessac-medium"


class VoiceConfig(BaseModel):
    trigger_mode: str = "push_to_talk"
    push_to_talk: PushToTalkConfig = PushToTalkConfig()
    wake_word: WakeWordConfig = WakeWordConfig()
    asr: AsrConfig = AsrConfig()
    tts: TtsConfig = TtsConfig()


class DialogueConfig(BaseModel):
    max_name_attempts: int = 3
    max_item_attempts: int = 2
    item_choice_default: str = "candy"


class DashboardConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080


class HardwareConfig(BaseModel):
    preferred_device: str = "cuda"
    hsa_override_gfx_version: str = "9.0.0"


class CandybotConfig(BaseModel):
    robot: RobotConfig
    camera: CameraConfig
    audio: AudioConfig
    voice: VoiceConfig
    dialogue: DialogueConfig
    dashboard: DashboardConfig
    hardware: HardwareConfig


def load_config(path: Path | str | None = None) -> CandybotConfig:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        raw = yaml.safe_load(f)

    if port := os.environ.get("CANDYBOT_ROBOT_PORT"):
        raw["robot"]["port"] = port
    if dash_port := os.environ.get("CANDYBOT_DASHBOARD_PORT"):
        raw["dashboard"]["port"] = int(dash_port)

    return CandybotConfig(**raw)
