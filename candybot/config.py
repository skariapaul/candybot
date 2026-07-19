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
    # Training dataset repo_id (e.g. "<hf_user>/candybot_chocolate") -- policy_runtime.py
    # needs this to load normalization stats via LeRobotDatasetMetadata, not just the
    # checkpoint weights. Must match what training/train_act.sh trained against.
    dataset_repo_id: str | None = None


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
    # V4L2 exposure controls -- this camera's auto-exposure ("Aperture Priority")
    # badly underexposed this room's mixed lighting (near-black frames). Manual
    # values are venue-specific; re-tune with `v4l2-ctl -d <device> --list-ctrls`
    # and `--set-ctrl=...` at a new location. auto_exposure: 1=manual, 3=auto.
    auto_exposure: int | None = 1
    exposure_time_absolute: int | None = 400
    gain: int | None = 30


class AudioProfile(BaseModel):
    label: str | None = None  # human-readable name shown in the startup prompt; falls back to the profile key
    input_device_name_hint: str
    output_device_name_hint: str


class AudioConfig(BaseModel):
    profile: str = "usb_headset"
    profiles: dict[str, AudioProfile]
    sample_rate: int

    @property
    def input_device_name_hint(self) -> str:
        return self.profiles[self.profile].input_device_name_hint

    @property
    def output_device_name_hint(self) -> str:
        return self.profiles[self.profile].output_device_name_hint


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
    if audio_profile := os.environ.get("CANDYBOT_AUDIO_PROFILE"):
        raw["audio"]["profile"] = audio_profile

    config = CandybotConfig(**raw)
    if config.audio.profile not in config.audio.profiles:
        raise ValueError(
            f"audio.profile {config.audio.profile!r} not in configured profiles: {sorted(config.audio.profiles)}"
        )
    return config
