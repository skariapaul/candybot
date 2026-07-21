"""Always-listening wake-word trigger via openWakeWord.

Loads either the custom-trained "hey_zen" model (see training/wakeword/) from
candybot/models/wakeword/, or one of openWakeWord's bundled pretrained
community models (e.g. "hey_jarvis") by name. Fully local, ONNX-based, no API
key.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import openwakeword
import sounddevice as sd
from openwakeword.model import Model

logger = logging.getLogger(__name__)

_TARGET_SR = 16000  # what openWakeWord's models expect
_CHUNK_DURATION_S = 0.08  # ~80ms chunks, openWakeWord's expected granularity

_CUSTOM_MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "wakeword"

# (Model, actual-prediction-key) -- openWakeWord derives its own internal key
# from the loaded file's basename (e.g. "hey_jarvis_v0.1.onnx" -> "hey_jarvis_v0.1"),
# which doesn't necessarily match the config's model_name, so it's captured
# once here at load time rather than assumed.
_model_cache: dict[str, tuple[Model, str]] = {}


def _resolve_model_path(model_name: str) -> str:
    custom_path = _CUSTOM_MODELS_DIR / f"{model_name}.onnx"
    if custom_path.exists():
        return str(custom_path)
    for path in openwakeword.get_pretrained_model_paths():
        if os.path.basename(path).startswith(model_name):
            return path
    raise FileNotFoundError(
        f"No wake-word model found for '{model_name}' "
        f"(checked {custom_path} and openWakeWord's bundled pretrained models)"
    )


def _get_model(model_name: str) -> tuple[Model, str]:
    if model_name not in _model_cache:
        model = Model(wakeword_model_paths=[_resolve_model_path(model_name)])
        loaded_key = next(iter(model.models))  # the key openWakeWord actually assigned
        _model_cache[model_name] = (model, loaded_key)
    return _model_cache[model_name]


def wait_for_wake_word(model_name: str, threshold: float, device: int | None, sample_rate: int) -> None:
    """Blocks until the wake word is detected in the live mic stream.

    Records at the input device's own native rate and resamples each chunk to
    the 16kHz openWakeWord requires -- not every device accepts an InputStream
    opened at an arbitrary rate (e.g. this laptop's built-in mic defaults to
    48kHz), the same class of PortAudioError play_audio() works around for
    output -- see audio_io.py.
    """
    model, loaded_key = _get_model(model_name)
    native_sr = int(sd.query_devices(device, kind="input")["default_samplerate"])
    chunk_size = max(1, int(round(native_sr * _CHUNK_DURATION_S)))

    with sd.InputStream(
        samplerate=native_sr, channels=1, dtype="int16", device=device, blocksize=chunk_size
    ) as stream:
        while True:
            block, _ = stream.read(chunk_size)
            audio = block[:, 0]
            if native_sr != _TARGET_SR:
                audio = _resample_int16(audio, native_sr, _TARGET_SR)
            predictions = model.predict(audio)
            score = predictions.get(loaded_key, 0.0)
            if score >= threshold:
                logger.info(f"Wake word '{model_name}' detected (score={score:.2f})")
                return


def _resample_int16(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if len(audio) == 0:
        return audio
    target_len = int(round(len(audio) * target_sr / orig_sr))
    audio_f = audio.astype(np.float32)
    resampled = np.interp(np.linspace(0, len(audio_f) - 1, num=target_len), np.arange(len(audio_f)), audio_f)
    return resampled.astype(np.int16)
