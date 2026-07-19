"""Always-listening wake-word trigger via openWakeWord.

Uses an interim pretrained community model ("hey_jarvis") until a custom
"hey candybot" model is trained -- flagged as a stretch goal, see
docs/VOICE_MODES.md. Fully local, ONNX-based, no API key.
"""

from __future__ import annotations

import logging

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

logger = logging.getLogger(__name__)

_TARGET_SR = 16000  # what openWakeWord's models expect
_CHUNK_DURATION_S = 0.08  # ~80ms chunks, openWakeWord's expected granularity

_model_cache: dict[str, Model] = {}


def _get_model(model_name: str) -> Model:
    if model_name not in _model_cache:
        _model_cache[model_name] = Model(wakeword_models=[model_name])
    return _model_cache[model_name]


def wait_for_wake_word(model_name: str, threshold: float, device: int | None, sample_rate: int) -> None:
    """Blocks until the wake word is detected in the live mic stream.

    Records at the input device's own native rate and resamples each chunk to
    the 16kHz openWakeWord requires -- not every device accepts an InputStream
    opened at an arbitrary rate (e.g. this laptop's built-in mic defaults to
    48kHz), the same class of PortAudioError play_audio() works around for
    output -- see audio_io.py.
    """
    model = _get_model(model_name)
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
            score = predictions.get(model_name, 0.0)
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
