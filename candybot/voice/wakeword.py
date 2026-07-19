"""Always-listening wake-word trigger via openWakeWord.

Uses an interim pretrained community model ("hey_jarvis") until a custom
"hey candybot" model is trained -- flagged as a stretch goal, see
docs/VOICE_MODES.md. Fully local, ONNX-based, no API key.
"""

from __future__ import annotations

import logging

import sounddevice as sd
from openwakeword.model import Model

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1280  # openWakeWord expects ~80ms chunks at 16kHz

_model_cache: dict[str, Model] = {}


def _get_model(model_name: str) -> Model:
    if model_name not in _model_cache:
        _model_cache[model_name] = Model(wakeword_models=[model_name])
    return _model_cache[model_name]


def wait_for_wake_word(model_name: str, threshold: float, device: int | None, sample_rate: int) -> None:
    """Blocks until the wake word is detected in the live mic stream."""
    model = _get_model(model_name)

    with sd.InputStream(
        samplerate=sample_rate, channels=1, dtype="int16", device=device, blocksize=_CHUNK_SIZE
    ) as stream:
        while True:
            block, _ = stream.read(_CHUNK_SIZE)
            predictions = model.predict(block[:, 0])
            score = predictions.get(model_name, 0.0)
            if score >= threshold:
                logger.info(f"Wake word '{model_name}' detected (score={score:.2f})")
                return
