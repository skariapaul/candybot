"""Local TTS via Piper's Python API, with the loaded voice cached across calls.

Originally shelled out to the `piper` CLI per-call for interface stability,
but that reloads the entire ONNX model from disk on every single utterance
-- measured at ~2.2s to synthesize ~2s of audio (i.e. barely real-time),
identical on repeated calls, confirming it was paying full model-load cost
every time rather than being compute-bound. A cached PiperVoice instance
(same pattern as asr.py's _model_cache) drops that to ~0.2s per utterance
after the one-time ~1.8s load, an order of magnitude faster -- this was the
dominant source of per-turn latency in the live demo.

Expects the voice model at models/<voice_model>.onnx (+ .onnx.json sidecar) --
see docs/SETUP_DEV_MACHINE.md for the one-time download step.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from piper import PiperVoice

logger = logging.getLogger(__name__)

_voice_cache: dict[str, PiperVoice] = {}


def _get_voice(voice_model: str, models_dir: str) -> PiperVoice:
    if voice_model not in _voice_cache:
        model_path = Path(models_dir) / f"{voice_model}.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Piper voice model not found at {model_path}. See docs/SETUP_DEV_MACHINE.md to download it."
            )
        _voice_cache[voice_model] = PiperVoice.load(model_path)
    return _voice_cache[voice_model]


def synthesize(text: str, voice_model: str, models_dir: str = "models") -> tuple[np.ndarray, int]:
    """Returns (samples, sample_rate) of `text` spoken in `voice_model`."""
    voice = _get_voice(voice_model, models_dir)
    chunks = list(voice.synthesize(text))
    if not chunks:
        return np.zeros(0, dtype=np.float32), voice.config.sample_rate
    samples = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
    return samples, chunks[0].sample_rate


def compute_envelope(samples: np.ndarray, sample_rate: int, window_s: float = 0.05) -> list[float]:
    """RMS amplitude per ~50ms window, normalized 0..1 by this utterance's own
    peak -- used to drive Zen's mouth animation (candybot/dashboard/static/
    avatar.js) via a pre-computed timeline rather than live audio analysis in
    the browser. See orchestrator/run.py's speak() closure.
    """
    window_size = max(1, int(sample_rate * window_s))
    n_windows = max(1, len(samples) // window_size)
    envelope = []
    for i in range(n_windows):
        chunk = samples[i * window_size : (i + 1) * window_size]
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2))) if len(chunk) else 0.0
        envelope.append(rms)
    peak = max(envelope) if envelope else 0.0
    if peak > 1e-6:
        envelope = [min(1.0, v / peak) for v in envelope]
    return envelope


def speak(text: str, voice_model: str, output_device: int | None = None) -> None:
    from candybot.voice.audio_io import play_audio

    samples, sample_rate = synthesize(text, voice_model)
    play_audio(samples, sample_rate, device=output_device)
