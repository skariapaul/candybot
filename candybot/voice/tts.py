"""Local TTS via Piper, invoked as a subprocess (the CLI is the stable interface
across piper-tts versions, unlike its Python bindings).

Expects the voice model at models/<voice_model>.onnx (+ .onnx.json sidecar) --
see docs/SETUP_DEV_MACHINE.md for the one-time download step.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_SAMPLE_RATE = 22050


def _sample_rate_for(model_path: Path) -> int:
    config_path = model_path.with_suffix(model_path.suffix + ".json")
    if config_path.exists():
        try:
            return json.loads(config_path.read_text())["audio"]["sample_rate"]
        except (KeyError, json.JSONDecodeError):
            logger.warning(f"Could not read sample rate from {config_path}, assuming {_DEFAULT_SAMPLE_RATE}")
    return _DEFAULT_SAMPLE_RATE


def synthesize(text: str, voice_model: str, models_dir: str = "models") -> tuple[np.ndarray, int]:
    """Returns (samples, sample_rate) of `text` spoken in `voice_model`."""
    model_path = Path(models_dir) / f"{voice_model}.onnx"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Piper voice model not found at {model_path}. See docs/SETUP_DEV_MACHINE.md to download it."
        )

    result = subprocess.run(
        ["piper", "--model", str(model_path), "--output-raw"],
        input=text.encode("utf-8"),
        capture_output=True,
        check=True,
    )
    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, _sample_rate_for(model_path)


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
