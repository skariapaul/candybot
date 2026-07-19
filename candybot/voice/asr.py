"""Local ASR via faster-whisper (CTranslate2), with built-in VAD trimming.

Runs on CPU (int8) deliberately, not the GPU that candybot.hardware_probe
targets for robot policy inference -- keeps the two workloads from contending
for the same iGPU, and CPU int8 latency is plenty for short name/item
utterances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

_model_cache: dict[str, WhisperModel] = {}


@dataclass
class TranscriptionResult:
    text: str
    avg_logprob: float
    no_speech_prob: float

    @property
    def is_confident(self) -> bool:
        return bool(self.text.strip()) and self.no_speech_prob < 0.6 and self.avg_logprob > -1.5


def _get_model(model_size: str) -> WhisperModel:
    if model_size not in _model_cache:
        _model_cache[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _model_cache[model_size]


def transcribe(
    audio: np.ndarray, sample_rate: int, model_size: str = "base.en", vad_filter: bool = True
) -> TranscriptionResult:
    if sample_rate != 16000:
        raise ValueError(f"faster-whisper expects 16kHz audio, got {sample_rate}")

    model = _get_model(model_size)
    segments, _info = model.transcribe(audio, vad_filter=vad_filter, language="en")
    segments = list(segments)
    if not segments:
        return TranscriptionResult(text="", avg_logprob=-999.0, no_speech_prob=1.0)

    text = " ".join(s.text.strip() for s in segments).strip()
    avg_logprob = sum(s.avg_logprob for s in segments) / len(segments)
    no_speech_prob = sum(s.no_speech_prob for s in segments) / len(segments)
    return TranscriptionResult(text=text, avg_logprob=avg_logprob, no_speech_prob=no_speech_prob)
