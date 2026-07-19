"""Low-level audio I/O: device selection and raw recording/playback via sounddevice.

listen_utterance() is the one mode-agnostic entry point the rest of candybot
calls -- dialogue.py doesn't need to know whether push-to-talk or wake-word
triggered it, only how to consume the resulting audio array.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np
import sounddevice as sd

if TYPE_CHECKING:
    from candybot.config import CandybotConfig

logger = logging.getLogger(__name__)


def find_device(name_hint: str, kind: str = "input") -> int:
    """Finds a sounddevice index whose name contains `name_hint` (case-insensitive).

    Falls back to the system default if no match is found, so a missing/renamed
    headset doesn't hard-crash the demo -- just logs a warning.
    """
    channel_field = "max_input_channels" if kind == "input" else "max_output_channels"
    for i, dev in enumerate(sd.query_devices()):
        if name_hint.lower() in dev["name"].lower() and dev[channel_field] > 0:
            return i
    logger.warning(f"No {kind} device matching '{name_hint}' found -- falling back to system default.")
    return sd.default.device[0 if kind == "input" else 1]


def record_stream(
    should_continue: Callable[[np.ndarray], bool],
    sample_rate: int = 16000,
    device: int | None = None,
    block_duration_s: float = 0.1,
    max_duration_s: float = 10.0,
) -> np.ndarray:
    """Records mono float32 audio in blocks, calling should_continue(audio_so_far)
    after each block; stops when it returns False or max_duration_s is hit.
    """
    block_size = int(sample_rate * block_duration_s)
    chunks: list[np.ndarray] = []
    total_samples = 0

    with sd.InputStream(
        samplerate=sample_rate, channels=1, dtype="float32", device=device, blocksize=block_size
    ) as stream:
        while total_samples < max_duration_s * sample_rate:
            block, _ = stream.read(block_size)
            chunks.append(block[:, 0].copy())
            total_samples += len(block)
            if not should_continue(np.concatenate(chunks)):
                break

    return np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")


def play_audio(samples: np.ndarray, sample_rate: int, device: int | None = None) -> None:
    sd.play(samples, samplerate=sample_rate, device=device)
    sd.wait()


def _trailing_silence_stop_condition(
    sample_rate: int, silence_s: float = 1.2, energy_threshold: float = 0.01
) -> Callable[[np.ndarray], bool]:
    """Naive, dependency-free energy-threshold trailing-silence detector: stops once
    the trailing `silence_s` seconds are all quiet, as long as speech was already
    captured (so it doesn't stop immediately on leading silence).
    """
    silence_samples = int(sample_rate * silence_s)

    def should_continue(audio: np.ndarray) -> bool:
        if len(audio) < silence_samples:
            return True
        tail_rms = float(np.sqrt(np.mean(audio[-silence_samples:] ** 2)))
        has_speech = float(np.sqrt(np.mean(audio**2))) > energy_threshold
        return not (has_speech and tail_rms < energy_threshold)

    return should_continue


def listen_utterance(config: "CandybotConfig") -> np.ndarray:
    """Records one utterance, bounded according to voice.trigger_mode.

    push_to_talk: waits for the configured key to be pressed, records while held.
    wake_word: waits for the wake word, then records until trailing silence or a
      max duration, so a non-responsive visitor doesn't hang the demo.
    """
    device = find_device(config.audio.input_device_name_hint, kind="input")

    if config.voice.trigger_mode == "push_to_talk":
        from candybot.voice.push_to_talk import is_pressed, wait_for_press

        key = config.voice.push_to_talk.key
        wait_for_press(key)
        return record_stream(
            should_continue=lambda audio: is_pressed(key),
            sample_rate=config.audio.sample_rate,
            device=device,
            max_duration_s=15.0,
        )

    if config.voice.trigger_mode == "wake_word":
        from candybot.voice.wakeword import wait_for_wake_word

        wait_for_wake_word(
            config.voice.wake_word.model, config.voice.wake_word.threshold, device, config.audio.sample_rate
        )
        return record_stream(
            should_continue=_trailing_silence_stop_condition(config.audio.sample_rate),
            sample_rate=config.audio.sample_rate,
            device=device,
            max_duration_s=6.0,
        )

    raise ValueError(f"Unknown voice.trigger_mode: {config.voice.trigger_mode!r}")
