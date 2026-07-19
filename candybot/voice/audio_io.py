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


def _resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or len(samples) == 0:
        return samples
    target_len = int(round(len(samples) * target_sr / orig_sr))
    orig_indices = np.arange(len(samples))
    target_indices = np.linspace(0, len(samples) - 1, num=target_len)
    return np.interp(target_indices, orig_indices, samples).astype(np.float32)


def play_audio(samples: np.ndarray, sample_rate: int, device: int | None = None) -> None:
    """Resamples to the output device's native rate before playing -- PortAudio's
    raw ALSA devices often reject a stream opened at a rate they don't natively
    support (e.g. this laptop's USB headset only accepts 16kHz, but Piper TTS
    outputs 22050Hz), raising PortAudioError('Invalid sample rate') otherwise.
    """
    device_info = sd.query_devices(device, kind="output") if device is not None else sd.query_devices(kind="output")
    target_sr = int(device_info["default_samplerate"])
    if sample_rate != target_sr:
        samples = _resample(samples, sample_rate, target_sr)
        sample_rate = target_sr

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


def wait_for_trigger(config: "CandybotConfig", device: int | None = None) -> None:
    """Blocks until the configured trigger fires (push-to-talk keypress or wake
    word), without recording anything -- used both to open listen_utterance()'s
    recording window and, in the orchestrator, to gate starting a new visitor's
    greeting so it doesn't loop straight back into an empty booth.
    """
    device = device if device is not None else find_device(config.audio.input_device_name_hint, kind="input")

    if config.voice.trigger_mode == "push_to_talk":
        from candybot.voice.push_to_talk import wait_for_press

        wait_for_press(config.voice.push_to_talk.key)
        return

    if config.voice.trigger_mode == "wake_word":
        from candybot.voice.wakeword import wait_for_wake_word

        wait_for_wake_word(
            config.voice.wake_word.model, config.voice.wake_word.threshold, device, config.audio.sample_rate
        )
        return

    raise ValueError(f"Unknown voice.trigger_mode: {config.voice.trigger_mode!r}")


def listen_utterance(config: "CandybotConfig") -> np.ndarray:
    """Records one utterance: waits for the trigger, then records until trailing
    silence or a max duration, so a non-responsive visitor doesn't hang the demo.
    push_to_talk is press-to-start rather than true hold-to-talk -- see
    push_to_talk.py's docstring for why (terminal keypress detection, not a
    global OS listener).
    """
    device = find_device(config.audio.input_device_name_hint, kind="input")
    wait_for_trigger(config, device=device)

    max_duration_s = 15.0 if config.voice.trigger_mode == "push_to_talk" else 6.0
    return record_stream(
        should_continue=_trailing_silence_stop_condition(config.audio.sample_rate),
        sample_rate=config.audio.sample_rate,
        device=device,
        max_duration_s=max_duration_s,
    )
