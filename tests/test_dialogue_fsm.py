"""Unit tests for candybot.voice.dialogue -- pure logic, no audio hardware.
Feeds canned transcripts via fake listen/transcribe/speak callables.
"""

from __future__ import annotations

import numpy as np

from candybot.voice.asr import TranscriptionResult
from candybot.voice.dialogue import CommandCaptureSession, ItemChoiceSession, NameCaptureSession


def _confident(text: str) -> TranscriptionResult:
    return TranscriptionResult(text=text, avg_logprob=-0.3, no_speech_prob=0.05)


def _unconfident() -> TranscriptionResult:
    return TranscriptionResult(text="", avg_logprob=-999.0, no_speech_prob=1.0)


class ScriptedVoice:
    """Feeds a scripted sequence of transcription results, one per listen() call."""

    def __init__(self, results: list[TranscriptionResult]):
        self._results = list(results)
        self.spoken: list[str] = []

    def listen(self) -> np.ndarray:
        return np.zeros(1, dtype="float32")

    def transcribe(self, _audio: np.ndarray) -> TranscriptionResult:
        return self._results.pop(0)

    def speak(self, text: str) -> None:
        self.spoken.append(text)


def test_name_capture_happy_path():
    voice = ScriptedVoice([_confident("my name is Alex"), _confident("yes")])
    session = NameCaptureSession(listen=voice.listen, transcribe=voice.transcribe, speak=voice.speak)
    assert session.run() == "Alex"
    assert len(voice.spoken) == 2


def test_name_capture_strips_filler_and_titlecases():
    voice = ScriptedVoice([_confident("it's paul"), _confident("yeah")])
    session = NameCaptureSession(listen=voice.listen, transcribe=voice.transcribe, speak=voice.speak)
    assert session.run() == "Paul"


def test_name_capture_retries_on_no_then_succeeds():
    voice = ScriptedVoice(
        [
            _confident("Alx"),
            _confident("no"),  # first attempt, rejected
            _confident("Alex"),
            _confident("yes"),  # second attempt, confirmed
        ]
    )
    session = NameCaptureSession(listen=voice.listen, transcribe=voice.transcribe, speak=voice.speak, max_attempts=3)
    assert session.run() == "Alex"


def test_name_capture_gives_up_after_max_attempts():
    voice = ScriptedVoice([_unconfident(), _unconfident(), _unconfident()])
    session = NameCaptureSession(listen=voice.listen, transcribe=voice.transcribe, speak=voice.speak, max_attempts=3)
    assert session.run() == "friend"


def test_name_capture_ambiguous_confirmation_reasked_then_treated_as_no():
    voice = ScriptedVoice(
        [
            _confident("Sam"),
            _confident("maybe"),  # ambiguous confirm
            _confident("huh"),  # re-ask, still ambiguous -> treated as "no"
            _confident("Sam"),
            _confident("yes"),  # retry succeeds
        ]
    )
    session = NameCaptureSession(listen=voice.listen, transcribe=voice.transcribe, speak=voice.speak, max_attempts=3)
    assert session.run() == "Sam"


def test_item_choice_chocolate():
    voice = ScriptedVoice([_confident("chocolate please")])
    session = ItemChoiceSession(listen=voice.listen, transcribe=voice.transcribe, speak=voice.speak, name="Alex")
    assert session.run() == "chocolate"


def test_item_choice_candy():
    voice = ScriptedVoice([_confident("candy")])
    session = ItemChoiceSession(listen=voice.listen, transcribe=voice.transcribe, speak=voice.speak, name="Alex")
    assert session.run() == "candy"


def test_item_choice_defaults_after_max_attempts():
    voice = ScriptedVoice([_confident("umm"), _confident("not sure")])
    session = ItemChoiceSession(
        listen=voice.listen,
        transcribe=voice.transcribe,
        speak=voice.speak,
        name="Alex",
        max_attempts=2,
        default_item="candy",
    )
    assert session.run() == "candy"


def test_command_capture_happy_path():
    voice = ScriptedVoice([_confident("pick up the gold cup"), _confident("yes")])
    session = CommandCaptureSession(listen=voice.listen, transcribe=voice.transcribe, speak=voice.speak, name="Alex")
    assert session.run() == "pick up the gold cup"


def test_command_capture_not_title_cased_or_filler_stripped():
    # Unlike NameCaptureSession, commands stay natural/as-spoken -- no title-casing,
    # no "my name is"-style filler stripping (which would mangle a real command).
    voice = ScriptedVoice([_confident("it's the white one on the left"), _confident("yeah")])
    session = CommandCaptureSession(listen=voice.listen, transcribe=voice.transcribe, speak=voice.speak, name="Alex")
    assert session.run() == "it's the white one on the left"


def test_command_capture_retries_on_no_then_succeeds():
    voice = ScriptedVoice(
        [
            _confident("pick up the wrong one"),
            _confident("no"),
            _confident("pick up the gold cup"),
            _confident("yes"),
        ]
    )
    session = CommandCaptureSession(
        listen=voice.listen, transcribe=voice.transcribe, speak=voice.speak, name="Alex", max_attempts=3
    )
    assert session.run() == "pick up the gold cup"


def test_command_capture_defaults_after_max_attempts():
    voice = ScriptedVoice([_unconfident(), _unconfident()])
    session = CommandCaptureSession(
        listen=voice.listen,
        transcribe=voice.transcribe,
        speak=voice.speak,
        name="Alex",
        max_attempts=2,
        default_command="pick up something and hand it over",
    )
    assert session.run() == "pick up something and hand it over"
