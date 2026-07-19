"""Name-capture and item-choice dialogue logic.

Both sessions take injectable listen/transcribe/speak callables so the
retry/confirm logic is unit-testable on canned transcripts without any audio
hardware -- see tests/test_dialogue_fsm.py. Classification for both
confirmation (yes/no) and item choice (chocolate/candy) uses small closed
keyword sets rather than freeform transcription -- a much easier, more
noise-robust ASR target than an open-vocabulary name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from candybot.voice.asr import TranscriptionResult

ListenFn = Callable[[], np.ndarray]
TranscribeFn = Callable[[np.ndarray], TranscriptionResult]
SpeakFn = Callable[[str], None]

_FILLER_PATTERNS = [
    r"^my name is\s+",
    r"^i am\s+",
    r"^i'm\s+",
    r"^it's\s+",
    r"^its\s+",
    r"^this is\s+",
    r"^call me\s+",
]

_YES_WORDS = {"yes", "yeah", "yep", "yup", "correct", "right", "sure"}
_NO_WORDS = {"no", "nope", "nah", "wrong", "incorrect"}

_CHOCOLATE_WORDS = {"chocolate", "choc", "chocolates"}
_CANDY_WORDS = {"candy", "candies", "sweet", "sweets", "gummy", "gummies"}


def _strip_fillers(text: str) -> str:
    cleaned = text.strip()
    for pattern in _FILLER_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip().strip(".!,").title()


def _classify(text: str, positive: set[str], negative: set[str]) -> bool | None:
    """Returns True if `text` matches only `positive` words, False if only
    `negative`, or None if ambiguous/no match."""
    words = set(re.findall(r"[a-z']+", text.lower()))
    hit_pos = bool(words & positive)
    hit_neg = bool(words & negative)
    if hit_pos and not hit_neg:
        return True
    if hit_neg and not hit_pos:
        return False
    return None


@dataclass
class NameCaptureSession:
    listen: ListenFn
    transcribe: TranscribeFn
    speak: SpeakFn
    max_attempts: int = 3

    def run(self) -> str:
        """Returns a confirmed name, or 'friend' if attempts are exhausted."""
        for attempt in range(1, self.max_attempts + 1):
            self.speak(
                "Hi! What's your name?"
                if attempt == 1
                else "Sorry, I didn't catch that -- could you say your name again?"
            )
            result = self.transcribe(self.listen())
            if not result.is_confident:
                continue

            name = _strip_fillers(result.text)
            if not name:
                continue

            self.speak(f"Did you say {name}? Yes or no?")
            confirmation = _classify(self.transcribe(self.listen()).text, _YES_WORDS, _NO_WORDS)

            if confirmation is None:
                self.speak(f"Sorry, was that a yes or a no -- did you say {name}?")
                confirmation = _classify(self.transcribe(self.listen()).text, _YES_WORDS, _NO_WORDS)

            if confirmation:
                return name
            # "no" (or still ambiguous, treated as no) -- loop back and retry capture

        return "friend"


@dataclass
class ItemChoiceSession:
    listen: ListenFn
    transcribe: TranscribeFn
    speak: SpeakFn
    name: str
    max_attempts: int = 2
    default_item: str = "candy"

    def run(self) -> str:
        """Returns 'chocolate' or 'candy' -- never fails to return one of the two."""
        for attempt in range(1, self.max_attempts + 1):
            self.speak(
                f"Would you like chocolate or candy, {self.name}?" if attempt == 1 else "Sorry, chocolate or candy?"
            )
            choice = _classify(self.transcribe(self.listen()).text, _CHOCOLATE_WORDS, _CANDY_WORDS)
            if choice is True:
                return "chocolate"
            if choice is False:
                return "candy"

        return self.default_item
