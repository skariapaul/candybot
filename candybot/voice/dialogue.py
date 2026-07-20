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

# Leading discourse markers/interjections people say before actually starting
# their sentence ("All right, my name is Paul") -- stripped repeatedly so
# stacked ones ("well, um, my name is...") all go, before name-pattern
# matching below even runs.
_LEADING_INTERJECTIONS = re.compile(
    r"^(?:um+|uh+|well|so|okay|ok|alright|all right|yeah|yep|hi|hello|hey)[,.\s]+", re.IGNORECASE
)

# Searched (not just prefix-matched) against the cleaned text, so a leading
# interjection the regex above didn't anticipate can't defeat name extraction
# the way it did before -- "All right, my name is Paul" only failed originally
# because "^my name is" required the phrase at position 0.
_NAME_PATTERNS = [
    re.compile(r"\bmy name is\s+(.+)", re.IGNORECASE),
    re.compile(r"\bmy name's\s+(.+)", re.IGNORECASE),
    re.compile(r"\bi am\s+(.+)", re.IGNORECASE),
    re.compile(r"\bi'm\s+(.+)", re.IGNORECASE),
    re.compile(r"\bthis is\s+(.+)", re.IGNORECASE),
    re.compile(r"\bcall me\s+(.+)", re.IGNORECASE),
    re.compile(r"\bit'?s\s+(.+)", re.IGNORECASE),
]

_YES_WORDS = {"yes", "yeah", "yep", "yup", "correct", "right", "sure"}
_NO_WORDS = {"no", "nope", "nah", "wrong", "incorrect"}

_CHOCOLATE_WORDS = {"chocolate", "choc", "chocolates"}
_CANDY_WORDS = {"candy", "candies", "sweet", "sweets", "gummy", "gummies"}


def _strip_fillers(text: str) -> str:
    cleaned = text.strip()

    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = _LEADING_INTERJECTIONS.sub("", cleaned).strip()

    for pattern in _NAME_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            cleaned = match.group(1)
            break

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
class CommandCaptureSession:
    """Captures an open-ended spoken command for smolVLA's language-conditioned
    picking (config.robot.action_mode == "smolvla"), replacing ItemChoiceSession's
    fixed chocolate/candy classification. There's no keyword set to classify
    against here -- the raw confirmed transcript *is* the language command
    passed straight to the policy (see candybot/robot/policy_runtime.py's
    run_command()), so unlike NameCaptureSession it isn't filler-stripped or
    title-cased -- a command should stay natural, as spoken.
    """

    listen: ListenFn
    transcribe: TranscribeFn
    speak: SpeakFn
    name: str
    max_attempts: int = 3
    default_command: str = "pick up an item and hand it to the visitor"

    def run(self) -> str:
        """Returns a confirmed command, or `default_command` if attempts are
        exhausted -- so a confused ASR pass never stalls the demo."""
        for attempt in range(1, self.max_attempts + 1):
            self.speak(
                f"What would you like me to pick up, {self.name}?"
                if attempt == 1
                else "Sorry, I didn't catch that -- what would you like me to pick up?"
            )
            result = self.transcribe(self.listen())
            if not result.is_confident:
                continue

            command = result.text.strip().strip(".!,")
            if not command:
                continue

            self.speak(f"Did you say: {command}? Yes or no?")
            confirmation = _classify(self.transcribe(self.listen()).text, _YES_WORDS, _NO_WORDS)

            if confirmation is None:
                self.speak(f"Sorry, was that a yes or a no -- did you say: {command}?")
                confirmation = _classify(self.transcribe(self.listen()).text, _YES_WORDS, _NO_WORDS)

            if confirmation:
                return command
            # "no" (or still ambiguous, treated as no) -- loop back and retry capture

        return self.default_command


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
