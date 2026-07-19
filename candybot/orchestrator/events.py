"""Shared event types published by the orchestrator, consumed by the dashboard (and logs)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class StateChangeEvent:
    state: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class TranscriptEvent:
    speaker: str  # "visitor" | "candybot"
    text: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class DialogueEvent:
    """Fine-grained sub-status within a top-level FSM state -- e.g. CAPTURE_NAME's
    internal confirm/retry sub-step (see candybot/voice/dialogue.py). The FSM
    itself only tracks the coarser states in fsm.py; this fills the gap for
    dashboard display.
    """

    stage: str
    detail: str = ""
    timestamp: float = field(default_factory=time.time)
