"""Top-level demo state machine.

IDLE -> GREET -> CAPTURE_NAME -> ASK_ITEM_CHOICE -> PICK_AND_HAND -> THANK_YOU -> RESET -> IDLE

Note: CONFIRM_NAME (from docs/ARCHITECTURE.md's conceptual flow) is handled as
a sub-loop *inside* CAPTURE_NAME's NameCaptureSession.run() rather than as a
separate top-level FSM state -- that's where the capture/confirm/retry logic
actually lives (candybot/voice/dialogue.py), unit-tested there in isolation.
The dashboard can get finer-grained status via DialogueEvents emitted during
that sub-loop, independent of this coarser FSM state.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from transitions.extensions.asyncio import AsyncMachine

from candybot.orchestrator.events import StateChangeEvent

logger = logging.getLogger(__name__)

STATES = ["IDLE", "GREET", "CAPTURE_NAME", "ASK_ITEM_CHOICE", "PICK_AND_HAND", "THANK_YOU", "RESET"]

TRANSITIONS = [
    {"trigger": "start", "source": "IDLE", "dest": "GREET"},
    {"trigger": "greeted", "source": "GREET", "dest": "CAPTURE_NAME"},
    {"trigger": "name_captured", "source": "CAPTURE_NAME", "dest": "ASK_ITEM_CHOICE"},
    {"trigger": "item_chosen", "source": "ASK_ITEM_CHOICE", "dest": "PICK_AND_HAND"},
    {"trigger": "handed_over", "source": "PICK_AND_HAND", "dest": "THANK_YOU"},
    {"trigger": "thanked", "source": "THANK_YOU", "dest": "RESET"},
    {"trigger": "reset_done", "source": "RESET", "dest": "IDLE"},
]

OnStateChange = Callable[[StateChangeEvent], Awaitable[None]]


class CandybotFSM:
    """Model driven by AsyncMachine. Trigger methods (start(), greeted(), ...)
    are added dynamically by the machine below -- call them as `await fsm.start()`.
    """

    def __init__(self, on_state_change: OnStateChange | None = None):
        self._on_state_change = on_state_change
        self.machine = AsyncMachine(
            model=self, states=STATES, transitions=TRANSITIONS, initial="IDLE", send_event=False
        )

    async def _notify(self, state: str) -> None:
        logger.info(f"FSM -> {state}")
        if self._on_state_change:
            await self._on_state_change(StateChangeEvent(state=state))

    # `transitions` wires these up by naming convention (on_enter_<STATE>).
    async def on_enter_IDLE(self) -> None:
        await self._notify("IDLE")

    async def on_enter_GREET(self) -> None:
        await self._notify("GREET")

    async def on_enter_CAPTURE_NAME(self) -> None:
        await self._notify("CAPTURE_NAME")

    async def on_enter_ASK_ITEM_CHOICE(self) -> None:
        await self._notify("ASK_ITEM_CHOICE")

    async def on_enter_PICK_AND_HAND(self) -> None:
        await self._notify("PICK_AND_HAND")

    async def on_enter_THANK_YOU(self) -> None:
        await self._notify("THANK_YOU")

    async def on_enter_RESET(self) -> None:
        await self._notify("RESET")
