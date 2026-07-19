"""Unit tests for candybot.orchestrator.fsm -- exercises state transitions
with a mocked on_state_change callback, no voice/robot hardware.
"""

from __future__ import annotations

import pytest

from candybot.orchestrator.events import StateChangeEvent
from candybot.orchestrator.fsm import CandybotFSM


@pytest.mark.asyncio
async def test_full_cycle_returns_to_idle():
    seen: list[str] = []

    async def on_change(event: StateChangeEvent) -> None:
        seen.append(event.state)

    fsm = CandybotFSM(on_state_change=on_change)
    assert fsm.state == "IDLE"

    await fsm.start()
    assert fsm.state == "GREET"
    await fsm.greeted()
    assert fsm.state == "CAPTURE_NAME"
    await fsm.name_captured()
    assert fsm.state == "ASK_ITEM_CHOICE"
    await fsm.item_chosen()
    assert fsm.state == "PICK_AND_HAND"
    await fsm.handed_over()
    assert fsm.state == "THANK_YOU"
    await fsm.thanked()
    assert fsm.state == "RESET"
    await fsm.reset_done()
    assert fsm.state == "IDLE"

    assert seen == [
        "GREET",
        "CAPTURE_NAME",
        "ASK_ITEM_CHOICE",
        "PICK_AND_HAND",
        "THANK_YOU",
        "RESET",
        "IDLE",
    ]


@pytest.mark.asyncio
async def test_invalid_transition_raises():
    fsm = CandybotFSM()
    with pytest.raises(Exception):  # transitions raises MachineError for an undefined trigger/state pair
        await fsm.item_chosen()  # can't jump straight to PICK_AND_HAND from IDLE


@pytest.mark.asyncio
async def test_works_without_callback():
    fsm = CandybotFSM()  # on_state_change=None must not raise
    await fsm.start()
    assert fsm.state == "GREET"
