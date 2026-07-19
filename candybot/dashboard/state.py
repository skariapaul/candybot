"""In-process shared state + pub/sub for the dashboard.

The orchestrator (candybot/orchestrator/run.py) and the camera publisher
(candybot/dashboard/server.py) push into this module; the WebSocket route
broadcasts to connected browser clients. Single-process, in-memory -- no
external broker needed for a single-laptop demo.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, is_dataclass
from typing import Any

_subscribers: set[asyncio.Queue] = set()

_latest_state: dict[str, Any] = {
    "fsm_state": "IDLE",
    "transcript": [],  # list of {speaker, text, timestamp}
    "dialogue": {"stage": "", "detail": ""},
    "telemetry": {},
}

_MAX_TRANSCRIPT = 50


def _to_jsonable(event: Any) -> dict:
    return asdict(event) if is_dataclass(event) else dict(event)


async def publish(event_type: str, event: Any) -> None:
    """Updates the latest-state snapshot and broadcasts to all connected clients."""
    payload = _to_jsonable(event)

    if event_type == "state_change":
        _latest_state["fsm_state"] = payload["state"]
    elif event_type == "transcript":
        _latest_state["transcript"].append(payload)
        _latest_state["transcript"] = _latest_state["transcript"][-_MAX_TRANSCRIPT:]
    elif event_type == "dialogue":
        _latest_state["dialogue"] = payload
    elif event_type == "telemetry":
        _latest_state["telemetry"] = payload

    message = json.dumps({"type": event_type, "data": payload})
    for queue in list(_subscribers):
        queue.put_nowait(message)


def subscribe() -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue()
    _subscribers.add(queue)
    return queue


def unsubscribe(queue: asyncio.Queue) -> None:
    _subscribers.discard(queue)


def snapshot() -> dict[str, Any]:
    """Full current state, sent to a client immediately on connect."""
    return dict(_latest_state)


# --- Camera frame (JPEG bytes), separate from the event stream since the
# MJPEG route (GET /camera) pulls it directly rather than going through the
# WebSocket broadcast. ---

_latest_frame: bytes | None = None
_frame_lock = asyncio.Lock()


async def set_latest_frame(jpeg_bytes: bytes) -> None:
    global _latest_frame
    async with _frame_lock:
        _latest_frame = jpeg_bytes


async def get_latest_frame() -> bytes | None:
    async with _frame_lock:
        return _latest_frame
