"""FastAPI dashboard: WebSocket for live state/telemetry/transcript, an MJPEG
camera route, and the static frontend. Runs in the same process/event loop as
the orchestrator (candybot/orchestrator/run.py's main()) so the WebSocket
reflects real events as they happen, and the camera route reads frames the
orchestrator publishes (candybot/dashboard/state.py) rather than opening the
camera device itself -- a second concurrent VideoCapture on the same UVC
device isn't reliable, see so101_controller.py's docstring.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from candybot.dashboard import state, telemetry

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

_TELEMETRY_INTERVAL_S = 1.0
_MJPEG_INTERVAL_S = 0.15


async def _telemetry_loop() -> None:
    while True:
        try:
            await state.publish("telemetry", telemetry.full_report())
        except Exception:
            logger.exception("Telemetry loop error")
        await asyncio.sleep(_TELEMETRY_INTERVAL_S)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    task = asyncio.create_task(_telemetry_loop())
    yield
    task.cancel()


def create_app() -> FastAPI:
    app = FastAPI(title="candybot dashboard", lifespan=_lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True}

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = state.subscribe()
        try:
            await websocket.send_text(json.dumps({"type": "snapshot", "data": state.snapshot()}))
            while True:
                message = await queue.get()
                await websocket.send_text(message)
        except WebSocketDisconnect:
            pass
        finally:
            state.unsubscribe(queue)

    @app.get("/camera")
    async def camera_feed() -> StreamingResponse:
        async def frame_generator():
            while True:
                frame = await state.get_latest_frame()
                if frame is not None:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                await asyncio.sleep(_MJPEG_INTERVAL_S)

        return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

    # Mounted last so /health, /ws, /camera (registered above) take priority.
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    print("Standalone dashboard (no orchestrator running) -- state will stay IDLE, no camera feed.")
    uvicorn.run(create_app(), host="0.0.0.0", port=8080)
