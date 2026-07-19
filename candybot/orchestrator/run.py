"""Entrypoint: wires voice + robot + dashboard into the orchestrator FSM and
runs the live demo loop.

Blocking calls (mic recording, ASR, TTS playback, arm motion) are pushed onto
worker threads via asyncio.to_thread() so the event loop stays free for the
dashboard's WebSocket and camera stream, which both need to keep running while
a multi-second listen/speak/motion call is in flight.

Transcript events originate inside those worker threads, so publishing them
back onto the dashboard's asyncio.Queue (not thread-safe across threads) goes
through asyncio.run_coroutine_threadsafe() rather than a direct await.
"""

from __future__ import annotations

import asyncio
import logging

import cv2
import numpy as np

from candybot.config import CandybotConfig, load_config
from candybot.dashboard import state as dashboard_state
from candybot.hardware_probe import get_device
from candybot.orchestrator.events import StateChangeEvent, TranscriptEvent
from candybot.orchestrator.fsm import CandybotFSM
from candybot.robot import scripted_actions
from candybot.robot.so101_controller import SO101Controller
from candybot.voice import tts
from candybot.voice.asr import TranscriptionResult
from candybot.voice.asr import transcribe as asr_transcribe
from candybot.voice.audio_io import find_device, listen_utterance
from candybot.voice.dialogue import ItemChoiceSession, NameCaptureSession

logger = logging.getLogger(__name__)

_CAMERA_PUBLISH_INTERVAL_S = 0.2


async def _dashboard_on_state_change(event: StateChangeEvent) -> None:
    await dashboard_state.publish("state_change", event)


async def _camera_publisher_loop(controller: SO101Controller) -> None:
    """Continuously publishes wrist-camera frames to the dashboard, independent
    of whether a pick action is currently running -- see so101_controller.py's
    internal hardware lock for why this is safe to run concurrently.
    """
    while True:
        try:
            obs = await asyncio.to_thread(controller.get_observation)
            frame_rgb = obs.get("wrist")
            if frame_rgb is not None:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                ok, jpeg = cv2.imencode(".jpg", frame_bgr)
                if ok:
                    await dashboard_state.set_latest_frame(jpeg.tobytes())
        except Exception:
            logger.exception("Camera publisher loop error")
        await asyncio.sleep(_CAMERA_PUBLISH_INTERVAL_S)


async def run_demo_loop(config: CandybotConfig | None = None) -> None:
    config = config or load_config()
    logger.info(f"Device for policy inference: {get_device()}")

    controller = SO101Controller(config)
    controller.connect(calibrate=True)
    output_device = find_device(config.audio.output_device_name_hint, kind="output")

    loop = asyncio.get_running_loop()

    def publish_transcript(speaker: str, text: str) -> None:
        if not text.strip():
            return
        asyncio.run_coroutine_threadsafe(
            dashboard_state.publish("transcript", TranscriptEvent(speaker=speaker, text=text)), loop
        )

    def speak(text: str) -> None:
        logger.info(f"candybot: {text}")
        publish_transcript("candybot", text)
        tts.speak(text, config.voice.tts.voice_model, output_device=output_device)

    def listen() -> np.ndarray:
        return listen_utterance(config)

    def transcribe(audio: np.ndarray) -> TranscriptionResult:
        result = asr_transcribe(
            audio, config.audio.sample_rate, config.voice.asr.model_size, config.voice.asr.vad_filter
        )
        logger.info(f"visitor: {result.text!r} (confident={result.is_confident})")
        publish_transcript("visitor", result.text)
        return result

    fsm = CandybotFSM(on_state_change=_dashboard_on_state_change)
    camera_task = asyncio.create_task(_camera_publisher_loop(controller))

    try:
        while True:
            await fsm.start()  # IDLE -> GREET
            await asyncio.to_thread(speak, "Hi there! Let's get you a treat.")

            await fsm.greeted()  # GREET -> CAPTURE_NAME
            name_session = NameCaptureSession(
                listen=listen, transcribe=transcribe, speak=speak, max_attempts=config.dialogue.max_name_attempts
            )
            name = await asyncio.to_thread(name_session.run)

            await fsm.name_captured()  # -> ASK_ITEM_CHOICE
            item_session = ItemChoiceSession(
                listen=listen,
                transcribe=transcribe,
                speak=speak,
                name=name,
                max_attempts=config.dialogue.max_item_attempts,
                default_item=config.dialogue.item_choice_default,
            )
            item = await asyncio.to_thread(item_session.run)

            await fsm.item_chosen()  # -> PICK_AND_HAND
            action_fn = scripted_actions.ACTIONS[item]
            await asyncio.to_thread(action_fn, controller, config)

            await fsm.handed_over()  # -> THANK_YOU
            await asyncio.to_thread(speak, f"Here you go, {name} -- thanks for stopping by!")

            await fsm.thanked()  # -> RESET
            await fsm.reset_done()  # -> IDLE
    finally:
        camera_task.cancel()
        controller.disconnect()


async def _run_combined(config: CandybotConfig) -> None:
    """Runs the dashboard's uvicorn server and the demo loop in the same event
    loop, so the WebSocket sees orchestrator events as they happen.
    """
    import uvicorn

    from candybot.dashboard.server import create_app

    app = create_app()
    server = uvicorn.Server(uvicorn.Config(app, host=config.dashboard.host, port=config.dashboard.port, log_level="warning"))

    logger.info(f"Dashboard on http://{config.dashboard.host}:{config.dashboard.port}")
    await asyncio.gather(run_demo_loop(config), server.serve())


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    asyncio.run(_run_combined(config))


if __name__ == "__main__":
    main()
