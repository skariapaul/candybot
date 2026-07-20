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
import os

import cv2
import numpy as np

from candybot.config import CandybotConfig, load_config
from candybot.dashboard import state as dashboard_state
from candybot.hardware_probe import get_device
from candybot.orchestrator.events import SpeechEvent, StateChangeEvent, TranscriptEvent
from candybot.orchestrator.fsm import CandybotFSM
from candybot.robot import policy_runtime, scripted_actions
from candybot.robot.so101_controller import SO101Controller
from candybot.voice import tts
from candybot.voice.asr import TranscriptionResult
from candybot.voice.asr import transcribe as asr_transcribe
from candybot.voice.audio_io import find_device, listen_utterance, play_audio, wait_for_trigger
from candybot.voice.dialogue import CommandCaptureSession, ItemChoiceSession, NameCaptureSession

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

        samples, sample_rate = tts.synthesize(text, config.voice.tts.voice_model)
        envelope = tts.compute_envelope(samples, sample_rate)
        duration_s = len(samples) / sample_rate
        asyncio.run_coroutine_threadsafe(
            dashboard_state.publish("speech", SpeechEvent(envelope=envelope, duration_s=duration_s)), loop
        )

        play_audio(samples, sample_rate, device=output_device)

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
            logger.info("Waiting for next visitor (trigger)...")
            await asyncio.to_thread(wait_for_trigger, config)

            await fsm.start()  # IDLE -> GREET
            await asyncio.to_thread(speak, "Hi there! Let's get you a treat.")

            await fsm.greeted()  # GREET -> CAPTURE_NAME
            name_session = NameCaptureSession(
                listen=listen, transcribe=transcribe, speak=speak, max_attempts=config.dialogue.max_name_attempts
            )
            name = await asyncio.to_thread(name_session.run)

            await fsm.name_captured()  # -> ASK_ITEM_CHOICE
            if config.robot.action_mode == "smolvla":
                command_session = CommandCaptureSession(
                    listen=listen,
                    transcribe=transcribe,
                    speak=speak,
                    name=name,
                    max_attempts=config.dialogue.max_command_attempts,
                    default_command=config.dialogue.command_default,
                )
                command = await asyncio.to_thread(command_session.run)

                await fsm.item_chosen()  # -> PICK_AND_HAND
                await asyncio.to_thread(policy_runtime.run_command, controller, config, command)
            else:
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
                action_fn = (
                    policy_runtime.ACTIONS[item] if config.robot.action_mode == "policy" else scripted_actions.ACTIONS[item]
                )
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


def _prompt_audio_profile(config: CandybotConfig) -> str:
    """Interactively asks which audio profile to use, unless CANDYBOT_AUDIO_PROFILE
    is already set in the environment (e.g. a non-interactive/booth launch that
    shouldn't block on stdin).
    """
    if os.environ.get("CANDYBOT_AUDIO_PROFILE"):
        return config.audio.profile

    names = list(config.audio.profiles.keys())
    default_index = names.index(config.audio.profile) + 1

    print("\nSelect audio device:")
    for i, name in enumerate(names, start=1):
        label = config.audio.profiles[name].label or name
        marker = " (default)" if name == config.audio.profile else ""
        print(f"  {i}) {label}{marker}")

    choice = input(f"Choice [{default_index}]: ").strip()
    if not choice:
        return config.audio.profile

    try:
        index = int(choice) - 1
        if 0 <= index < len(names):
            return names[index]
    except ValueError:
        pass

    print(f"Invalid choice -- using default: {config.audio.profile}")
    return config.audio.profile


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    config.audio.profile = _prompt_audio_profile(config)
    logger.info(f"Using audio profile: {config.audio.profile}")
    asyncio.run(_run_combined(config))


if __name__ == "__main__":
    main()
