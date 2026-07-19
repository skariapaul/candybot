"""Entrypoint: wires voice + robot into the orchestrator FSM and runs the live
demo loop.

Blocking calls (mic recording, ASR, TTS playback, arm motion) are pushed onto
worker threads via asyncio.to_thread() so the event loop stays free -- once
candybot.dashboard (task 8) runs a WebSocket server in this same process, it
needs to keep broadcasting state while a multi-second listen/speak/motion call
is in flight, not freeze for the duration.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from candybot.config import CandybotConfig, load_config
from candybot.hardware_probe import get_device
from candybot.orchestrator.events import StateChangeEvent
from candybot.orchestrator.fsm import CandybotFSM, OnStateChange
from candybot.robot import scripted_actions
from candybot.robot.so101_controller import SO101Controller
from candybot.voice import tts
from candybot.voice.asr import TranscriptionResult
from candybot.voice.asr import transcribe as asr_transcribe
from candybot.voice.audio_io import find_device, listen_utterance
from candybot.voice.dialogue import ItemChoiceSession, NameCaptureSession

logger = logging.getLogger(__name__)


async def _default_on_state_change(event: StateChangeEvent) -> None:
    logger.info(f"[state] {event.state}")


async def run_demo_loop(
    config: CandybotConfig | None = None,
    on_state_change: OnStateChange | None = None,
) -> None:
    config = config or load_config()
    logger.info(f"Device for policy inference: {get_device()}")

    controller = SO101Controller(config)
    controller.connect(calibrate=True)
    output_device = find_device(config.audio.output_device_name_hint, kind="output")

    def speak(text: str) -> None:
        logger.info(f"candybot: {text}")
        tts.speak(text, config.voice.tts.voice_model, output_device=output_device)

    def listen() -> np.ndarray:
        return listen_utterance(config)

    def transcribe(audio: np.ndarray) -> TranscriptionResult:
        result = asr_transcribe(
            audio, config.audio.sample_rate, config.voice.asr.model_size, config.voice.asr.vad_filter
        )
        logger.info(f"visitor: {result.text!r} (confident={result.is_confident})")
        return result

    fsm = CandybotFSM(on_state_change=on_state_change or _default_on_state_change)

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
        controller.disconnect()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_demo_loop())


if __name__ == "__main__":
    main()
