# Voice trigger modes

Candybot supports two interchangeable trigger modes, set via `voice.trigger_mode` in `configs/candybot.yaml` (`push_to_talk` or `wake_word`). Both converge on the same `candybot.voice.audio_io.listen_utterance(mode)` call, so `dialogue.py` doesn't need to know which mode is active — only how the listening window opens and closes differs.

## Push-to-talk (default, recommended for a loud floor)

A button hold/release bounds the recording window. Implemented via `candybot.voice.push_to_talk` using `pynput` — a physical booth button hasn't been sourced yet, but most cheap USB arcade/macro buttons enumerate as an HID keyboard, so a single keycode listener (default: `space`) transparently covers both a keyboard stand-in today and the real button later, with no code change needed.

Immune to background noise and false triggers — the recommended mode whenever the environment is uncertain.

## Wake word (always-listening)

`candybot.voice.wakeword` uses `openwakeword` (fully local, ONNX, no API key) with an interim pretrained community model (`hey_jarvis`) as the wake phrase, until a custom "hey candybot" model is trained (flagged as a stretch goal — training one requires a small labeled audio dataset via a tool like `Piper-sample-generator`/`speechbrain`). Utterance window is bounded by Silero VAD trailing-silence (~1.2s) with a hard cap (~6s) so a non-responsive visitor doesn't hang the demo.

More hands-free and impressive, but carries real risk of false triggers or missed activations on a loud trade-show floor — switch to push-to-talk if the booth turns out too noisy.
