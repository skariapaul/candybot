# Voice trigger modes

Candybot supports two interchangeable trigger modes, set via `voice.trigger_mode` in `configs/candybot.yaml` (`push_to_talk` or `wake_word`) -- or picked interactively at launch (`candybot/orchestrator/run.py`'s `_prompt_trigger_mode()`, same shape as the audio-profile prompt; `CANDYBOT_TRIGGER_MODE` env var skips the prompt for non-interactive/booth launches). Both converge on the same `candybot.voice.audio_io.listen_utterance(mode)` call, so `dialogue.py` doesn't need to know which mode is active — only how the listening window opens differs; both now stop the same way, on trailing silence (see below).

## Push-to-talk (default, recommended for a loud floor)

Implemented via `candybot.voice.push_to_talk`, which reads a single keypress **directly from the terminal candybot is running in** (raw/cbreak mode on stdin), not a global OS-level listener.

**Why not a global listener:** the original implementation used `pynput` for true system-wide hold-to-talk (press and hold, release to stop). That was found not to work on this dev machine's Wayland desktop session — Wayland blocks apps from globally snooping keyboard input by design, so `pynput`'s listener ran without error but never received a single keypress. Switching to `evdev`/`uinput` for a real global listener would need the user in the `input` group (another sudo step) with no guarantee of full reliability either. Reading from the app's own terminal sidesteps this entirely and is the right model anyway for a kiosk-style app that owns the terminal it launches in — a real physical USB booth button that enumerates as an HID keyboard will work correctly as long as that terminal window has focus, which is trivial to guarantee in a booth setup (fullscreen terminal).

**Behavior change this implies:** since a terminal can only reliably report "a key was pressed," not press/release timing, this is **press-to-start**, not true hold-to-talk. Pressing the key (default: `space`) starts recording; it then stops automatically on trailing silence or a 15s max, same mechanism wake-word mode uses below.

The trailing-silence detector (`candybot.voice.audio_io._trailing_silence_stop_condition`) is **adaptive**, not a fixed threshold -- it tracks the quietest ~100ms block observed so far in the current recording as a running noise-floor estimate, since a fixed global threshold doesn't generalize across mics/rooms (found live: a hardcoded value sat above one real setup's actual ambient noise floor, so trailing silence never registered).

Still the recommended mode whenever the environment is uncertain — deliberate manual trigger, immune to background noise/false triggers.

## Wake word (always-listening)

`candybot.voice.wakeword` uses `openwakeword` (fully local, ONNX, no API key). Ships with the interim pretrained community model (`hey_jarvis`) as a fallback, but candybot has its own custom phrase, **"Hey Zen"** -- see `training/wakeword/README.md` for the training pipeline (vendored from an actively-maintained fork of openWakeWord's own training tooling, since the official notebook pins a TensorFlow version with no Python 3.12 wheels). Once trained, drop the exported model in `candybot/models/wakeword/` and set `voice.wake_word.model: hey_zen` -- no code changes needed, the trigger-mode prompt's label updates automatically from the configured filename.

Utterance window is bounded by the same trailing-silence detector (~1.2s) with a hard cap (~6s) so a non-responsive visitor doesn't hang the demo. This mode reads raw audio directly and isn't affected by the Wayland/X11 issue above.

More hands-free and impressive, but carries real risk of false triggers or missed activations on a loud trade-show floor — switch to push-to-talk if the booth turns out too noisy. If false triggers happen with the trained "Hey Zen" model specifically, `voice.wake_word.threshold` (0.5 default) can be tuned without retraining -- see `training/wakeword/README.md`'s threshold-tuning section.
