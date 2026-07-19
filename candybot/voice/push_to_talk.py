"""Push-to-talk trigger via a single terminal keypress.

candybot runs as a terminal-launched kiosk app, so this reads directly from
the process's own stdin (raw/cbreak mode) rather than a global OS-level
listener. Confirmed empirically on this dev machine: a global listener
(originally implemented with pynput) needs X11, and this desktop session is
Wayland, which blocks apps from globally snooping keyboard input by design --
pynput's Listener never received a single event here despite running
correctly with no errors. Reading from the app's own terminal sidesteps the
whole X11/Wayland/evdev-permissions problem entirely, and is the right model
anyway for an app that owns the terminal it's running in.

A terminal can only reliably report "a key was pressed", not press/release
timing, so this is press-to-start rather than true hold-to-talk -- recording
then stops on trailing silence via audio_io's VAD stop condition, the same
mechanism wake-word mode already uses. See docs/VOICE_MODES.md.
"""

from __future__ import annotations

import sys
import termios
import tty

_KEY_CHARS = {
    "space": " ",
    "enter": "\r",
}


def wait_for_press(key_name: str) -> None:
    """Blocks until `key_name` is pressed in this terminal (no ENTER needed)."""
    target_char = _KEY_CHARS.get(key_name, key_name[:1] if key_name else " ")
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == target_char:
                return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
