"""Push-to-talk trigger via a keyboard keycode (pynput).

A physical booth button hasn't been sourced yet -- most cheap USB
arcade/macro buttons enumerate as an HID keyboard, so this one listener
transparently covers both a keyboard stand-in today and the real button
later, with no code change needed. See docs/VOICE_MODES.md.
"""

from __future__ import annotations

import threading
import time

from pynput import keyboard

_pressed_keys: set[str] = set()
_lock = threading.Lock()
_listener: keyboard.Listener | None = None


def _key_name(key: keyboard.Key | keyboard.KeyCode) -> str:
    if isinstance(key, keyboard.KeyCode):
        return key.char or ""
    return key.name


def _on_press(key: keyboard.Key | keyboard.KeyCode) -> None:
    with _lock:
        _pressed_keys.add(_key_name(key))


def _on_release(key: keyboard.Key | keyboard.KeyCode) -> None:
    with _lock:
        _pressed_keys.discard(_key_name(key))


def _ensure_listener() -> None:
    global _listener
    if _listener is None:
        _listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
        _listener.start()


def is_pressed(key_name: str) -> bool:
    _ensure_listener()
    with _lock:
        return key_name in _pressed_keys


def wait_for_press(key_name: str, poll_interval_s: float = 0.02) -> None:
    _ensure_listener()
    while not is_pressed(key_name):
        time.sleep(poll_interval_s)
