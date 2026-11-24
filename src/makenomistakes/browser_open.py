"""`mnm open`: a web UI page in this machine's browser. Under WSL, Windows' default
browser opens it (`wslview` from wslu where installed, else `explorer.exe`, which WSL runs on
the Windows side; Windows' localhost reaches the web UI through WSL's localhost forwarding).
On a native box `xdg-open`, only with a desktop session: without one (a box reached over SSH)
it would start a text browser in the terminal. When nothing opens it, the caller says the
address instead.

An opener runs apart from mnm's terminal (`Detached`): xdg-open's generic handler waits for
the browser it starts, so a capturing run would hold `mnm open` 15 s and then say "nothing here
opens a page" of a browser that had opened."""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

WSL_OPENERS = ("wslview", "explorer.exe")
NATIVE_OPENERS = ("xdg-open",)
DESKTOP = ("DISPLAY", "WAYLAND_DISPLAY")  # a desktop session's environment
WAIT_SECONDS = 5.0  # an opener still running by then has handed the page to a browser


class Launcher(Protocol):
    def launch(self, argv: list[str]) -> int | None:
        """The opener's exit code, or None while it still runs after the wait. OSError when it
        cannot run at all (not found; WSL without Windows interop: exec format error)."""
        ...


@dataclass(frozen=True)
class Detached:
    """Runs an opener with no input, its output dropped, in a session of its own, so a browser
    it starts in the foreground neither holds the terminal nor keeps `mnm open` waiting; one
    still running after `wait_seconds` is left to run, reaped when it ends."""

    wait_seconds: float = WAIT_SECONDS

    def launch(self, argv: list[str]) -> int | None:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=self.wait_seconds)
        except subprocess.TimeoutExpired:
            threading.Thread(target=process.wait, daemon=True).start()
            return None


def openers(wsl: bool, environ: Mapping[str, str]) -> tuple[str, ...]:
    """What may open a page here, in the order to try them."""
    if wsl:
        return WSL_OPENERS
    return NATIVE_OPENERS if any(environ.get(name) for name in DESKTOP) else ()


def open_page(url: str, launcher: Launcher, wsl: bool, environ: Mapping[str, str]) -> str | None:
    """The opener that took `url`, else None. One still running after the wait handed the page
    to a browser; `explorer.exe` answers 1 when it opened the page as well as when it did not,
    so any answer it gives counts; one that cannot run is passed over for the next."""
    for opener in openers(wsl, environ):
        try:
            code = launcher.launch([opener, url])
        except OSError:
            continue
        if code is None or code == 0 or opener == "explorer.exe":
            return opener
    return None
