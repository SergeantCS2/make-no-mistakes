"""ComfyUI gives the card back after a quiet spell, whoever ran it: the web, the CLI,
a film's clips or a chat over MCP. Every run's end leaves a stamp beside the ledger (never
mounted into the container); the router's idle sweep, every 30 s, frees ComfyUI's VRAM once
when the last run ended ten minutes ago and nothing is queued. Before this, only a run started
from the web page armed the timer, so a run from anywhere else kept the card held."""

from __future__ import annotations

import time
from pathlib import Path

from makenomistakes.comfy.engine import FREED_STAMP, ComfyEngine, State
from makenomistakes.comfy.image import ComfyBusyError

IDLE_SECONDS = 600.0  # the same ten minutes as a model's idle unload
LAST_RUN = "last-run"  # touched when a run ends
FREED = FREED_STAMP  # touched when its card was freed (or the sweep found nothing to free)


def ran(engine: ComfyEngine) -> None:
    """A run just ended (finished, failed or interrupted)."""
    _touch(engine.base / LAST_RUN)


def freed(engine: ComfyEngine) -> None:
    """Its card was freed by hand (the Status page's Free VRAM): the sweep has nothing to do
    until the next run ends, and the panel says when it was freed."""
    _touch(engine.base / FREED)


def release_if_idle(engine: ComfyEngine, now: float | None = None) -> bool:
    """Free ComfyUI's VRAM once per quiet spell. True when it was asked to; a box where no
    run ended since the last free costs two file reads."""
    last = _mtime(engine.base / LAST_RUN)
    if last is None or last <= (_mtime(engine.base / FREED) or 0.0):
        return False
    if (now if now is not None else time.time()) - last < IDLE_SECONDS:
        return False
    state = engine.state()
    if state in (State.STOPPED, State.NOT_INSTALLED):  # stopped meanwhile: nothing is held
        _touch(engine.base / FREED)
        return False
    address = engine.address(state)  # None while Docker does not say: the next sweep asks
    if address is None or engine.queue_busy(address) is not False:
        return False  # a run going (its end stamps again), or no answer: unknown, never idle
    try:
        freed = engine.release_vram()
    except ComfyBusyError:  # a run queued since: its end stamps again
        return False
    if freed:
        _touch(engine.base / FREED)
    return freed


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None
