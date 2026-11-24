"""What tells a reading of the card it was not one instance's own: something else
started or stopped on the box while it was taken, in any process (the router, the CLI, the
web UI).

A start (the box's start lock, `swap.start_turn`: the run, the load, auto-fix's halts and
retries, a workflow's every member) and a stop or removal are each a **change**
(`changing`). For as long as it lasts, a change holds a marker file under an exclusive
`flock`, and it moves the stamp as it begins and ends. A window is not alone when the stamp
moved during it, or when a change other than its own thread's is in progress as it closes
(`others_changing`). The kernel drops a dead process's lock, so a marker left by a killed
start (`wsl --shutdown`, an OOM kill) is known stale whatever pid it names, and is removed.

What nothing here sees: llama-server's own idle sleep (`sleep_idle_seconds`, off unless set),
and an engine that dies by itself; both free memory with no event.
"""

from __future__ import annotations

import contextlib
import fcntl
import itertools
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path

from makenomistakes.config import Settings

FILE = "events"  # under <state>/swap/, beside the start lock
MARKER = "changing-"  # <state>/swap/changing-<pid>-<n>: a change in progress, locked

_mine = threading.local()  # the markers this thread holds
_numbers = itertools.count()


def _folder(settings: Settings) -> Path:
    return settings.expanded_state_dir() / "swap"


def touch(settings: Settings) -> None:
    """Move the stamp; a failure to write counts as nothing (a reading taken meanwhile may
    then be counted wrongly, never a start or stop failed)."""
    folder = _folder(settings)
    with contextlib.suppress(OSError):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / FILE
        path.touch()
        now = time.time_ns()
        os.utime(path, ns=(now, now))


def stamp(settings: Settings) -> int | None:
    """When something last started or stopped; None when nothing has or it cannot be read."""
    try:
        return (_folder(settings) / FILE).stat().st_mtime_ns
    except OSError:
        return None


@contextlib.contextmanager
def changing(settings: Settings) -> Iterator[None]:
    """Around a start or a stop: its marker, locked while it lasts, and the stamp moved as it
    begins and ends."""
    folder = _folder(settings)
    number = next(_numbers)
    marker = folder / f"{MARKER}{os.getpid()}-{number}"
    held: set[str] = getattr(_mine, "markers", set())
    _mine.markers = held
    handle = None
    staging = folder / f".staging-{os.getpid()}-{number}"
    with contextlib.suppress(OSError):
        folder.mkdir(parents=True, exist_ok=True)
        # Locked under another name, then renamed: a probe never sees the marker unlocked.
        handle = staging.open("a")
        fcntl.flock(handle, fcntl.LOCK_EX)
        staging.rename(marker)
    with contextlib.suppress(OSError):  # left by a failed rename, never for good
        staging.unlink(missing_ok=True)
    held.add(marker.name)
    touch(settings)
    try:
        yield
    finally:
        held.discard(marker.name)
        with contextlib.suppress(OSError):
            marker.unlink(missing_ok=True)
        if handle is not None:
            handle.close()  # drops the lock
        touch(settings)


def others_changing(settings: Settings) -> bool:
    """Whether a start or stop other than this thread's own is in progress on the box. A
    marker nobody holds locked is stale and removed; an unreadable folder counts as yes."""
    mine: set[str] = getattr(_mine, "markers", set())
    try:
        markers = [p for p in _folder(settings).glob(f"{MARKER}*") if p.name not in mine]
    except OSError:
        return True
    return any(_held(marker) for marker in markers)


def _held(marker: Path) -> bool:
    try:
        with marker.open("a") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return True  # its holder is alive
            marker.unlink(missing_ok=True)  # left by a process that died holding it
            return False
    except FileNotFoundError:
        return False  # finished meanwhile
    except OSError:
        return True
