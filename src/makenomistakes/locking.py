"""One lock per state folder, across threads and processes.

The web service and a terminal are two processes writing the same folder (a download
checkpoint and `mnm set`): threads of one process queue on an
RLock, processes on `flock` of a `.lock` file in the folder. `flock` is not re-entrant
across opens, so it is taken once per process and depth-counted.
"""

from __future__ import annotations

import fcntl
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

LOCK_FILE = ".lock"  # the file each locked folder keeps for `flock`


class FolderLock:
    def __init__(self, folder: Path, name: str = LOCK_FILE) -> None:
        self._folder = folder
        self._name = name  # the file `flock` takes in the folder
        self._lock = threading.RLock()
        self._held = 0
        self._handle: IO[str] | None = None

    @contextmanager
    def __call__(self, waiting: Callable[[], None] | None = None) -> Iterator[None]:
        """`waiting` is told once when another process holds the lock (a start that queues
        behind another says so instead of hanging silently)."""
        with self._lock:
            if self._held == 0:
                self._folder.mkdir(parents=True, exist_ok=True)
                handle = (self._folder / self._name).open("a")
                try:
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        if waiting is not None:
                            waiting()
                        fcntl.flock(handle, fcntl.LOCK_EX)
                except BaseException:
                    handle.close()  # never held: nothing else would close it
                    raise
                self._handle = handle
            self._held += 1
            try:
                yield
            finally:
                self._held -= 1
                if self._held == 0 and self._handle is not None:
                    fcntl.flock(self._handle, fcntl.LOCK_UN)
                    self._handle.close()
                    self._handle = None

    @contextmanager
    def attempt(self) -> Iterator[bool]:
        """The lock only if nobody else holds it now (True), else False at once: a status
        read never waits for a start that holds it for minutes."""
        if not self._lock.acquire(blocking=False):
            yield False
            return
        try:
            if self._held == 0:
                self._folder.mkdir(parents=True, exist_ok=True)
                handle: IO[str] = (self._folder / self._name).open("a")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    yield False  # another process holds it
                    return
                self._handle = handle
            self._held += 1
            try:
                yield True
            finally:
                self._held -= 1
                if self._held == 0 and self._handle is not None:
                    fcntl.flock(self._handle, fcntl.LOCK_UN)
                    self._handle.close()
                    self._handle = None
        finally:
            self._lock.release()


_SHARED: dict[tuple[Path, str], FolderLock] = {}
_SHARED_GUARD = threading.Lock()


def shared_lock(folder: Path, name: str = LOCK_FILE) -> FolderLock:
    """The one `FolderLock` this process holds for `folder` (and lock file `name`): every
    caller that must take turns with the others asks here, so a thread re-entering through
    another object is depth-counted instead of deadlocking on its own `flock`."""
    key = folder.resolve()
    with _SHARED_GUARD:
        return _SHARED.setdefault((key, name), FolderLock(key, name))
