"""The router's activity: its last `KEEP` journal lines (`journal.py`: loads and their
seconds, the parks they made, idle unloads, every refusal it answers itself, every `/mcp`
call), each with its time, kept in memory and served at `GET /__mnm/events` behind the key,
for the web System page. Each line still goes to the journal (stderr) as before.

The ring lives in the router's process only: a restart starts it empty, the journal keeps the
rest (`mnm service logs`, `mnm report`)."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from datetime import datetime

from pydantic import BaseModel

from makenomistakes.journal import Note, to_stderr

KEEP = 200
EVENTS_PATH = "/__mnm/events"


class Event(BaseModel):
    at: datetime  # the box's own clock, with its zone
    line: str


class Events(BaseModel):
    """`GET /__mnm/events`: oldest first."""

    events: list[Event]


def _now() -> datetime:
    return datetime.now().astimezone()


class EventLog:
    """`note` is the journal: it writes each line on (`out`) and keeps the newest `keep`."""

    def __init__(
        self,
        out: Note = to_stderr,
        keep: int = KEEP,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self._out = out
        self._clock = clock
        self._lines: deque[Event] = deque(maxlen=keep)
        self._lock = threading.Lock()

    def note(self, line: str) -> None:
        self._out(line)
        with self._lock:
            self._lines.append(Event(at=self._clock(), line=line))

    def recent(self) -> Events:
        with self._lock:
            return Events(events=list(self._lines))
