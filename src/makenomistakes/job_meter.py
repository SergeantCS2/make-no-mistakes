"""sd-server's queued jobs, timed: a clip (`POST /sdcpp/v1/vid_gen`) or an engine-page
picture (`/sdcpp/v1/img_gen`) is queued with a 202 that names the job, then polled at
`/sdcpp/v1/jobs/<id>` until it says completed. Its seconds run from the POST to the poll that
says so: within the client's poll interval (`mnm try` polls every 2 s) of the engine's own.

sd-server writes its JSON with sorted keys (nlohmann's default object), so a completed job
ends `…,"started":<ms>,"status":"completed"}` after the clip's base64: the end of the reply,
which is what the router's meter keeps (`forward.REPLY_TAIL`), carries the word.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from makenomistakes.measurements import SpeedUnit

if TYPE_CHECKING:
    from makenomistakes.reading_stamp import CardBefore

JOB_ROUTES = {"/sdcpp/v1/vid_gen": SpeedUnit.CLIP, "/sdcpp/v1/img_gen": SpeedUnit.IMAGE}
JOB_POLL = "/sdcpp/v1/jobs/"
MAX_JOBS = 64  # queued jobs remembered per router; a client that never polls is forgotten
_ID = re.compile(rb'"id"\s*:\s*"([A-Za-z0-9_.-]{1,128})"')
_DONE = re.compile(rb'"status"\s*:\s*"completed"')
_ENDED = re.compile(rb'"status"\s*:\s*"(failed|cancelled)"')


def job_unit(method: str, path: str) -> SpeedUnit | None:
    """The unit a queued job's time is read in, for a POST that queues one; else None."""
    if method != "POST":
        return None
    return JOB_ROUTES.get(path.split("?", 1)[0].rstrip("/"))


def polled_job(method: str, path: str) -> str | None:
    """The job a poll asks about (`GET /sdcpp/v1/jobs/<id>`), else None."""
    route = path.split("?", 1)[0]
    if method != "GET" or JOB_POLL not in route:
        return None
    job = route.split(JOB_POLL, 1)[1].strip("/")
    return job if job and "/" not in job else None


@dataclass(frozen=True)
class Queued:
    """A job the meter will time when a poll says it is done."""

    began: float  # the meter's clock at the POST
    unit: SpeedUnit
    cold: bool  # the first generation after its container started: it moved the weights in
    build: str
    started: str | None
    card: CardBefore | None = None  # the card when it was queued


class JobClock:
    """The jobs queued through this router, by instance and id; thread-safe."""

    def __init__(self) -> None:
        self._jobs: dict[tuple[str, str], Queued] = {}
        self._lock = threading.Lock()

    def queued(self, name: str, reply_tail: bytes, job: Queued) -> None:
        """A 202's reply names the job; one without an id is not timed."""
        found = _ID.search(reply_tail)
        if found is None:
            return
        with self._lock:
            while len(self._jobs) >= MAX_JOBS:
                self._jobs.pop(next(iter(self._jobs)))
            self._jobs[(name, found.group(1).decode())] = job

    def done(self, name: str, job: str, reply_tail: bytes) -> Queued | None:
        """The queued job a completed poll ends, taken off the list; None for any other poll
        (still queued or generating, failed, or a job queued elsewhere)."""
        if _DONE.search(reply_tail) is None:
            if _ENDED.search(reply_tail) is not None:
                with self._lock:
                    self._jobs.pop((name, job), None)
            return None
        with self._lock:
            return self._jobs.pop((name, job), None)
