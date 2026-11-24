"""What the router's long card jobs share: the tuner's trials and the trainer's
trainings take the card only while the user leaves it alone, one job at a time.

- **The quiet minute**: a job starts only once the router has been idle a minute (nothing in
  flight, nothing arrived: its own `Traffic`); after a job the minute starts again.
- **The nightly's hold** (`POST /__mnm/tune-hold`): a hold cuts
  the job running (as a request does: it resumes later from what it saved) and keeps every
  job waiting until it is released or its lease runs out (the nightly renews it; one that
  dies lets go by itself).
- **One job on the card**: a training never starts while a trial runs, nor a trial during a
  training (`take`, `give_back`).
- **A request cuts the job running** (`arrived`: the router calls it once for every request;
  the hold cuts the same way).

The runners themselves share one loop (`card_runner.CardJobRunner`)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel, Field

from makenomistakes.journal import Note
from makenomistakes.router_client import Traffic

IDLE_SECONDS = 60.0  # the router idle this long before a queued job starts
HOLD_LIMIT_SECONDS = 4 * 3600.0  # a lease at most this long: a hold never outlives its night


class HoldAsk(BaseModel):
    seconds: float = Field(ge=0, le=HOLD_LIMIT_SECONDS)
    who: str = Field(default="the nightly", max_length=60)


class CardJob(Protocol):
    """A runner whose job a request or a hold cuts (`card_runner.CardJobRunner`)."""

    def running(self) -> str | None: ...  # "the tune of tiny", or None

    def arrived(self, name: str = "", why: str = ...) -> None: ...  # stop it; it resumes later


class CardGate:
    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        idle_seconds: float = IDLE_SECONDS,
        note: Note | None = None,
    ) -> None:
        self._clock = clock
        self._idle_seconds = idle_seconds
        self._note = note
        self._lock = threading.Lock()
        self._arrivals: int | None = None
        self._quiet_since = clock()
        self._held: tuple[float, str] | None = None  # (until, who)
        self._using: str | None = None
        self._jobs: list[CardJob] = []

    def join(self, job: CardJob) -> None:
        self._jobs.append(job)

    # --- the quiet minute ---------------------------------------------------------------

    def quiet(self, traffic: Traffic) -> bool:
        """The router idle for the minute (`traffic`: what it serves now, read by the job's
        runner): nothing in flight, nothing arrived."""
        now = self._clock()
        with self._lock:
            if traffic.busy and any(traffic.busy.values()):
                self._quiet_since = now
            if self._arrivals != traffic.arrivals:
                self._arrivals = traffic.arrivals
                self._quiet_since = now
            return now - self._quiet_since >= self._idle_seconds

    def rested(self) -> None:
        """A job ended: the next one waits the minute again."""
        with self._lock:
            self._quiet_since = self._clock()

    # --- the nightly's hold -------------------------------------------------------------

    def hold(self, asked: HoldAsk) -> tuple[int, str]:
        """Hold every job off for `asked.seconds` (the one running cut first); 0 lets go."""
        with self._lock:
            was = self._held
            self._held = (self._clock() + asked.seconds, asked.who) if asked.seconds > 0 else None
        if asked.seconds <= 0:
            if was is not None:
                self._say(
                    f"{was[1].capitalize()} lets the tuner and the trainer have the card again."
                )
                self.rested()  # a job cut for it waits the minute again
            return 200, "released"
        paused = [(job, words) for job in self._jobs if (words := job.running()) is not None]
        if not paused:
            return 200, "held: no tune or training runs"
        for job, _words in paused:
            job.arrived("", f"{asked.who} holds the card")  # the job's own line says so
        # A job cut for the hold resumes after it; one saving its result as the hold came
        # ended instead (it held nothing of the card by then): its own line says which.
        return 200, f"held: {paused[0][1]} gave the card back"

    def holder(self) -> str | None:
        """Who holds the card against the jobs now, None when nobody (or the lease ran out)."""
        with self._lock:
            if self._held is None:
                return None
            until, who = self._held
            if self._clock() >= until:
                self._held = None
                return None
            return who

    # --- one job on the card -------------------------------------------------------------

    def take(self, job: str) -> str | None:
        """The card for `job` ("the tune of tiny"): None when it got it, else the job holding
        it, in words."""
        with self._lock:
            if self._using is not None and self._using != job:
                return self._using
            self._using = job
            return None

    def give_back(self, job: str) -> None:
        with self._lock:
            if self._using == job:
                self._using = None

    def using(self) -> str | None:
        with self._lock:
            return self._using

    def arrived(self, name: str) -> None:
        """A request is about to be served: every job running gives the card back first."""
        for job in self._jobs:
            job.arrived(name)

    def _say(self, line: str) -> None:
        if self._note is not None:
            self._note(line)
