"""The queue's figures: what one window of an instance's requests waited, as a
`Kind.QUEUE` reading keeps it (`router_queue` counts them), and what waits now (`/__mnm/busy`'s
`queue`). A leaf module: `measurements` (the reading's field) and `router_client` (the busy
answer) read it without the router's machinery; the summaries are `queue_summary`."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Reason(StrEnum):
    LOAD = "load"  # waited for its model to be ready: a load, its turn, the card taken back
    SLOT = "slot"  # waited at the engine behind its slots


class Tally(BaseModel):
    """How many, their total and the largest (seconds, or tokens)."""

    n: int = 0
    total: float = 0.0
    top: float = 0.0

    def add(self, value: float) -> None:
        self.n += 1
        self.total += value
        self.top = max(self.top, value)

    def merged(self, other: Tally) -> Tally:
        top = max(self.top, other.top)
        return Tally(n=self.n + other.n, total=self.total + other.total, top=top)

    @property
    def mean(self) -> float | None:
        return self.total / self.n if self.n else None


class QueueFigures(BaseModel):
    """One window of an instance's work requests: a burst with less than
    `router_queue.QUIET_SECONDS` between them, at most `WINDOW_SECONDS` long, within one
    container start. Every figure is counted in the window open when it is measured."""

    since: datetime  # the window's first request arrived
    until: datetime  # its last one ended
    requests: int = 0  # ended in it: answered, refused before the engine, or the client gone
    forwarded: int = 0  # of them, the ones that had reached the engine (the shares' base)
    peak: int = 0  # the most in flight at once (arrived, not ended)
    slots: int = Field(default=1, ge=1)  # what the engine served at once (`model_meta.at_once`)
    # forwarded while every slot was taken, and how long until one was theirs (the router's
    # count: the engines take requests in order)
    slot_waits: Tally = Field(default_factory=Tally)
    # waited past the grace for their model to be ready: a load, their turn, the card back
    load_waits: Tally = Field(default_factory=Tally)
    timed: int = 0  # answers that carried llama.cpp's timings
    # of those, llama.cpp's own: the time at the engine outside its prompt and generation
    # (deferred behind others, its prompt cache's copy), past `router_queue.WAIT_FLOOR`
    engine_waits: Tally = Field(default_factory=Tally)
    context: Tally = Field(default_factory=Tally)  # the tokens one request held (prompt + reply)

    def merged(self, other: QueueFigures) -> QueueFigures:
        """Two windows of one settings as one (the summary): counts added, the largest kept."""
        return QueueFigures(
            since=min(self.since, other.since),
            until=max(self.until, other.until),
            requests=self.requests + other.requests,
            forwarded=self.forwarded + other.forwarded,
            peak=max(self.peak, other.peak),
            slots=self.slots,
            slot_waits=self.slot_waits.merged(other.slot_waits),
            load_waits=self.load_waits.merged(other.load_waits),
            timed=self.timed + other.timed,
            engine_waits=self.engine_waits.merged(other.engine_waits),
            context=self.context.merged(other.context),
        )


class Waiting(BaseModel):
    """What waits on one instance now (`/__mnm/busy`'s `queue`, only where something does)."""

    load: int = 0  # past the grace for its model to be ready (a load, its turn, the card)
    slot: int = 0  # at the engine, behind its slots
    answering: int = 0  # at the engine, in a slot

    @property
    def total(self) -> int:
        return self.load + self.slot


def waiting_words(name: str, now: Waiting) -> str:
    """`2 waiting on coder (for a slot; 1 answering)`, `1 waiting on zimage (for it to be
    ready)`: the Status board's and `mnm ls`'s."""
    why = []
    if now.slot:
        why.append(f"{now.slot} for a slot" if now.load else "for a slot")
    if now.load:
        why.append(f"{now.load} for it to be ready" if now.slot else "for it to be ready")
    said = ", ".join(why)
    if now.answering:
        said += f"; {now.answering} answering"
    return f"{now.total} waiting on {name} ({said})"
