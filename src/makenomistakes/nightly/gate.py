"""When the nightly may use the card: the way the router's tuner waits (`tune/runner`),
asked from outside the router, since the nightly is a oneshot of its own. It runs once the
router has been idle a minute: nothing in flight and no request arrived (`/__mnm/busy`'s
counts; `--now` skips the minute, never the rest), no tune or training running in the router
(`tuning`, `training`), no start or stop in progress on the box (`box_events`), ComfyUI
running no job. It waits for that at most `LIMIT_SECONDS`, then gives up and the night's report
says why. Between `try --all` rows it looks again (`in_use`): its own rows went through the
router, so only what is in flight then for another instance than the row just tried, a tune or
someone else's start counts (a request the row timed out on stays in flight in the router,
and must not stop the night on its own row).

A tune in the router never makes it give up (else the tunes one night queued hold the card
through the next night's wait): the nightly holds the tuner off the
card (`POST /__mnm/tune-hold`), which cuts a running trial as a request would and keeps every
tune waiting while the night runs; the lease (`HOLD_SECONDS`) is renewed between rows and
steps and let go at the end, so a night that dies lets go by itself."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from makenomistakes import box_events
from makenomistakes.comfy.engine import ComfyEngine
from makenomistakes.config import Settings
from makenomistakes.router_client import RouterNow, hold_tunes, router_now

IDLE_SECONDS = 60.0
LIMIT_SECONDS = 30 * 60.0
EVERY_SECONDS = 5.0
HOLD_SECONDS = 45 * 60.0  # the hold's lease: longer than any one row (Bolt's was 13 min)
NOT_SAID = "the router did not say what it serves (`/__mnm/busy` answered something else)"


@dataclass(frozen=True)
class Gate:
    router: Callable[[], RouterNow | None]  # None: a router is there and did not say
    changing: Callable[[], bool]  # a start or stop by someone else is in progress
    comfy_busy: Callable[[], str | None]  # why ComfyUI holds the card (`busy_words`)
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    # hold the router's tuner off the card for N seconds (0 lets go): its words, None when no
    # router took it (none runs, or an older one)
    hold: Callable[[float], str | None] = lambda _seconds: None

    def blocked(self, now: RouterNow | None, own: frozenset[str] = frozenset()) -> str | None:
        """Why the card is not the nightly's now, the idle minute aside; None when it is.
        Requests in flight for `own` (the row just tried) do not count."""
        if now is None:
            return NOT_SAID
        if (job := _card_job(now)) is not None:
            return job
        if serving := sorted(n for n, count in now.busy.items() if count and n not in own):
            return f"the router is serving {', '.join(serving)}"
        if self.changing():
            return "a start or stop is in progress"
        if why := self.comfy_busy():  # running a workflow, or not saying
            return why
        return None

    def in_use(self, own: frozenset[str] = frozenset()) -> str | None:
        """Between rows: what took the router meanwhile, or None. A request the row just
        tried still has in flight (one its client gave up on: the engine still answers it)
        is the nightly's own, never a reason to stop. The tuner's hold is renewed first."""
        self.hold(HOLD_SECONDS)
        return self.blocked(self.router(), own)


def _card_job(now: RouterNow) -> str | None:
    """A tune or a training the router runs, in words."""
    if now.tuning:
        return f"a tune runs in the router: {now.tuning}"
    if now.training:
        return f"a training runs in the router: {now.training}"
    return None


@dataclass(frozen=True)
class Waited:
    seconds: float
    gave_up: str | None = None  # why it never got the card
    held: bool = False  # it holds the tuner off the card: the caller lets go (`hold(0)`)


def wait(
    gate: Gate,
    idle_seconds: float = IDLE_SECONDS,
    limit_seconds: float = LIMIT_SECONDS,
    every: float = EVERY_SECONDS,
    say: Callable[[str], None] = lambda _line: None,
    who: str = "The nightly",
) -> Waited:
    """Until the router has been idle `idle_seconds` and nothing else holds the card, or
    `limit_seconds` passed; each new reason to wait is said once."""
    began = gate.clock()
    quiet_since = began
    arrivals: int | None = None
    said: str | None = None
    holding = False
    while True:
        now = gate.router()
        jobs = now is not None and (now.tuning or now.training)
        if jobs and not holding:  # a tune or a training: the night comes first
            said_hold = gate.hold(HOLD_SECONDS)
            holding = said_hold is not None
            if holding:
                say(f"{who} holds the router's card jobs off the card: {said_hold}.")
                now = gate.router()
        moment = gate.clock()
        why = gate.blocked(now)
        if why is not None or (now is not None and now.arrivals != arrivals):
            quiet_since = moment  # a request taken meanwhile starts the minute again
        arrivals = now.arrivals if now is not None else None
        quiet = moment - quiet_since >= idle_seconds
        if why is None and quiet:
            return Waited(moment - began, held=holding)
        reason = why or "the router has not been idle for a minute yet"
        if moment - began >= limit_seconds:
            return Waited(moment - began, gave_up=reason, held=holding)
        if reason != said:
            said = reason
            say(f"{who} waits: {reason}.")
        gate.sleep(every)


def gate_for(settings: Settings, who: str = "the nightly") -> Gate:
    """The real gate: the router on this machine, the box's start markers, ComfyUI; `who`
    is what the router's journal says holds the tuner off the card."""
    return Gate(
        router=lambda: router_now(settings),
        changing=lambda: box_events.others_changing(settings),
        comfy_busy=ComfyEngine(settings).busy_words,  # unknown waits, never reads as idle
        hold=lambda seconds: hold_tunes(settings, seconds, who),
    )
