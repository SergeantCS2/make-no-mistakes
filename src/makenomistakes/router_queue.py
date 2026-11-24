"""The router's queue: which requests wait for a model, and for how long.

Every work request the router takes (`request_meter.measured_route`: a generation POST, on the
OpenAI routes and Ollama's) has a `Ticket` from before it is counted in flight to its end
(`Router.request`). Before it is forwarded it waits for its model (`Router.ready`: loaded first
when parked, or its turn behind a start in progress, or the card taken back from a tune or a
training); past `LOAD_GRACE` that is a load wait. Forwarded, it waits for a slot while the
engine holds as many requests as it serves at once (`model_meta.at_once`). By the router's
count a ticket has a slot while it is among the first `slots` of its container start's tickets
at the engine, in the order they went: every engine here takes them in order (llama-server's
deferred queue is first in, first out; sd-server, whisper-server and audio.cpp's server run
one request at a time behind one lock, each read in its source). llama.cpp's answer says it
from inside too (`engine_figures`): its `timings` give the time it read the prompt and
generated, so the rest of the request's time at the engine it waited, and the tokens the
request held, which a context split into more slots would have to hold.

`waiting` is what waits now (`/__mnm/busy`, `/metrics`, the Status board, `mnm ls`). A window
per instance gathers the figures (`QueueFigures`): a burst with less than `QUIET_SECONDS`
between requests, within one container start (a new start closes it), cut at `WINDOW_SECONDS`
whatever is in flight (the next window opens at once and takes the rest). A window counts
under the settings its container started with: that start's load reading
(`measurements.start_reading`, injected as `started_with`), else the recipe the request found.
The router's sweep writes each closed window that reached the engine as a `Kind.QUEUE` reading
(`write`); a window in memory when the router stops is lost. Not counted as a wait: an
sd-server job queued through its own route (`img_gen`, `vid_gen`: no request in flight while
it runs), and a request vLLM holds back for its KV cache below `--max-num-seqs`. Nothing here
fails a request: a figure that cannot be read is left out.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from makenomistakes.loading_notes import GRACE_SECONDS
from makenomistakes.measurements import Kind, Reading, reading_for
from makenomistakes.model_meta import at_once
from makenomistakes.queue_figures import QueueFigures, Reason, Tally, Waiting
from makenomistakes.reading_stamp import stamp_for
from makenomistakes.recipes import Recipe
from makenomistakes.reply_tail import last_number
from makenomistakes.request_meter import Container

if TYPE_CHECKING:
    from makenomistakes.instances import InstanceService

LOAD_GRACE = GRACE_SECONDS  # a loaded model is let through well inside it (loading notes')
QUIET_SECONDS = 120.0  # a window ends after this long with nothing in flight
WINDOW_SECONDS = 3600.0  # and is cut at this age whatever is in flight
WAIT_FLOOR = 0.25  # under this outside llama.cpp's prompt and generation: HTTP, the tokenizer
MAX_TOKENS = 10**9  # a count past this in a reply's tail is not one request's tokens

StartedWith = Callable[[str, str], Reading | None]  # (name, started at) -> its load reading


def _no_reading(_name: str, _started: str) -> Reading | None:
    return None


@dataclass(eq=False)
class Ticket:
    """One request from its arrival to its end; `counted` False: not work, never kept. The
    handler sets `tail` (the end of the reply it passed on) before the ticket ends."""

    name: str
    arrived: float
    counted: bool = True
    forwarded: float | None = None
    start: str | None = None  # the container start it went to
    queued: bool = False  # forwarded while every slot was taken
    began: float | None = None  # when it had a slot, by the router's count
    load_waited: float = 0.0  # its wait for the model, past the grace (counted where it ends)
    done: bool = False
    tail: bytes = b""


@dataclass(frozen=True)
class Inside:
    """What an answer says of its request from inside the engine (None: it does not say)."""

    seconds: float | None  # llama.cpp's prompt and generation time
    tokens: int | None  # the tokens it held: cached, read and written


def engine_figures(tail: bytes) -> Inside:
    """From the end of an answer (`reply_tail`): llama.cpp's `timings` (its seconds, and the
    tokens as `cache_n + prompt_n + predicted_n`), else the tokens of an OpenAI `usage` block
    (vLLM, or a llama.cpp route without timings) or Anthropic's (`/v1/messages`). A figure out
    of what one request can be (negative, past `MAX_TOKENS`) is left out."""
    if not tail:
        return Inside(None, None)

    def got(key: str) -> float | None:
        return last_number(tail, key)

    prompt, predicted = got("prompt_ms"), got("predicted_ms")
    seconds = (prompt + predicted) / 1000 if prompt is not None and predicted is not None else None
    held = [got("cache_n"), got("prompt_n"), got("predicted_n")]
    tokens: float | None = None
    if all(part is not None for part in held):
        tokens = sum(part for part in held if part is not None)
    elif (total := got("total_tokens")) is not None:
        tokens = total
    elif (said := got("input_tokens")) is not None and (wrote := got("output_tokens")) is not None:
        tokens = said + wrote + (got("cache_read_input_tokens") or 0.0)
    return Inside(
        seconds if seconds is not None and math.isfinite(seconds) else None,
        int(tokens) if tokens is not None and tokens <= MAX_TOKENS else None,
    )


@dataclass
class _Window:
    recipe: Recipe  # the request's, at the window's open
    opened: float  # the clock's
    last: float  # the clock's, at its newest arrival or end
    figures: QueueFigures
    started: str | None = None  # the container start it counts (None until one went to it)
    build: str | None = None
    load: Reading | None = None  # that start's load reading: the settings it counts under


@dataclass
class _Model:
    pending: list[Ticket] = field(default_factory=list)  # arrived, not forwarded
    engine: list[Ticket] = field(default_factory=list)  # forwarded, not ended, in their order
    window: _Window | None = None

    @property
    def in_flight(self) -> int:
        return len(self.pending) + len(self.engine)


@dataclass(frozen=True)
class Closed:
    """A window done, for the sweep to write."""

    recipe: Recipe
    build: str | None
    started: str | None
    load: Reading | None
    figures: QueueFigures


def _utc() -> datetime:
    return datetime.now(UTC)


class QueueBook:
    """One per router; thread-safe (its lock is held for counting only)."""

    def __init__(
        self,
        container: Container,  # `Router.container`: the router's own cache, Docker once
        started_with: StartedWith = _no_reading,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], datetime] = _utc,
    ) -> None:
        self._container = container
        self._started_with = started_with
        self._clock = clock
        self._wall = wall
        self._lock = threading.Lock()
        self._models: dict[str, _Model] = {}
        # name -> (its container start, that start's load reading or None): asked once a start
        self._loads: dict[str, tuple[str, Reading | None]] = {}
        self._closed: list[Closed] = []
        self._totals: dict[tuple[str, Reason], Tally] = {}  # since the router started

    def take(self, recipe: Recipe, work: bool) -> Ticket:
        """A ticket for a request about to wait for `recipe`; one that is not work (a listing,
        a page, a poll: `request_meter.measured_route`) is never counted."""
        if not work:
            return Ticket(recipe.name, 0.0, counted=False)
        with self._lock:  # every clock read under the lock: the order of events is the lock's
            now = self._clock()
            ticket = Ticket(recipe.name, now)
            model = self._models.setdefault(recipe.name, _Model())
            model.pending.append(ticket)
            window = model.window or self._open(model, recipe, now)
            window.last = now
            window.figures.peak = max(window.figures.peak, model.in_flight)
        return ticket

    def forwarded(self, ticket: Ticket, recipe: Recipe) -> None:
        """It goes to the engine now: its wait for the model ends, its wait for a slot may
        begin."""
        if not ticket.counted:
            return
        build, started = self._start_of(ticket.name)  # outside the lock: Docker, the store
        load = self._load_of(ticket.name, started)
        with self._lock:
            now = self._clock()
            model = self._models.get(ticket.name)
            if model is None or ticket not in model.pending:
                return
            model.pending.remove(ticket)
            window = self._of_start(model, recipe, build, started, load, now)
            if (waited := now - ticket.arrived) > LOAD_GRACE:
                ticket.load_waited = waited
            ticket.forwarded, ticket.start = now, started
            ahead = sum(1 for t in model.engine if t.start == started)
            ticket.queued = ahead >= window.figures.slots
            if not ticket.queued:
                ticket.began = now
            model.engine.append(ticket)
            window.figures.peak = max(window.figures.peak, model.in_flight)

    def ended(self, ticket: Ticket) -> None:
        """However it ended: answered (`ticket.tail`), refused, or its client gone. Once only;
        the ticket leaves first, whatever its figures say."""
        if not ticket.counted or ticket.done:
            return
        try:  # outside the lock; a reader that failed would strand the ticket
            inside = engine_figures(ticket.tail)
        except Exception:
            inside = Inside(None, None)
        with self._lock:
            now = self._clock()
            ticket.done = True
            model = self._models.get(ticket.name)
            if model is None:
                return
            waited_for_model = ticket in model.pending
            if waited_for_model:
                model.pending.remove(ticket)
            at_engine = ticket in model.engine
            if at_engine:
                self._left_engine(model, ticket, now)
            window = model.window  # a ticket in flight always has one (`take`, `_close`)
            if window is None:
                return
            if waited_for_model:  # refused, or its client left while it waited
                self._load_wait(window, ticket, now - ticket.arrived)
            elif at_engine:
                self._answered(window, ticket, now, inside)
            window.figures.requests += 1
            window.figures.until = self._wall()
            window.last = now

    def waiting(self) -> dict[str, Waiting]:
        """What waits on each instance now; only the instances where something does."""
        found: dict[str, Waiting] = {}
        with self._lock:
            now = self._clock()
            for name, model in self._models.items():
                load = sum(1 for t in model.pending if now - t.arrived > LOAD_GRACE)
                slot = sum(1 for t in model.engine if t.began is None)
                if load or slot:
                    found[name] = Waiting(load=load, slot=slot, answering=len(model.engine) - slot)
        return found

    def totals(self) -> dict[tuple[str, Reason], Tally]:
        """(instance, reason) -> the waits since the router started (`/metrics`)."""
        with self._lock:
            return {key: tally.model_copy() for key, tally in self._totals.items()}

    def closed(self) -> list[Closed]:
        """The windows done, once each: those a new container start closed, each quiet one
        (nothing in flight for `QUIET_SECONDS`), and each `WINDOW_SECONDS` old (what is in
        flight then goes on in the next window, opened at once)."""
        with self._lock:
            now = self._clock()
            for model in self._models.values():
                window = model.window
                if window is None:
                    continue
                if not model.in_flight and now - window.last >= QUIET_SECONDS:
                    self._close(model, now)
                elif now - window.opened >= WINDOW_SECONDS:
                    self._close(model, now, reopen=bool(model.in_flight))
            done, self._closed = self._closed, []
        return done

    # --- outside the lock --------------------------------------------------------------

    def _start_of(self, name: str) -> tuple[str | None, str | None]:
        try:
            return self._container(name)
        except Exception:  # Docker could not say: the window keeps no start
            return None, None

    def _load_of(self, name: str, started: str | None) -> Reading | None:
        """The start's load reading, asked of the store once per start (a start mnm made
        writes it before it answers; one with none counts under the request's recipe)."""
        if started is None:
            return None
        with self._lock:
            known = self._loads.get(name)
        if known is not None and known[0] == started:
            return known[1]
        try:
            found = self._started_with(name, started)
        except Exception:  # unreadable readings: the recipe stands in
            found = None
        with self._lock:
            self._loads[name] = (started, found)
        return found

    # --- under the lock ----------------------------------------------------------------

    def _open(self, model: _Model, recipe: Recipe, now: float) -> _Window:
        wall = self._wall()
        figures = QueueFigures(since=wall, until=wall, slots=at_once(recipe.args))
        model.window = _Window(recipe, now, now, figures)
        return model.window

    def _of_start(
        self,
        model: _Model,
        recipe: Recipe,
        build: str | None,
        started: str | None,
        load: Reading | None,
        now: float,
    ) -> _Window:
        """The window this container start's requests count in: a new start closes the open
        one (its settings may differ); its first request fixes the settings it counts under."""
        window = model.window or self._open(model, recipe, now)
        if window.started is not None and started is not None and window.started != started:
            self._close(model, now)
            window = self._open(model, recipe, now)
        if window.started is None and started is not None:
            window.recipe, window.started, window.build, window.load = recipe, started, build, load
            window.figures.slots = at_once(load.args if load is not None else recipe.args)
        return window

    def _load_wait(self, window: _Window, ticket: Ticket, waited: float) -> None:
        if waited > LOAD_GRACE:
            window.figures.load_waits.add(waited)
            self._totals.setdefault((ticket.name, Reason.LOAD), Tally()).add(waited)

    def _left_engine(self, model: _Model, ticket: Ticket, now: float) -> None:
        """Its place frees: the first `slots` of its start's tickets still there have a slot
        (one that leaves before its turn moves nobody ahead of it)."""
        model.engine.remove(ticket)
        window = model.window
        if window is None or window.started != ticket.start:
            return  # a request to a container gone: its start's count is over
        same = [t for t in model.engine if t.start == ticket.start]
        for other in same[: window.figures.slots]:
            if other.began is None:
                other.began = now

    def _answered(self, window: _Window, ticket: Ticket, now: float, inside: Inside) -> None:
        """A request that reached the engine ends: counted where it ends, as every figure is
        (one in flight at an hour's cut counts in the next window)."""
        window.figures.forwarded += 1
        self._load_wait(window, ticket, ticket.load_waited)
        forwarded = ticket.forwarded if ticket.forwarded is not None else now
        if ticket.queued:  # its client may have left before its turn: it waited till then
            waited = max((ticket.began if ticket.began is not None else now) - forwarded, 0.0)
            window.figures.slot_waits.add(waited)
            self._totals.setdefault((ticket.name, Reason.SLOT), Tally()).add(waited)
        if inside.seconds is not None:
            window.figures.timed += 1
            if (own := (now - forwarded) - inside.seconds) > WAIT_FLOOR:
                window.figures.engine_waits.add(own)
        if inside.tokens is not None:
            window.figures.context.add(inside.tokens)

    def _close(self, model: _Model, now: float, reopen: bool = False) -> None:
        """Hand the window to the sweep when a request that reached the engine ended in it (a
        burst of refusals says nothing of a slot, and is no use of the model); `reopen`: what
        is in flight goes on in a window of the same start, opened now."""
        window, model.window = model.window, None
        if window is None:
            return
        if window.figures.forwarded:
            found = Closed(window.recipe, window.build, window.started, window.load, window.figures)
            self._closed.append(found)
        if reopen:
            following = self._open(model, window.recipe, now)
            following.started, following.build = window.started, window.build
            following.load, following.figures.slots = window.load, window.figures.slots
            following.figures.peak = model.in_flight


def write(service: InstanceService, closed: list[Closed]) -> int:
    """Each closed window as a `Kind.QUEUE` reading of this box, dated by its last request's
    end and keyed by the settings its container started with (the sweep's thread); the
    readings written. One whose instance went (`mnm rm`) or that cannot be saved is left: a
    reading is a hint."""
    written = 0
    for window in closed:
        try:
            if service.find(window.recipe.name) is None:
                continue
            reading = reading_for(
                window.recipe,
                Kind.QUEUE,
                window.build,
                stamp=stamp_for(service, window.recipe),
                started_with=window.load,
                container_started=window.started,
                queue=window.figures,
                at=window.figures.until,
            )
            service.measurements.add(window.recipe.name, reading)
            written += 1
        except Exception:  # the next window is tried all the same
            continue
    return written
