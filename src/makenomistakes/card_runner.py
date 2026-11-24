"""One loop for the router's card jobs (the tuner, the trainer), written once: each is
a thread in the router that serves a queue of jobs, each job taking the card only while the
user leaves it alone, and giving it back the moment a request comes.

A runner says what its queue holds (`entries`), what one job needs before it may start
(`own_waits`), how to run it (`work`, handed the `Cut` that ends it early) and how to stop it
from another thread (`stop_job`); this base does the rest the same way for both:

- **Waits** (`waits`): the nightly's hold, requests in flight, the router's quiet minute
  (`--now` skips it, never the rest), a start or stop by someone else, ComfyUI running a
  workflow, a pinned model or a set in use holding the card, and the card gate's one job at
  a time (`CardGate.take`, which names the job holding it).
- **A request cuts the job** (`arrived`, which the router reaches through `CardGate.arrived`
  for every request, and the gate's hold for the nightly): the cut is set, the job stopped
  (a training's `docker stop` gives Applio up to 30 s), then the caller waits at most
  `CUT_WAIT_SECONDS` more while it ends. A request that came after the
  waits were read and before the job began is caught by comparing the router's arrivals with
  this runner's own snapshot.
- **Each reason to wait is journaled once** per job (`_said`); "not idle yet" and "requests in
  flight" are not journaled at all (they come and go every minute).
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Sequence
from threading import Event
from typing import ClassVar

from makenomistakes import box_events
from makenomistakes.card_jobs import CardGate
from makenomistakes.instances import InstanceService
from makenomistakes.journal import Lasting, Note
from makenomistakes.router_client import Traffic
from makenomistakes.set_store import WorkflowStore
from makenomistakes.tune.room import held
from makenomistakes.tune.trial_types import CANCELLED, REQUEST_CAME, Cut

TICK_SECONDS = 5.0
CUT_WAIT_SECONDS = 60.0  # how long a request waits for a cut job to clean up
QUIET_WAITS = frozenset({"the router is not idle yet", "requests in flight"})


class CardJobRunner[E, R]:
    """`E` a queued job (a tune's entry, a training's run), `R` what running one came to."""

    noun: ClassVar[str]  # "tune", "training": "the tune of tiny"
    name: ClassVar[str]  # "tuner", "trainer": "The trainer failed and tries again …"

    def __init__(
        self,
        service: InstanceService,
        gate: CardGate,
        traffic: Callable[[], Traffic],
        comfy_busy: Callable[[], str | None] = lambda: None,  # why ComfyUI holds the card
        note: Note | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.service = service
        self.gate = gate
        self._traffic = traffic
        self._comfy_busy = comfy_busy
        self._note = note
        self._clock = clock
        self._lock = threading.Lock()
        self._cut = Cut()
        self._done = Event()
        self._done.set()
        self.active: str | None = None  # the running job's key
        self._active_words: str | None = None
        self._seen: int | None = None  # the router's arrivals at this runner's last look
        self._waits_said = Lasting(self.say)  # each job's reason to wait, said once
        self._cancelled: set[str] = set()
        self._workflows = WorkflowStore(service.settings.expanded_state_dir())
        gate.join(self)

    # --- what a runner says --------------------------------------------------------------

    def entries(self) -> Sequence[E]:
        raise NotImplementedError

    def key(self, entry: E) -> str:
        raise NotImplementedError

    def target(self, entry: E) -> str:
        """What the job is of, in words ("tiny", "alto")."""
        raise NotImplementedError

    def work(self, entry: E, cut: Cut) -> R:
        raise NotImplementedError

    def stop_job(self, key: str) -> None:
        """Stop the running job's process from another thread (its container, its copy)."""
        raise NotImplementedError

    def asked_now(self, entry: E) -> bool:
        return False

    def own_waits(self, entry: E) -> str | None:
        """Why this job alone cannot start yet; the jobs behind it are still asked (the card's
        own check, `waits`, may hold them all: `tune.room.held` while a recipe is unreadable)."""
        return None

    def after(self, entry: E, result: R, cancelled: bool) -> None:
        """What becomes of the job once it ran (queued again after a cut, …)."""

    def before_tick(self) -> None:
        """Anything the runner does every tick before it looks at the queue."""

    def job_words(self, entry: E) -> str:
        return f"the {self.noun} of {self.target(entry)}"

    # --- the router's side ----------------------------------------------------------------

    def arrived(self, name: str = "", why: str = REQUEST_CAME) -> None:
        """A request is about to be served (or the nightly holds the card): a running job
        gives the card back first, its stop on this thread, then `CUT_WAIT_SECONDS` at most
        for it to end."""
        del name  # any request, for any model: the card is the user's again
        if self.cut_running(why):
            self._done.wait(CUT_WAIT_SECONDS)

    def cut_running(self, why: str) -> bool:
        """The running job cut and stopped, without waiting for it to end (what a job's own
        watcher calls: the job joins that watcher before it ends); False when none runs."""
        with self._lock:
            active = self.active
            if active is None:
                return False
            self._cut.cut(why)
        self._stop_quietly(active)
        return True

    def running(self) -> str | None:
        """The job on the card, in words; None when this runner runs none."""
        return self._active_words if self.active is not None else None

    def cancel_running(self, key: str, otherwise: Callable[[], None] = lambda: None) -> bool:
        """Cut the job `key` for good if it runs (its stop on a thread of its own: it takes
        seconds), and remember the cancel for `after` (a later cut cannot undo it); False when
        it does not run. `otherwise` runs under the same lock then, so a job about to start
        cannot slip between the check and what it does."""
        with self._lock:
            if self.active != key:
                otherwise()
                return False
            self._cut.cut(CANCELLED)
            self._cancelled.add(key)
        threading.Thread(target=self._stop_quietly, args=(key,), daemon=True).start()
        return True

    def was_cancelled(self, key: str) -> bool:
        """The user cancelled the job `key` while it ran: kept until it ends, so a later cut
        (a request, the nightly's hold) never words it as a pause."""
        with self._lock:
            return key in self._cancelled

    def locked(self) -> threading.Lock:
        """The runner's lock: a job's `work` re-reads what it runs under it."""
        return self._lock

    def _stop_quietly(self, key: str) -> None:
        with contextlib.suppress(Exception):  # already gone: nothing to stop
            self.stop_job(key)

    # --- the loop -------------------------------------------------------------------------

    def loop(self, stopped: Event, every: float = TICK_SECONDS) -> None:
        """Tick until `stopped`; a tick that fails is journaled and the next one tries. A
        failure that lasts is said once (`journal.Lasting`), never every 5 s."""
        said = Lasting(self.say)
        while not stopped.wait(every):
            try:
                self.tick()
            except Exception as exc:
                said.failed(f"The {self.name} failed and tries again every {every:.0f} s: {exc}")
            else:
                said.worked()

    def tick(self) -> R | None:
        """Start the next queued job if the card is free for it; what it came to, else None."""
        traffic = self._traffic()
        quiet = self.gate.quiet(traffic)
        self._seen = traffic.arrivals
        self.before_tick()
        for entry in self.entries():
            if (own := self.own_waits(entry)) is not None:
                self._said(entry, own)
                continue
            why = self.waits(entry, quiet)
            if why is None:
                holder = self.gate.take(self.job_words(entry))
                if holder is None:
                    return self._run(entry)
                why = f"{holder} has the card"
            self._said(entry, why)
            return None  # the queue is served in order while the card is not free
        return None

    def waits(self, entry: E, quiet: bool) -> str | None:
        """Why no job may take the card now; None when it may."""
        if (who := self.gate.holder()) is not None:  # even a `--now` job
            return f"{who} holds the card"
        if any(self._traffic().busy.values()):
            return "requests in flight"
        if not self.asked_now(entry) and not quiet:
            return "the router is not idle yet"
        if box_events.others_changing(self.service.settings):
            return "a start or stop is in progress"
        if why := self._comfy_busy():  # running a workflow, or not saying
            return why
        return held(self.service, self._workflows)

    def _run(self, entry: E) -> R:
        key, words = self.key(entry), self.job_words(entry)
        self._waits_said.worked(key)
        with self._lock:
            self._cut.clear()
            self._done.clear()
            self.active, self._active_words = key, words
        late = self._traffic()  # one that came after the waits were read
        if any(late.busy.values()) or (self._seen is not None and late.arrivals != self._seen):
            with self._lock:  # nothing runs yet to stop, and never `arrived`: it would wait
                self._cut.cut(REQUEST_CAME)  # on this very job for a minute
        try:
            result = self.work(entry, self._cut)
        finally:
            with self._lock:
                self.active = self._active_words = None
                cancelled = key in self._cancelled
                self._cancelled.discard(key)
            self._done.set()
            self.gate.give_back(words)
            self.gate.rested()  # the minute starts again after a job
        self.after(entry, result, cancelled)
        return result

    def _said(self, entry: E, why: str) -> None:
        """A reason a job waits, said once while it lasts. A quiet one (requests in flight, the
        router not idle yet) is neither said nor kept, so a lasting reason is not said again
        after every request."""
        if why in QUIET_WAITS:
            return
        words = self.job_words(entry)
        self._waits_said.failed(f"{words[0].upper()}{words[1:]} waits: {why}.", self.key(entry))

    def say(self, line: str) -> None:
        if self._note is not None:
            self._note(line)
