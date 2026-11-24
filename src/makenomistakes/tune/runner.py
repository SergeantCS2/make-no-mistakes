"""The tuner's loop inside the router process, beside the idle sweep
(`idle.sweep_forever` is its model: a thread the router starts and stops).

Every few seconds it looks at the queue. The next tune starts only once the router has been
idle for a minute (no request in flight, none arrived: its own `Traffic`), no start or stop
is in progress on the box (`box_events`), ComfyUI runs nothing, and nothing the router never
parks holds the card (a pinned model running, a set in use: `held_back`); a `--now` tune
skips the minute but waits for the rest. Each trial makes room by the router's rules (`tune.room`).

**An outside request cuts the trial**: the router calls `arrived` before it serves any
request; the trial's copy is stopped there and then, the request waits (at most
`CUT_WAIT_SECONDS`) until the trial has cleaned up, and is served as usual (the model it
asks for loads). The tune stays queued and resumes from its last completed trial once the
router is idle again. Tunes whose key no longer holds (another build, card, driver, RAM,
files) are queued again every `STALE_EVERY` seconds.

**The nightly holds the card** (`POST /__mnm/tune-hold`: else the tunes one night queued
take the idle card from the next night's run): a hold cuts a running trial as a request does
(the copy stopped, the trials kept for the resume) and keeps every tune waiting until it is
released or its lease runs out (the nightly renews it between rows and steps; one that dies
lets go by itself).

The loop itself is `card_runner.CardJobRunner`, which the trainer runs too, and the quiet
minute, the hold and the card are a `card_jobs.CardGate` they share: one job on the
card at a time, both held by the nightly."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from pydantic import ValidationError

from makenomistakes.card_jobs import IDLE_SECONDS, CardGate, HoldAsk
from makenomistakes.card_runner import CardJobRunner
from makenomistakes.instance_types import InstanceError
from makenomistakes.journal import Lasting, Note
from makenomistakes.local_time import local
from makenomistakes.measurements import settings_key
from makenomistakes.quality.entries import Bystander, end_plan, sweep
from makenomistakes.quality.steps import refusal
from makenomistakes.recipes import Recipe
from makenomistakes.router_client import Traffic
from makenomistakes.tune.apply import key_or_none, stale_word
from makenomistakes.tune.cancel import cancel_tune
from makenomistakes.tune.ends import EndOutcome, left_words
from makenomistakes.tune.orphans import LEFT_MARKED, Refused, Verdict, queue_speed, settle
from makenomistakes.tune.session import TuneEnd, Tuner, untunable
from makenomistakes.tune.store import (
    Entry,
    Holds,
    QualityQueued,
    Queued,
    TuneAsk,
    TuneStoreError,
)
from makenomistakes.tune.trial import copy_name
from makenomistakes.tune.trial_types import Cut
from makenomistakes.tune.words import named, unread_waits

STALE_EVERY = 300.0


class TuneRunner(CardJobRunner[Entry, TuneEnd]):
    noun = "tune"
    name = "tuner"

    def __init__(
        self,
        tuner: Tuner,
        traffic: Callable[[], Traffic],
        comfy_busy: Callable[[], str | None] = lambda: None,  # why ComfyUI holds the card
        note: Note | None = None,
        clock: Callable[[], float] = time.monotonic,
        idle_seconds: float = IDLE_SECONDS,
        gate: CardGate | None = None,
    ) -> None:
        self.tuner = tuner
        self._stale_at: float | None = None
        self._sweep_said: frozenset[str] = frozenset()  # a lasting problem is said once
        self._queue_said = Lasting(self.say)  # why the queue cannot be listed, said once
        self._unsettled: dict[str, str] = {}  # entries a settle left marked, and why
        super().__init__(
            tuner.service,
            gate or CardGate(clock, idle_seconds, note),
            traffic,
            comfy_busy,
            note,
            clock,
        )

    # --- the queue, as the shared loop asks for it -----------------------------------------

    def entries(self) -> Sequence[Entry]:
        return self._waiting()

    def _waiting(self) -> list[Entry]:
        """The queue; one whose folder the system will not list is nothing to run or end this
        tick, its reason said once while it lasts (read as empty it would let the sweep
        delete a ready plan's downloads, and raised it would fail the tick every 5 s)."""
        try:
            found = self.tuner.store.waiting()
        except TuneStoreError as exc:
            self._queue_said.failed(f"The tuner waits: {exc}.")
            return []
        self._queue_said.worked()
        return found

    def key(self, entry: Entry) -> str:
        return entry.name

    def target(self, entry: Entry) -> str:
        return entry.name

    def asked_now(self, entry: Entry) -> bool:
        return entry.now

    def own_waits(self, entry: Entry) -> str | None:
        """Why this entry alone cannot run yet; the entries behind it are still asked, and
        the card's own check (`waits`) may hold them all."""
        holder = self.tuner.store.holder(entry)
        if holder is not None and holder.by == "run":  # this runner runs nothing at a tick
            return f"another process (pid {holder.pid}) runs {entry.name}'s tune"
        if (left := self._unsettled.get(entry.name)) is not None:  # settled at every tick
            return left
        if isinstance(entry, QualityQueued) and not entry.ready:
            return _approval_waits(entry, holder)
        try:
            recipe = self.tuner.service.find(entry.name)
        except InstanceError as exc:  # a newer mnm's recipe, a damaged one: said once, never
            return unread_waits(entry.name, str(exc))  # raised at every tick
        if recipe is not None and not recipe.downloaded:  # `mnm add --no-download`, evicted
            return f"{entry.name} is not downloaded yet (`mnm start {entry.name}` downloads it)"
        return None

    def work(self, entry: Entry, cut: Cut) -> TuneEnd:
        return self.tuner.tune(entry, cut)

    def stop_job(self, key: str) -> None:
        self.tuner.service.supervisor.stop(copy_name(key))

    def after(self, entry: Entry, result: TuneEnd, cancelled: bool) -> None:
        """A cut or held tune waits for the next idle minute, its place kept (only this entry:
        one queued in its place since stays). A cancelled one ended its entry itself: its
        session read the mark on its entry, which no later cut overwrites."""
        del cancelled  # the entry's mark says it, wherever the cancel came from
        if result.status in {"cut", "held"}:
            self.tuner.store.wait_again(entry)

    def before_tick(self) -> None:
        self._end_orphans()
        self._requeue_stale()

    def _end_orphans(self) -> None:
        """A tune nothing will run is settled here: one whose mark outlived its run (one rule
        for both kinds, `tune.orphans`: a cancel completed, a landed save ended, an unlanded
        one unmarked so its next run saves it), and a quality plan whose approval stopped
        before its downloads ended (a web UI restarted mid-job, a killed terminal) or whose
        instance was removed (`mnm rm` keeps its entry: the store alone cannot tell which
        files are listed). Each is judged again under the store's lock, where an approval
        takes one over or another process starts running it, and what it said is said once
        that lock is let go. One that cannot be settled now (its recipe unreadable, its end
        failing) waits, its reason said once (`own_waits`), and is asked again every tick."""
        store, service = self.tuner.store, self.tuner.service
        unsettled: dict[str, str] = {}
        for entry in self._waiting():
            lines: list[str] = []  # said once the lock is let go
            try:
                with store.locked():
                    found = store.current(entry)
                    if found is None:
                        continue
                    settled = settle(service, store, found)
                    why = self._orphaned(found) if settled.verdict is Verdict.UNTOUCHED else None
                    end = None
                    if why is not None and isinstance(found, QualityQueued):
                        end = end_plan(service, found, lines.append, Bystander())
            except TuneStoreError:  # unreadable since `waiting` read it: the next tick sees
                continue
            if settled.verdict in LEFT_MARKED:  # it waits: `own_waits` says why, once
                unsettled[found.name] = settled.why
            for line in (*settled.news, *lines):  # a mark it left: its reason, said once
                self.say(line)
            if end is not None and end.outcome is EndOutcome.ENDED:
                self.say(f"Ended {named(found)}: {why}.")
        self._unsettled = unsettled
        # Holds whose entry is gone (an end cut short after its unqueue) and the files a
        # failed delete left pending, every tick; a problem that lasts is said once.
        swept = sweep(service, store)
        for problem in swept.problems:
            if problem not in self._sweep_said:
                self.say(problem)
        self._sweep_said = frozenset(swept.problems)
        if swept.total:
            self.say(f"Deleted what a quality tune left behind: {swept.total / 2**30:.1f} GiB.")

    def _orphaned(self, found: Entry) -> str | None:
        """Why nothing will ever run a quality plan queued as `found` (the store's lock held):
        its approval stopped mid-download, or its instance was removed."""
        store, service = self.tuner.store, self.tuner.service
        if not isinstance(found, QualityQueued) or found.closing is not None:
            return None  # a mark is the orphan rule's (`settle`): held by a live run, here
        if store.holder(found) is not None:
            return None  # its live run or approval goes on: that one ends it
        if not found.ready:
            return "its approval stopped mid-download"
        try:
            gone = service.find(found.name) is None
        except InstanceError:
            return None  # an unreadable recipe is no removed one
        return "it was removed" if gone else None

    # --- the router's side ----------------------------------------------------------------

    def hold(self, body: bytes) -> tuple[int, str]:
        """`POST /__mnm/tune-hold`: `{"seconds": N, "who": "the nightly"}` holds every
        card job off for N seconds (the tune or training running cut first); `{"seconds": 0}`
        lets go."""
        try:
            asked = HoldAsk.model_validate_json(body or b"{}")
        except ValidationError as exc:
            return 400, f"not a hold: {exc.errors()[0]['msg']}"
        return self.gate.hold(asked)

    def holder(self) -> str | None:
        """Who holds the card against the card jobs now, None when nobody."""
        return self.gate.holder()

    def ask(self, name: str, body: bytes) -> tuple[int, str]:  # noqa: PLR0911 - an answer each
        """`POST /__mnm/tune/<name>`: queue a tune (`now` skips the idle minute) or cancel
        one; the status and the words for the CLI."""
        try:
            asked = TuneAsk.model_validate_json(body or b"{}")
        except ValidationError as exc:
            return 400, f"not a tune request: {exc.errors()[0]['msg']}"
        if asked.action == "cancel":  # before the recipe: one that cannot be read is the
            return self.cancel(name)  # cancel's to end (never a 503)
        service = self.tuner.service
        recipe = service.find(name)
        if recipe is None:
            return 404, f"no instance named {name!r}"
        if (why := untunable(recipe)) is not None:
            return 409, f"{name}: {why}"
        if asked.kind == "quality":
            return self._ask_quality(recipe, asked)
        if (refused := self._gave_up(recipe, asked)) is not None:
            return 409, refused
        store = self.tuner.store
        entry = Queued(name=name, mode=asked.mode, minutes=asked.minutes, now=asked.now)
        said: list[str] = []  # a settle's words: journaled, and answered to the asker too
        queued = queue_speed(
            service, store, entry.model_copy(update={"reason": "asked"}), said.append
        )
        for line in said:
            self.say(line)
        if isinstance(queued, Refused):
            return 409, " ".join([*said, queued.words])
        queued_words = f"Queued {name}: the router's tuner runs it {self._when(asked)}."
        return 202, " ".join([*said, queued_words])

    def _when(self, asked: TuneAsk) -> str:
        """When the router's tuner runs a tune it just queued: never "now" while the queue's
        folder cannot be listed, since the tuner reads its queue by that listing (else the
        answer promises a run that waits for the folder)."""
        try:
            self.tuner.store.queued_names()
        except TuneStoreError as exc:
            return f"once its queue can be listed again ({exc}; `mnm tune --status`)"
        when = "now" if asked.now else "once the router has been idle a minute"
        return f"{when} (`mnm tune --status`)"

    def _ask_quality(self, recipe: Recipe, asked: TuneAsk) -> tuple[int, str]:
        """The quality tune the asker wrote into the queue, named by its plan's
        fingerprint: the router runs plans from disk, never one sent to it. One it
        cannot run (the instance moved, a quant's files went) is unqueued with the quants it
        downloaded; else it runs now if asked."""
        name, store, service = recipe.name, self.tuner.store, self.tuner.service
        queued = store.queued(name)
        if not isinstance(queued, QualityQueued) or queued.quality.fingerprint != asked.fingerprint:
            return 409, (
                f"no quality tune of {name} is queued under that plan: `mnm tune {name} "
                "--quality` plans and queues one"
            )
        if not queued.ready:
            return 409, f"{queued.words} is still downloading its quants: ask again once it is in"
        if self.active == name:  # running here already: nothing to judge or move
            return 202, f"The router's tuner runs {queued.words} now."
        if (why := refusal(service, recipe, queued.quality)) is not None:
            return self._unrunnable(queued, why)
        if asked.now and not store.mark_now(queued):  # cancelled since it was read
            return 409, f"{queued.words} was cancelled or ended meanwhile"
        return 202, f"Queued {queued.words}: the router's tuner runs it {self._when(asked)}."

    def _unrunnable(self, queued: QualityQueued, why: str) -> tuple[int, str]:
        """A plan the router cannot run ends with its downloads, and the asker hears what
        went. `end_plan` judges it (a cancel or an end since, a run that began meanwhile,
        here or in another process, is not ended here: that run ends it itself)."""
        said: list[str] = []

        def say(line: str) -> None:
            said.append(line)
            self.say(line)

        end = end_plan(self.tuner.service, queued, say, Bystander())
        if end.outcome is EndOutcome.GONE:
            return 409, f"{queued.words} was cancelled or ended meanwhile"
        if (left := left_words(end, queued.name)) is not None:
            return 409, f"{why}. {left}"
        return 409, " ".join([f"{why}: its quality tune ends.", *said])

    def _gave_up(self, recipe: Recipe, asked: TuneAsk) -> str | None:
        """Why the nightly's ask is not queued (a tune that gave up on these settings is
        never queued again every night for the same reason); the user's ask clears it."""
        store, name = self.tuner.store, recipe.name
        gave_up = store.gave_up(name)
        if (
            gave_up is not None
            and gave_up.settings == settings_key(recipe)
            and asked.by == "nightly"
        ):
            return (
                f"the tune of {name} gave up on {local(gave_up.at):%Y-%m-%d} on these settings "
                f"({gave_up.why}): `mnm tune {name} --now` tries again"
            )
        store.clear_gave_up(name)  # the user asks again, or the settings moved
        return None

    def cancel(self, name: str) -> tuple[int, str]:
        """`mnm tune <name> --cancel` asked of the router (the client's POST waits 2 s): judged
        under the tuning folder's lock (`tune.cancel.cancel_tune`), never on what the router
        saw before. A tune a live run holds is marked cancelled for that run to end at its
        next look (one of this router's own is cut too, so it looks now); one saving its
        result is too late. The lines go to the journal and back to the asker."""
        cancelled = cancel_tune(self.tuner.service, self.tuner.store, name, self.cancel_running)
        for line in cancelled.lines:
            self.say(line)
        return cancelled.status, cancelled.words

    def line(self) -> str | None:
        """`tuning tiny (trial 7 of ≤24, 31 min left)` while a tune runs."""
        state = self.tuner.state
        return state.line if self.active is not None and state is not None else None

    def _requeue_stale(self) -> None:
        now = self._clock()
        if self._stale_at is not None and now - self._stale_at < STALE_EVERY:
            return
        self._stale_at = now
        service = self.tuner.service
        store = self.tuner.store
        try:
            recipes = [r for r in service.all() if r.tuned is not None]
        except Exception:
            return
        snapshot = service.memory() if recipes else None
        for recipe in recipes:
            if store.queued(recipe.name) is not None:
                continue
            key = key_or_none(service, recipe, snapshot)
            if key is None or store.declined(recipe.name, key):  # cancelled on this key
                continue
            if (why := stale_word(recipe, key)) is not None and isinstance(
                queue_speed(
                    service, store, Queued(name=recipe.name, reason=f"tuned for {why}"), self.say
                ),
                Queued,
            ):
                self.say(f"{recipe.name} was tuned for {why}: queued to tune again.")


def _approval_waits(entry: QualityQueued, holder: Holds | None) -> str:
    """Why a plan still downloading waits: its approval goes on (cancelled, it stops once the
    file under way is in), or it stopped and the orphan sweep ends it."""
    if holder is None or holder.by != "approval":  # (a run holds no plan downloading)
        return "its approval stopped mid-download"
    if entry.cancelled:
        return f"its approval was cancelled and stops once its file is in (pid {holder.pid})"
    return f"its approval is still downloading its quants (pid {holder.pid})"
