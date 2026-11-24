"""What a queued tune whose mark outlived its run becomes: one rule for both kinds, decided
here and nowhere else (`settle`), asked first by every way into the queue (`queue_speed`, a
quality approval) and by everything that runs, cancels or ends a queued tune: the router's
tick, a session about to run it, the user's cancel.

- `cancelled`: the user's cancel is completed: a speed tune is unqueued with its progress, a
  quality plan ends through `end_plan` (what it downloaded deleted).
- `saving` whose result landed (the instance's recipe carries a record of the entry's kind
  made since the entry was queued: `Recipe.tuned`, `Recipe.quality`): it ends as its run would
  have, a quality plan with the quant its save moved the instance away from and a speed tune
  queued for the new quant's settings (`quality.entries.end_saved`).
- `saving` whose result never landed (the run died in its save: a SIGTERM from `mnm update`'s
  restart, the OOM killer, a terminal closed): unmarked, so its next run saves it, never
  ended with its downloads deleted under words saying it was kept.
- `saving` whose recipe cannot be read now: left marked and asked again, never an end on a
  guess; one whose instance was removed since ends (nothing could save it any more).

`landed` holds because nothing merges into a marked entry: `TuneStore.queue` refuses one, and
every way into the queue settles it first, so an entry never inherits a save it did not make
(a merge that kept a dead run's `queued_at` would read that run's landed save as the next
run's own). `settle` runs under the store's lock, which its caller holds; what its end
says comes back on the outcome, for the caller to say once that lock is let go (a terminal
stopped mid-line would hold it)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.quality.entries import Bystander, end_plan, end_saved
from makenomistakes.recipes import Recipe
from makenomistakes.recipes.quality import QualityTuned
from makenomistakes.recipes.tuned import Tuned
from makenomistakes.tune.ends import End, EndOutcome
from makenomistakes.tune.store import Entry, QualityQueued, Queued, TuneStore, TuneStoreError
from makenomistakes.tune.words import (
    HELD_WHY,
    REMOVED_WHY,
    SAVED_WHY,
    UNSAVED_WHY,
    cancelled_why,
    capital,
    closing_words,
    failed_why,
    named,
    unsure_why,
    what,
)

Say = Callable[[str], None]
EndStatus = Literal["cancelled", "skipped", "held", "failed"]  # a session's, when it runs none


class Verdict(StrEnum):
    UNTOUCHED = "untouched"  # no mark: it waits, or runs, as it is
    HELD = "held"  # a live run or approval holds it: that one ends it
    UNMARKED = "unmarked"  # its run stopped before its save landed: it waits again
    ENDED = "ended"  # a cancel completed, a landed save ended, or its instance gone
    UNSURE = "unsure"  # whether its save landed cannot be read now: left marked
    FAILED = "failed"  # the tuning folder kept it from taking (read, write): left, asked again


LEFT_MARKED = frozenset({Verdict.UNSURE, Verdict.FAILED})  # what it could not settle now


@dataclass(frozen=True)
class Settled:
    """What `settle` made of an entry: the verdict, the entry as found, the journal's reason,
    and what its end said (for the caller to say once the store's lock is let go)."""

    verdict: Verdict
    entry: Entry
    why: str = ""
    lines: tuple[str, ...] = ()
    saved: bool = False  # it ended because its run's save had landed: a cancel is too late
    problem: str = ""  # UNSURE, FAILED: what the recipe or the tuning folder said

    @property
    def runs(self) -> bool:
        """Nothing of the mark stands in the way of a run of it, or of a queue writing it."""
        return self.verdict in {Verdict.UNTOUCHED, Verdict.UNMARKED}

    @property
    def line(self) -> str | None:
        """The one line every caller says about it; None when nothing came of it."""
        if self.verdict is Verdict.ENDED:
            return f"Ended {named(self.entry)}: {self.why}."
        if self.verdict in {Verdict.UNMARKED, Verdict.UNSURE, Verdict.FAILED}:
            return f"{capital(what(self.entry))}: {self.why}."
        return None

    @property
    def told(self) -> tuple[str, ...]:
        """What to say of it, in order: its end's lines, then its own line."""
        return (*self.lines, self.line) if self.line is not None else self.lines

    @property
    def news(self) -> tuple[str, ...]:
        """What to say of it beside a caller's own words for a mark it left (the router's wait,
        a refusal): `told` when it settled the mark, nothing when it left it (unsure, failed:
        it ended nothing, so it carries no end's lines, and its line is the caller's)."""
        return () if self.verdict in LEFT_MARKED else self.told

    @property
    def status(self) -> EndStatus:
        """A session's status for an entry it does not run (`runs` False)."""
        if self.verdict is Verdict.ENDED:
            return "cancelled" if self.entry.cancelled else "skipped"
        return "held" if self.verdict is Verdict.HELD else "failed"


def settle(service: InstanceService, store: TuneStore, entry: Entry) -> Settled:
    """The rule above for `entry` as queued now (the store's lock held by the caller)."""
    if entry.closing is None:
        return Settled(Verdict.UNTOUCHED, entry)
    if store.holder(entry) is not None:  # a live run or approval: it ends it itself
        return Settled(Verdict.HELD, entry, HELD_WHY)
    try:  # a mark it leaves ended nothing: no end's lines (each step before raises first)
        return _decided(service, store, entry)
    except (TuneStoreError, OSError) as exc:  # the read, the unmark or the unqueue itself
        return Settled(Verdict.FAILED, entry, failed_why(str(exc)), problem=str(exc))


@dataclass(frozen=True)
class _Unread:
    problem: str  # the recipe's own words


def _judged_by(service: InstanceService, entry: Entry) -> Recipe | _Unread | None:
    """What a save's landing is judged by: the instance's recipe as it reads now (None: the
    instance was removed), or why it cannot be read (a newer mnm's, a damaged file, the
    disk). The one read, for `settle` and `unjudged` alike."""
    try:
        return service.find(entry.name)
    except InstanceError as exc:
        return _Unread(str(exc))


def unjudged(service: InstanceService, entry: Entry) -> str | None:
    """Why `settle` cannot judge `entry`'s mark now, in the recipe's own words: a save while
    its recipe cannot be read; None when it can, for a cancel (ended without the recipe) and
    for no mark. System words a mark by it, through the settle's own read (never one of its
    own)."""
    read = _judged_by(service, entry) if entry.saving else None
    return read.problem if isinstance(read, _Unread) else None


def closing_now(service: InstanceService, store: TuneStore, entry: Entry) -> str | None:
    """`entry`'s state as the user reads it in `mnm tune --status`, the Tuning tab and System
    (`closing_words`): who holds it now, and a mark nothing holds worded by `unjudged`."""
    holder = store.holder(entry)
    return closing_words(entry, holder, unread=None if holder else unjudged(service, entry))


def _unsure(entry: Entry, problem: str) -> Settled:
    return Settled(Verdict.UNSURE, entry, unsure_why(entry.name, problem), problem=problem)


def _decided(service: InstanceService, store: TuneStore, entry: Entry) -> Settled:
    """A mark nothing holds any more: a cancel completed; a save that landed ended, one that
    did not unmarked; an instance removed since ends whatever its mark."""
    if entry.cancelled:
        return _ended(service, store, entry, cancelled_why(entry))
    recipe = _judged_by(service, entry)
    if isinstance(recipe, _Unread):  # whether the save landed is unknown: left, never a guess
        return _unsure(entry, recipe.problem)
    if recipe is None:
        return _ended(service, store, entry, REMOVED_WHY)
    if not _landed(recipe, entry):
        store.unmark(entry)
        return Settled(Verdict.UNMARKED, entry, UNSAVED_WHY)
    moved_from = _moved_from(recipe, entry)
    return _ended(service, store, entry, SAVED_WHY, saved=True, moved_from=moved_from)


@dataclass(frozen=True)
class Refused:
    """Why a speed tune was not queued, decided in the same hold of the lock as the settle
    before it: the settle's own line when it left the mark there (`left`: a sentence naming
    the tune whose mark it is), else the store's word. `why` is that reason alone when the
    words name the tune, for a row that names it itself (adopt's table)."""

    words: str
    left: bool = False
    why: str | None = None

    def line(self, name: str) -> str:
        """The refusal said where nothing else words it (`mnm add`, an engine update); never
        capitalised, which would misname an instance its words begin with."""
        if self.left:
            return f"{self.words} No tune of {name} was queued."
        return f"No tune of {name} was queued: {self.words.removesuffix('.')}."


def queue_speed(
    service: InstanceService, store: TuneStore, entry: Queued, say: Say
) -> Queued | Refused:
    """Queue a speed tune the way every caller queues one (the CLI, the web UI, the router's
    asks and its stale sweep, `mnm add`, adopt, an engine update): the mark of the entry it
    would merge into settled first, under the same hold of the lock, and the entry stamped
    with the time of that write (`TuneStore.queue`). What the settle said is said once the
    lock is let go; a refusal comes back in words decided under it (never the store's word
    read again after the lock)."""
    settled = None
    with store.locked():
        if (before := store.queued(entry.name)) is not None:
            settled = settle(service, store, before)
        queued = store.queue(entry)
        result = queued if queued is not None else _refusal(store, entry, settled)
    for line in settled.news if settled is not None else ():
        say(line)
    return result


def _refusal(store: TuneStore, entry: Queued, settled: Settled | None) -> Refused:
    """The settle's own line when it left the mark there, else the store's word."""
    if settled is not None and settled.verdict in LEFT_MARKED and settled.line is not None:
        return Refused(settled.line, left=True, why=settled.why)
    return Refused(store.blocking(entry) or f"{entry.name}'s tune could not be queued")


def _record(recipe: Recipe, entry: Entry) -> Tuned | QualityTuned | None:
    return recipe.quality if isinstance(entry, QualityQueued) else recipe.tuned


def _landed(recipe: Recipe, entry: Entry) -> bool:
    """`entry`'s run saved its result before it stopped: the recipe's record of that kind was
    saved for this very entry (`made_for`, stamped in the same recipe write as the result:
    one tune of an instance runs at a time, its entry is unqueued once its result is saved,
    and nothing merges into a marked entry). Neither the clocks (a box's clock can step back
    2.27 s every 32 s) nor a later mnm's serialization (the update that restarts a router mid-
    save) move it, and a restored older record names another entry. A record saved before the
    stamp existed is judged by the time rule."""
    record = _record(recipe, entry)
    if record is None:
        return False
    if record.made_for is not None:
        return record.made_for == entry.queued_at
    return record.at >= entry.queued_at


def _moved_from(recipe: Recipe, entry: Entry) -> str | None:
    """The quant a landed quality save moved the instance away from (None: it kept it)."""
    if not isinstance(entry, QualityQueued):
        return None
    current = entry.quality.current
    return current if recipe.quant != current else None


def _ended(
    service: InstanceService,
    store: TuneStore,
    entry: Entry,
    why: str,
    saved: bool = False,
    moved_from: str | None = None,
) -> Settled:
    """`entry` ended as its run would have: a speed tune unqueued with its progress, a quality
    plan through `end_plan` (`end_saved` after a landed save: its follow-up speed tune too).
    An end that did not take said nothing: each step before the unqueue raises first."""
    lines: list[str] = []
    if not isinstance(entry, QualityQueued):
        if (left := store.end_speed(entry)) is not None:  # raises only for the unqueue
            lines.append(left)
        return Settled(Verdict.ENDED, entry, why, tuple(lines), saved)
    asker = Bystander()
    end: End = (
        end_saved(service, entry, lines.append, asker, moved_from)
        if saved
        else end_plan(service, entry, lines.append, asker)
    )
    if end.outcome is EndOutcome.ENDED:
        return Settled(Verdict.ENDED, entry, why, tuple(lines), saved)
    if end.outcome is not EndOutcome.FAILED or end.why is None:  # held, gone: never, under
        raise RuntimeError(f"a settle's end of {entry.name} came to {end.outcome}")  # this lock
    return Settled(Verdict.FAILED, entry, failed_why(end.why), problem=end.why)
