"""A cancel of a queued tune: one for the router, the web UI and the CLI
(`cancel_tune`), judged under the tuning folder's lock by `unqueue_here`, never by its caller
first, and worded here.

A tune nothing runs ends there. One a live run holds (or a quality plan a live approval
downloads) is marked cancelled on its queue entry (`TuneStore.mark_cancelled`): that run reads
the mark at its next look and ends with nothing saved, and no later cut, held trial or pause
can undo it. The router and the web UI also cut their own run so it looks now; a terminal's
run looks after the trial or step under way. One whose run is saving its result is too late,
and so is one whose run saved it before it stopped: a mark a stopped run left is settled
first, by the one rule (`tune.orphans.settle`), and the cancel goes on from what that leaves.
`decline` remembers the key a cancel took on, so the router's stale sweep does not queue that
tune again until the key moves."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.quality.entries import OwnerCancel, end_plan
from makenomistakes.tune.apply import key_or_none
from makenomistakes.tune.ends import End, EndOutcome, left_words, marked_end, runs_words
from makenomistakes.tune.orphans import Settled, Verdict, settle
from makenomistakes.tune.store import QualityQueued, Queued, TuneStore, TuneStoreError
from makenomistakes.tune.words import WORDS, capital, of, reason, what

Say = Callable[[str], None]
CutOwn = Callable[[str], bool]  # cut this process's own run of a tune; False: none runs
TOO_LATE = "too late to cancel (`mnm tune --status` shows what it kept)"


@dataclass(frozen=True)
class Cancelled:
    """What the cancel came to, in words, the headline first."""

    status: int  # 200: it took, or nothing was queued; 409: it did not
    lines: tuple[str, ...]
    took: bool  # the tune ends by it (not: nothing was queued, too late, refused)

    @property
    def ok(self) -> bool:
        return self.status < 400

    @property
    def words(self) -> str:
        return " ".join(self.lines)


def cancel_tune(
    service: InstanceService, store: TuneStore, name: str, cut_own: CutOwn | None = None
) -> Cancelled:
    """Cancel `name`'s tune: judged by `unqueue_here`; a run of this process's own that it
    marked is cut by `cut_own` (the router's tuner, the web UI's run), so it looks now."""
    said: list[str] = []
    end = unqueue_here(service, store, name, said.append)
    ours = end.outcome is EndOutcome.CANCELLING and end.pid == os.getpid()
    cut = ours and cut_own is not None and cut_own(name)
    status, headline = _worded(name, end, cut, said)  # before the decline's line: never hidden
    if end.took and (left := decline(service, store, name)) is not None:
        said.append(left)  # the cancel took: what kept it from being remembered, said
    return Cancelled(status, (*headline, *said), end.took)


Answer = tuple[int, tuple[str, ...]]  # the status, the headline (before what the end said)


def _worded(name: str, end: End, cut: bool, said: list[str]) -> Answer:
    answer: Answer
    match end.outcome:
        case EndOutcome.ENDED if end.entry is None:  # an unreadable entry removed, or progress
            nothing = WORDS["speed"].nothing
            answer = 200, () if said else (f"The tune of {name} is cancelled; {nothing}.",)
        case EndOutcome.ENDED:
            entry = end.found
            answer = 200, (f"{capital(what(entry))} is cancelled; {of(entry).nothing}.",)
        case EndOutcome.CANCELLING:
            entry = end.found
            how = f"{of(entry).work} is being stopped" if cut else _stops(entry, end.pid)
            answer = 200, (f"Cancelling {what(entry)}: {how}, {of(entry).ending}.",)
        case EndOutcome.SAVING:
            answer = 409, (_too_late(end, "has finished and is saving its result"),)
        case EndOutcome.SAVED:
            answer = 409, (_too_late(end, "had saved its result before its run stopped"),)
        case EndOutcome.UNREAD:
            unknown = f"whether its run had saved its result could not be read ({reason(end.why)})"
            answer = 200, (f"{capital(what(end.found))} is cancelled; {unknown}.",)
        case EndOutcome.STOPPING:
            answer = 200, ()  # its line said: the approval ends it
        case EndOutcome.GONE:
            answer = 200, (f"No tune of {name} was queued.",)
        case EndOutcome.RUNS if end.entry is None:  # beside an entry it cannot read
            answer = (
                409,
                (
                    f"{capital(runs_words(name, end.pid))} under a queue entry this mnm cannot "
                    f"read: it goes on there (`mnm update`, or `mnm tune {name} --cancel` once it "
                    "stops).",
                ),
            )
        case _:
            answer = 409, (left_words(end, name) or f"The tune of {name} was not cancelled.",)
    return answer


def _too_late(end: End, doing: str) -> str:
    return f"{capital(what(end.found))} {doing}: {TOO_LATE}."


def _stops(entry: Queued | QualityQueued, pid: int | None) -> str:
    who = f"pid {pid}: a terminal's `mnm tune --now`, the web UI or the router"
    return f"its run ({who}) stops after the {of(entry).unit} under way"


def decline(service: InstanceService, store: TuneStore, name: str) -> str | None:
    """A cancel that took: remember the key it was made on, so a stale tune is not queued
    again by the router's sweep until the key changes or the user queues it (none to
    remember while the recipe cannot be read: the sweep cannot read it either). What kept it
    from being written comes back in words, never raised: the cancel took (a full disk here
    must never answer a cancel that ended its tune with a traceback)."""
    try:
        recipe = service.find(name)
    except InstanceError:
        return None
    key = key_or_none(service, recipe, None) if recipe is not None else None
    if key is None:
        return None
    try:
        store.decline(name, key)
    except (TuneStoreError, OSError) as exc:
        return (
            f"The cancel of {name}'s tune could not be remembered ({exc}): the router may queue "
            "it again once it reads as tuned for something else."
        )
    return None


def unqueue_here(service: InstanceService, store: TuneStore, name: str, say: Say) -> End:
    """A cancel of `name`'s queued tune, judged under the tuning folder's lock. A mark a
    stopped run left is settled first (`tune.orphans`): a cancel it completes is this one
    (ENDED), a save that had landed is too late (SAVED), one it cannot settle now fails (its
    words); otherwise the cancel goes on from what the settle left. A quality plan goes
    through `end_plan` (ended, or marked for the live run or approval holding it). A speed
    tune nothing runs is unqueued with its progress; one a live run holds is marked cancelled
    (CANCELLING), unless that run is saving its result (SAVING). Nothing queued: a leftover
    progress nobody runs is forgotten; an entry mnm cannot read is removed (a cancel starts
    over), unless a live run holds it."""
    lines: list[str] = []  # said once the lock is let go (a terminal stopped mid-line holds it)
    try:
        with store.locked():
            return _unqueued(service, store, name, lines.append)
    except (TuneStoreError, OSError) as exc:
        return End(EndOutcome.FAILED, why=str(exc))
    finally:
        for line in lines:
            say(line)


def _unqueued(service: InstanceService, store: TuneStore, name: str, say: Say) -> End:
    try:
        queued = store.queued(name)
    except TuneStoreError:  # a newer mnm's, or an earlier shape: no run of this mnm reads it
        return _unreadable(store, name, say)
    if queued is None:  # a run without its entry stops at its next look
        ended = store.run_beside(name) is None and _forgot_progress(store, name)
        return End(EndOutcome.ENDED if ended else EndOutcome.GONE)
    settled = settle(service, store, queued)  # a mark a stopped run left: the one rule's
    for line in settled.lines:  # what its end said (the cancel words its own headline)
        say(line)
    if (end := _settled(service, store, settled, say)) is not None:
        return end
    now = store.current(queued)  # as the settle left it (unmarked, or as it was)
    if now is None:
        return End(EndOutcome.GONE)
    if isinstance(now, QualityQueued):
        return end_plan(service, now, say, OwnerCancel())
    return _speed(store, now, say)


def _settled(service: InstanceService, store: TuneStore, settled: Settled, say: Say) -> End | None:
    """What the settle came to for a cancel: a cancel it completed is this one (ENDED), a
    save that had landed is too late (SAVED), an end that did not take fails (its words), a
    save it cannot judge is ended the conservative way (`_unread`: the user asked); None
    when the cancel goes on (no mark, one a live run or approval holds, or one it unmarked)."""
    if settled.verdict is Verdict.ENDED:
        outcome = EndOutcome.SAVED if settled.saved else EndOutcome.ENDED
        return End(outcome, entry=settled.entry)
    if settled.verdict is Verdict.FAILED:
        return End(EndOutcome.FAILED, why=settled.problem)
    if settled.verdict is Verdict.UNSURE:
        return _unread(service, store, settled.entry, settled.problem, say)
    return None


def _unread(
    service: InstanceService,
    store: TuneStore,
    queued: Queued | QualityQueued,
    problem: str,
    say: Say,
) -> End:
    """A mark whose run's save may or may not have landed (its recipe unreadable): the user's
    cancel ends it the conservative way, which is right whichever it was (never a 409 while
    the words promise the cancel): a speed tune is unqueued with its
    progress; a quality plan ends with no save named, so the quant a landed save moved away
    from is kept, and the sweep deletes nothing while a recipe cannot be read."""
    if isinstance(queued, QualityQueued):
        end = end_plan(service, queued, say, OwnerCancel())
        if end.outcome is not EndOutcome.ENDED:
            return end
    elif (left := store.end_speed(queued)) is not None:
        say(left)
    return End(EndOutcome.UNREAD, entry=queued, why=problem)


def _speed(store: TuneStore, queued: Queued, say: Say) -> End:
    """A speed tune nothing runs is unqueued with its progress; one a live run holds is
    marked for that run to end, too late once it saves its result."""
    if (held := store.holder(queued)) is None:
        if (left := store.end_speed(queued)) is not None:
            say(left)
        return End(EndOutcome.ENDED, entry=queued)
    return marked_end(store.mark_cancelled(queued), held.pid, queued)


def _unreadable(store: TuneStore, name: str, say: Say) -> End:
    """An entry mnm cannot read is removed (a cancel starts over), unless a live run holds it."""
    if (pid := store.run_beside(name)) is not None:
        return End(EndOutcome.RUNS, pid)
    store.unqueue(name)
    _forgot_progress(store, name)
    say(f"Removed {name}'s queue entry, which this mnm cannot read: nothing of it ran.")
    return End(EndOutcome.ENDED)


def _forgot_progress(store: TuneStore, name: str) -> bool:
    """A tune in progress with no entry and no live run (an end cut short) is forgotten."""
    try:
        leftover = store.state(name) is not None
    except TuneStoreError:  # one mnm cannot read: a cancel starts over, as its words say
        leftover = True
    if leftover:
        store.drop_state(name)
    return leftover
