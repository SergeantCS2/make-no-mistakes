"""What ending a queued tune came to: judged once, under the
tuning folder's lock, by what ends it (`quality.entries.end_plan` for a quality plan,
`tune.cancel.unqueue_here` for a cancel of either kind), never by a caller before it calls:
a check made outside that lock is stale by the time the end runs. An `End` carries facts only
(what it came to, who holds the entry, the entry as it was found); the user's cancel words
its own (`tune.cancel`), and an end nobody asked for words what it left with `left_words`."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from makenomistakes.tune.records import Entry, Marked
from makenomistakes.tune.words import capital


class EndOutcome(StrEnum):
    ENDED = "ended"  # unqueued (a quality plan's holds released, what nobody holds deleted)
    GONE = "gone"  # nothing of it was queued any more: cancelled, ended or replaced since
    CANCELLING = "cancelling"  # the user's cancel marked it; the live run holding it (`pid`)
    # ends it at its next look, nothing saved
    SAVING = "saving"  # the live run holding it (`pid`) saves its result now: too late
    SAVED = "saved"  # its run had saved its result before it stopped: ended now, too late
    UNREAD = "unread"  # the user's cancel ended it without knowing whether its run's save
    # had landed (its recipe unreadable, `why`): the conservative end, a moved quant kept
    STOPPING = "stopping"  # the user's cancel marked it; its live approval (`pid`) ends it
    RUNS = "runs"  # a live run holds it (`pid`): an end nobody cancelled leaves it to that run
    # a live approval downloads its quants (`pid`): likewise (no caller today asks an end
    # nobody cancelled of an entry still downloading, since a ready entry never regains an
    # approver; typed so that one never ends a plan under a live approval)
    DOWNLOADING = "downloading"
    FAILED = "failed"  # its queue entry could not be read or removed (`why`)


WITH_PID = frozenset(
    {
        EndOutcome.CANCELLING,
        EndOutcome.SAVING,
        EndOutcome.STOPPING,
        EndOutcome.RUNS,
        EndOutcome.DOWNLOADING,
    }
)
# the outcomes of an entry someone holds, or held: the entry as found comes with them
WITH_ENTRY = frozenset(
    {EndOutcome.CANCELLING, EndOutcome.SAVING, EndOutcome.SAVED, EndOutcome.UNREAD}
)
WITH_WHY = frozenset({EndOutcome.FAILED, EndOutcome.UNREAD})


@dataclass(frozen=True)
class End:
    outcome: EndOutcome
    pid: int | None = None  # who runs it or downloads its quants (`WITH_PID`)
    why: str | None = None  # what the tuning folder (FAILED) or the recipe (UNREAD) said
    entry: Entry | None = None  # the entry as the end found it (None: none, or unreadable)

    def __post_init__(self) -> None:
        if (self.outcome in WITH_PID) != (self.pid is not None):
            raise ValueError(f"an end that is {self.outcome} names a pid, and only then")
        if (self.outcome in WITH_WHY) != (self.why is not None):
            raise ValueError(f"an end that is {self.outcome} says why only if failed or unread")
        if self.outcome in WITH_ENTRY and self.entry is None:
            raise ValueError(f"an end that is {self.outcome} carries the entry it found")

    @property
    def found(self) -> Entry:
        """The entry of an outcome that carries one (`WITH_ENTRY`)."""
        if self.entry is None:
            raise ValueError(f"an end that is {self.outcome} carries no entry")
        return self.entry

    @property
    def took(self) -> bool:
        """A cancel took: the tune ended, or its run or approval ends it at its next look."""
        return self.outcome in {
            EndOutcome.ENDED,
            EndOutcome.UNREAD,
            EndOutcome.CANCELLING,
            EndOutcome.STOPPING,
        }


def marked_end(marked: Marked, pid: int, entry: Entry) -> End:
    """What the user's cancel of an entry a live run holds (`pid`) came to, by what marking
    it said (`TuneStore.mark_cancelled`): too late once that run saves its result."""
    if marked is Marked.TOO_LATE:
        return End(EndOutcome.SAVING, pid, entry=entry)
    if marked is Marked.GONE:  # (read under the same hold as the mark: kept for the type)
        return End(EndOutcome.GONE)
    return End(EndOutcome.CANCELLING, pid, entry=entry)


def left_words(end: End, name: str) -> str | None:
    """What an end nobody cancelled left, as a sentence from no process's side (the router
    writes it, the CLI prints it); None when it ended or nothing was queued."""
    if end.outcome is EndOutcome.RUNS:
        why = f"{runs_words(name, end.pid)}: it goes on there and ends its plan itself"
    elif end.outcome is EndOutcome.DOWNLOADING:
        why = (
            f"an approval of {name}'s quality tune is downloading its quants now (pid "
            f"{end.pid}): it queues the tune when they are in, or ends it"
        )
    elif end.outcome is EndOutcome.FAILED:
        why = f"the tune of {name} could not be ended: {end.why}"
    else:
        return None
    return f"{capital(why)}."


def runs_words(name: str, pid: int | None) -> str:
    """`a tune of tiny is running (pid 4242: …)`, from no process's own side (the router
    writes a refusal the CLI prints)."""
    return (
        f"a tune of {name} is running (pid {pid}: a terminal's `mnm tune --now`, the web UI or "
        "the router)"
    )


def stopping_words(name: str, words: str, pid: int) -> str:
    """The line a cancel says when it asks a live approval to stop."""
    return (
        f"Asked the approval of {words} (pid {pid}) to stop: it deletes what it downloaded once "
        "the file under way is in (a process stopped meanwhile, by Ctrl-Z say, does so when it "
        f"goes on; if it is gone, the router or another `mnm tune {name} --cancel` ends it)."
    )
