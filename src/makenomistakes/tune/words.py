"""A queued tune in words, out of the records that hold it: what each kind is called
and does between two looks at its entry (`KindWords`, one row per kind), the states its
lifecycle marks (`closing`) as the user reads them in `mnm tune --status`, the Tuning tab, the
System page and the router's journal, and what `tune.orphans` made of a mark its run outlived.
The user's cancel words its own answers from these (`tune.cancel`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from makenomistakes.tune.records import AFTER_QUALITY, Entry, Holds, QualityQueued, TuneKind


@dataclass(frozen=True)
class KindWords:
    noun: str  # `tune`, `quality tune`: "the quality tune of tiny"
    unit: str  # what a run does between two looks at its entry: `trial`
    work: str  # what a cut stops: `the trial's copy`
    nothing: str  # what a cancelled one leaves: `its trials so far stay in its log`
    ending: str  # how a cancelled one ends once its run looks


WORDS: Final[dict[TuneKind, KindWords]] = {
    "speed": KindWords(
        noun="tune",
        unit="trial",
        work="the trial's copy",
        nothing="its trials so far stay in its log",
        ending="then it ends with nothing saved (its trials so far stay in its log)",
    ),
    "quality": KindWords(
        noun="quality tune",
        unit="step",
        work="its step's container",
        nothing="nothing changed",
        ending="then the plan ends with what it downloaded; nothing changes",
    ),
}


def of(entry: Entry) -> KindWords:
    return WORDS[entry.kind]


def capital(words: str) -> str:
    return f"{words[0].upper()}{words[1:]}"


def what(entry: Entry) -> str:
    """`the tune of tiny`, `the quality tune of tiny`; the speed tune a quality tune queued for
    its new quant says so (else a cancel of it reads as the quality tune's)."""
    if entry.kind == "speed" and entry.reason == AFTER_QUALITY:
        return f"the speed tune of {entry.name} that its quality tune queued"
    return f"the {of(entry).noun} of {entry.name}"


def named(entry: Entry) -> str:
    """The tune in a line about it: a quality plan names its quants."""
    return entry.words if isinstance(entry, QualityQueued) else what(entry)


def closing_words(entry: Entry, holder: Holds | None, *, unread: str | None) -> str | None:
    """A queued tune that an approval holds, or that its run is ending (cancelled, or saving
    its result), in words for `mnm tune --status`, the Tuning tab and the System page
    (`holder`: who holds it now, `TuneStore.holder`); None for one simply waiting. An approval
    first: it holds no run lease, so a plan whose approval is stopping never reads as one
    whose run stopped. `unread`, which every caller gives: why a mark it left cannot be
    judged now (`tune.orphans.unjudged`); no router's look settles it then, the user's cancel
    ends it (so the words never promise a router's look)."""
    if isinstance(entry, QualityQueued) and not entry.ready:
        if holder is None or holder.by != "approval":  # (a run holds no plan downloading)
            return f"approval stopped ({ENDED_BY.format(name=entry.name)})"
        pid = holder.pid
        return f"cancelled, stopping (pid {pid})" if entry.cancelled else f"downloading (pid {pid})"
    if entry.closing is None:
        return None
    if holder is not None:
        unit = of(entry).unit
        return f"cancelled, stopping after its {unit}" if entry.cancelled else "saving its result"
    if unread is not None:  # (`unjudged`: a save alone, never a cancel)
        return f"saving, its run stopped ({UNJUDGED.format(name=entry.name, why=reason(unread))})"
    return f"{entry.closing}, its run stopped ({SETTLED_BY.format(name=entry.name)})"


def downloading_words(entry: QualityQueued, pid: int) -> str:
    """Why nothing may be queued over a quality plan a live approval (`pid`) downloads."""
    if entry.cancelled:
        return (
            f"an approval of {entry.words} was cancelled and stops once the file it is "
            f"downloading is in (pid {pid}): ask again after that"
        )
    return (
        f"an approval of {entry.words} is downloading its quants now (pid {pid}): it queues "
        f"the tune when its downloads end (`mnm tune {entry.name} --cancel` stops it)"
    )


def blocking_words(entry: Entry, holder: Holds | None) -> str:
    """Why nothing may be queued over `entry` while its holder is ending it (a live run, or a
    cancelled approval finishing its file), or while it still carries the mark of a run that
    stopped (no holder: what a settle that could not end it leaves, since every way into the
    queue settles such a mark first, `tune.orphans`)."""
    if holder is None:
        left = "its cancel" if entry.cancelled else "an unfinished save"
        return (
            f"{capital(what(entry))} still carries {left} from a run that stopped: "
            f"{SETTLED_BY.format(name=entry.name)}"
        )
    if holder.by == "approval" and isinstance(entry, QualityQueued):
        return capital(downloading_words(entry, holder.pid))
    doing = (
        "is saving its result"
        if entry.saving
        else f"was cancelled and stops after the {of(entry).unit} under way"
    )
    return f"{capital(what(entry))} {doing} (pid {holder.pid}): ask again after that"


# --- what `tune.orphans` made of a mark its run outlived ----------------------------------------

SETTLED_BY = "a router's next look settles it, as does `mnm tune {name} --cancel`"
UNJUDGED = "whether its save landed cannot be read now ({why}): `mnm tune {name} --cancel` ends it"
ENDED_BY = "a router's next look ends it, as does `mnm tune {name} --cancel`"
SAVED_WHY = "its run stopped after it saved its result (`mnm tune --status` shows what it kept)"
UNSAVED_WHY = "its run stopped while it saved its result, so its next run saves it"
REMOVED_WHY = "its instance was removed"
HELD_WHY = "a live run or approval holds it and ends it itself"


def reason(problem: object) -> str:
    """A store's or the system's words set inside a sentence of ours: on one line, with no
    full stop of their own before our next word (never
    "…(the disk).): `mnm tune tiny --cancel` ends it")."""
    return " ".join(str(problem).split()).removesuffix(".")


def cancelled_why(entry: Entry) -> str:
    """A cancel completed: of a plan whose approval stopped mid-download, or of a run's."""
    holder = "approval" if isinstance(entry, QualityQueued) and not entry.ready else "run"
    return f"it was cancelled and its {holder} stopped"


def unsure_why(name: str, problem: str) -> str:
    """A save whose landing cannot be read now: left marked, never ended on a guess; the
    user's cancel ends it the conservative way (`tune.cancel`)."""
    return (
        f"its run stopped while it saved its result, and whether that save landed cannot be "
        f"read now ({reason(problem)}): it stays marked until it can, or `mnm tune {name} "
        "--cancel` ends it"
    )


def unread_waits(name: str, problem: str) -> str:
    """Why a queued tune waits while its instance's recipe cannot be read (a newer mnm's, a
    damaged file, the disk): said once by the router. Every other card job waits meanwhile
    too, since what holds the card cannot be listed (`tune.room.held`)."""
    return (
        f"its recipe cannot be read now ({reason(problem)}): it runs once it can, or `mnm tune "
        f"{name} --cancel` ends it"
    )


def failed_why(problem: str) -> str:
    """A settle the tuning folder kept from taking (its read, an unmark, an unqueue; a failed
    unmark is never worded "its end did not take"): whatever touches
    the tune next (a router's look, `mnm tune` asked again, a cancel) tries it again."""
    return (
        f"the tuning folder could not be read or written ({reason(problem)}), so whatever "
        "touches it next tries again"
    )


def replaced_words(entry: Entry, now: Entry | None) -> str:
    """A run that found its entry gone between the queue's read and its start: cancelled, or
    replaced by another tune (a quality plan's approval takes a waiting speed tune's place)."""
    if now is not None and now.kind != entry.kind:
        return f"{capital(what(entry))} was replaced by {named(now)}: not run."
    return f"{capital(what(entry))} was cancelled or replaced: not run."
