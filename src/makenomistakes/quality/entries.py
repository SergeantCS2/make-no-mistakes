"""A quality tune's queue entry from its approval to its end: written before
anything downloads, under the tuning folder's lock with its holds on the downloads ledger,
and ended by one function, `end_plan`, whoever ends it.

- `admit`: the queue's word first (another tune holds it, or a live run is ending it:
  refused in words, nothing written). The same plan queued and ready is joined (`now`
  merged); the same plan whose approval stopped mid-download is taken over (this approval
  goes on from what is here); one a live approval downloads is refused. An entry carrying a
  mark is never joined: the approval settles whatever the name holds first (`tune.orphans`),
  and `blocking` refuses a mark still there. Else a new entry, under this approval's lease
  (`approver`).
- `still_mine`: the entry as queued now while it is still this approval's (the same entry,
  under this very approval, not marked cancelled), checked before each download is claimed
  and once they all landed.
- `ready`: the downloads done, the plan with its files, if still this approval's.
- `end_plan`: judged under the lock, never by its caller beforehand (a cancel that checked
  first could end a run that began meanwhile, which would then save): the entry still queued,
  no live run of it but the asker's own, no live approval of it but the asker's (who asks is
  one typed value, `Asker`: the plan's own run, its own approval, the user's cancel, or the
  router as a bystander). The user's cancel marks an entry a live run
  or approval holds (`cancelled`: that run or approval ends it at its next look) unless the
  run is saving its result (`saving`: too late). Otherwise the entry is unqueued, its holds
  released in one write, its own state dropped, and what nobody holds that a tune made its
  own deleted (`sweep`). What came of it is a `tune.ends.End`; it never raises for the store
  or the disk.
- `end_saved`: a plan whose result is saved ends through `end_plan`, and when its save moved
  the instance to another quant a speed tune is queued for the new quant's settings: what
  every saved plan leaves, whether its run's finish ends it or `tune.orphans` does for a run
  that died once its save landed.
- `sweep`: releases holds whose entry is gone (an end cut short after its unqueue) and
  deletes what is pending, at every end and every router tick; what it could not do comes
  back in words (`Swept.problems`), for the caller to say once."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Collection
from dataclasses import dataclass, field

from makenomistakes.comfy.models import ModelError, Models
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.model_cleanup import delete_own_files
from makenomistakes.quality import QualityPlanError
from makenomistakes.quality.cache import found_here
from makenomistakes.quality.downloads import Downloads
from makenomistakes.quality.plan import QualityPlan
from makenomistakes.quality.steps import work_folder
from makenomistakes.tune.ends import End, EndOutcome, marked_end, stopping_words
from makenomistakes.tune.store import (
    AFTER_QUALITY,
    Approval,
    Holds,
    QualityQueued,
    Queued,
    TuneStore,
    TuneStoreError,
)
from makenomistakes.tune.words import downloading_words

Say = Callable[[str], None]


class CancelledError(QualityPlanError):
    """The approval's entry was cancelled (or ended) while it downloaded."""


@dataclass(frozen=True)
class Admitted:
    entry: QualityQueued  # as queued now
    replaced: str | None  # a waiting speed tune it took the place of, in words


@dataclass(frozen=True)
class Swept:
    """What a sweep deleted, per (repo, quant), and what it could not do, in words."""

    freed: dict[tuple[str, str], int] = field(default_factory=dict)
    problems: tuple[str, ...] = ()

    @property
    def total(self) -> int:
        return sum(self.freed.values())


def admit(store: TuneStore, wanted: QualityQueued, token: str) -> Admitted:
    """The approval's entry, written before anything downloads, under the approval's lease
    (`token`, going for the whole approval); the caller holds the store's lock and holds its
    quants on the ledger under it. QualityPlanError says why not."""
    if (why := store.blocking(wanted)) is not None:
        raise QualityPlanError(why)
    before = store.queued(wanted.name)
    now = wanted.now
    if isinstance(before, QualityQueued):  # the same plan: `blocking` let only that through
        held = store.holder(before)
        if held is not None and held.by == "approval":  # a live approval downloads it now
            raise QualityPlanError(downloading_words(before, held.pid))
        # A mark is never joined: the approval settled it first (`tune.orphans`), and
        # `blocking` refuses one still there (a live run's, or one its settle left).
        joined = before.model_copy(update={"now": now or before.now})
        if not before.ready:  # its approval stopped mid-download: this one goes on
            joined = _leased(joined, token)
        store.put(joined)
        return Admitted(joined, None)
    replaced = (
        f"it takes the place of {wanted.name}'s waiting speed tune"
        if isinstance(before, Queued)
        else None
    )
    fresh = store.put_new(_leased(wanted, token))  # dated under the lock, after the settle
    return Admitted(fresh, replaced)


def _leased(entry: QualityQueued, token: str) -> QualityQueued:
    return entry.model_copy(update={"approver": Approval.here(token)})


def still_mine(store: TuneStore, entry: QualityQueued) -> QualityQueued:
    """The entry as queued now, while it is still under this approval (`entry.approver`) and
    not marked cancelled; CancelledError otherwise."""
    found = store.current(entry)
    mine = found is not None and entry.approver is not None and entry.approver.same(found.approver)
    if found is None or not mine or found.cancelled:
        raise CancelledError(f"the quality tune of {entry.name} was cancelled while it downloaded")
    return found


def release(store: TuneStore, entry: QualityQueued) -> None:
    """An approval that ends without having ended its entry (that end failed) marks its lease
    gone, so no process reads it as downloading (the router ends the entry as orphaned at its
    next tick); nothing when the entry is no longer this approval's."""
    try:
        with store.locked():
            found = store.current(entry)
            ours = entry.approver
            if found is None or ours is None or not ours.same(found.approver):
                return
            store.put(found.model_copy(update={"approver": ours.model_copy(update={"gone": True})}))
    except (TuneStoreError, OSError):  # the store itself: the router's tick tries again
        return


def ready(store: TuneStore, entry: QualityQueued, plan: QualityPlan) -> QualityQueued:
    """The downloads done: the entry ready with the plan as fetched, if still this
    approval's (CancelledError otherwise)."""
    with store.locked():
        found = still_mine(store, entry)
        done = found.model_copy(update={"quality": plan, "approver": None})
        store.put(done)
    return done


@dataclass(frozen=True)
class OwnRun:
    """The run ending its own plan (its token, going in this process: `tune.lease.going`)."""

    token: str


@dataclass(frozen=True)
class OwnApproval:
    """The approval ending its own entry, under its lease."""

    lease: Approval | None


@dataclass(frozen=True)
class OwnerCancel:
    """The user's cancel: an entry a live run or approval holds is marked cancelled (that run
    or approval ends it at its next look), unless the run is saving its result."""


@dataclass(frozen=True)
class Bystander:
    """The router ending a plan nothing will run, or one it cannot run."""


Asker = OwnRun | OwnApproval | OwnerCancel | Bystander


@dataclass(frozen=True)
class Saved:
    """What the finish saved: the quant the save moved the instance away from (None: it kept
    the quant it had)."""

    moved_from: str | None


def end_plan(
    service: InstanceService,
    entry: QualityQueued,
    say: Say,
    asker: Asker,
    saved: Saved | None = None,
) -> End:
    """`entry`'s plan ends, if it may, judged here under the tuning folder's lock: not while a
    live run of it holds its state but the asker's own (`OwnRun`) or a live approval downloads
    its quants but the asker's (`OwnApproval`): the user's cancel marks it for that run or
    approval to end (`OwnerCancel`), any other end leaves it to them. Then the entry is
    unqueued first (once out, nothing runs it again), its holds released in one write (with
    `--keep` every quant it holds the user's; else the quant a save moved the instance away
    from a tune's own), its own state dropped, and what is pending deleted (`sweep`). A step
    after the unqueue that fails is said and left to the next end or the router's sweep; safe
    to call again."""
    store = TuneStore(service.settings.expanded_state_dir())
    lines: list[str] = []  # said once the lock is let go (a terminal stopped mid-line holds it)
    try:
        with store.locked():
            found = store.current(entry)
            if found is None or not _still_its(asker, found):
                end = End(EndOutcome.GONE)  # ended, replaced, or no longer this approval's
                # an end cut short after its unqueue left its holds: released and deleted here
                lines.extend(_swept_words(sweep(service, store), set(), saved))
            elif (held := _held(store, found, asker)) is None:
                end = _ended(service, store, found, saved, lines)
            elif held.by == "run":
                end = _marked(store, found, held.pid, asker)
            elif isinstance(asker, OwnerCancel):
                store.mark_cancelled(found)  # an approval saves nothing: never too late
                lines.append(stopping_words(found.name, found.words, held.pid))
                end = End(EndOutcome.STOPPING, held.pid, entry=found)
            else:
                end = End(EndOutcome.DOWNLOADING, held.pid, entry=found)
    except (TuneStoreError, OSError) as exc:  # the entry itself, read or removed
        end = End(EndOutcome.FAILED, why=str(exc))
    for line in lines:
        say(line)
    return end


def end_saved(
    service: InstanceService,
    entry: QualityQueued,
    say: Say,
    asker: Asker,
    moved_from: str | None,
) -> End:
    """`entry`'s plan, whose result is saved, ends (`end_plan` with what the save moved the
    instance away from); once it ended and the quant moved, the new quant's settings are
    queued for a speed tune (`AFTER_QUALITY`; its own end leaves nothing to settle there).
    Like `end_plan`, it never raises once the entry is out: a follow-up kept out is a line."""
    end = end_plan(service, entry, say, asker, Saved(moved_from))
    moved = end.outcome is EndOutcome.ENDED and moved_from is not None
    if moved and (left := _follow_up(service, entry.name)) is not None:
        say(left)
    return end


def _follow_up(service: InstanceService, name: str) -> str | None:
    """The new quant's speed tune, queued; what kept it out, in words, never raised (the plan's
    end took: a full disk here must never read as an end that did not)."""
    store = TuneStore(service.settings.expanded_state_dir())
    follow_up = Queued(name=name, reason=AFTER_QUALITY)
    try:
        if store.queue(follow_up) is not None:
            return None
        why = store.blocking(follow_up) or "the queue refused it"
    except (TuneStoreError, OSError) as exc:
        why = str(exc)
    return (
        f"The speed tune of {name}'s new quant was not queued ({why}): `mnm tune {name}` asks "
        "for it."
    )


def _marked(store: TuneStore, found: QualityQueued, pid: int, asker: Asker) -> End:
    """A live run holds `found`: the user's cancel marks it (the run ends it at its next
    look), too late once that run saves its result; any other end leaves it to that run."""
    if not isinstance(asker, OwnerCancel):
        return End(EndOutcome.RUNS, pid, entry=found)
    return marked_end(store.mark_cancelled(found), pid, found)


def _still_its(asker: Asker, found: QualityQueued) -> bool:
    """An approval ends only the entry it still holds (not one taken over, nor made ready)."""
    return not isinstance(asker, OwnApproval) or (
        asker.lease is not None and asker.lease.same(found.approver)
    )


def _held(store: TuneStore, found: QualityQueued, asker: Asker) -> Holds | None:
    """Who holds `found` but the asker (`TuneStore.holder`, the one answer: the plan's own run
    or approval asking is never its own holder)."""
    token = asker.token if isinstance(asker, OwnRun) else None
    lease = asker.lease if isinstance(asker, OwnApproval) else None
    return store.holder(found, token, lease)


def _ended(
    service: InstanceService,
    store: TuneStore,
    found: QualityQueued,
    saved: Saved | None,
    lines: list[str],
) -> End:
    """The end itself, under the lock: unqueued first (once it is out nothing runs it again;
    a failure here is the caller's FAILED), then each later step's failure said, not raised."""
    store.unqueue(found.name, found)
    moved_from = saved.moved_from if saved is not None else None
    left = _released(Downloads(store), found, moved_from, lines)
    _drop_state(store, found, lines)
    shutil.rmtree(work_folder(service, found.name), ignore_errors=True)  # its base file
    lines.extend(_swept_words(sweep(service, store), left, saved))
    return End(EndOutcome.ENDED, entry=found)


def _swept_words(swept: Swept, left: set[tuple[str, str]], saved: Saved | None) -> list[str]:
    """What the sweep could not do, then what it freed: this end's own (`left`), then what
    earlier ends left."""
    lines = list(swept.problems)
    mine = sum(freed for key, freed in swept.freed.items() if key in left)
    if mine:
        what = "the quants not chosen" if saved is not None else "what the quality tune downloaded"
        lines.append(f"Deleted {what}: {mine / 2**30:.1f} GiB freed.")
    if swept.total > mine:
        earlier = (swept.total - mine) / 2**30
        lines.append(f"Deleted what an earlier quality tune left: {earlier:.1f} GiB freed.")
    return lines


def _released(
    downloads: Downloads, entry: QualityQueued, moved_from: str | None, problems: list[str]
) -> set[tuple[str, str]]:
    try:
        return downloads.end(entry.holder, moved_from)
    except (TuneStoreError, OSError) as exc:
        problems.append(
            f"The downloads ledger could not be read or written ({exc}): the next end, or the "
            f"router's sweep, releases what {entry.words} held."
        )
        return set()


def _drop_state(store: TuneStore, entry: QualityQueued, problems: list[str]) -> None:
    try:
        store.drop_own_state(entry)
    except OSError as exc:
        problems.append(f"{entry.name}'s tune progress could not be removed: {exc}")


def sweep(service: InstanceService, store: TuneStore) -> Swept:
    """Under the store's lock: release every hold whose entry is gone, then delete every
    pending quant's files where nothing else lists them (an approval holding one meanwhile
    keeps it). It never raises: a quant whose files could not be deleted, or a ledger or a
    list it could not read, stays pending and is said in `Swept.problems`."""
    downloads = Downloads(store)
    try:
        with store.locked():
            queued = store.queued_holders()
            if queued is not None:  # an entry it cannot read: none of its holds looks orphaned
                downloads.release_orphans(queued)
            pending = downloads.pending()
            if not pending:
                return Swept()
            try:
                others = service.all()
                listing = Models(service.models_dir, service.settings.expanded_state_dir())
                linked = frozenset(link.target for link in listing.ledger().links)
            except (InstanceError, ModelError) as exc:
                return Swept(problems=(f"{KEPT_UNREAD}: {exc}",))
            freed: dict[tuple[str, str], int] = {}
            problems = []
            for hold in pending:
                recipe = found_here(service.models_dir, hold.recipe)  # a download cut short too
                try:
                    deleted = delete_own_files(recipe, others, service.models_dir, linked)
                except OSError as exc:
                    problems.append(
                        f"{hold.quant}'s files could not all be deleted (tried again later): {exc}"
                    )
                    continue
                freed[(hold.repo_id, hold.quant)] = deleted.freed_bytes
            problems.extend(_forgotten(downloads, freed.keys()))
            return Swept(freed, tuple(problems))
    except (TuneStoreError, OSError) as exc:
        return Swept(problems=(f"{LEDGER_UNREAD}: {exc}",))


KEPT_UNREAD = "The quants quality tunes downloaded are kept until mnm can read what lists them"
LEDGER_UNREAD = "What quality tunes downloaded stays until mnm can read the tuning folder"


def _forgotten(downloads: Downloads, done: Collection[tuple[str, str]]) -> list[str]:
    try:
        downloads.forget(done)
    except (TuneStoreError, OSError) as exc:
        return [f"The downloads ledger could not be written after the deletes: {exc}"]
    return []
