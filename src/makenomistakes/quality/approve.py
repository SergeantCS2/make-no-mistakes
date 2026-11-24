"""Approving a quality tune: one path for `mnm tune --quality` and the web's
Optimise for quality.

Its queue entry is written first, under the tuning folder's lock with its plan's quants held
on the downloads ledger (`quality.entries.admit`: refused there, before anything downloads,
when another tune holds the instance's queue). Before that, in the same hold of the lock,
whatever entry the name holds is settled by the one rule (`tune.orphans`: a mark its run
outlived, of either kind), said once the lock is let go; a mark it cannot settle now refuses
the approval in its words. The `full` image is pulled and each candidate
claimed and downloaded (`claim`: under the lock, the entry still this approval's and the files
not here), the entry is made ready with the plan's files, and the router is asked to run it
by the plan's fingerprint (the router runs plans from disk, never one sent to it).
A failure, Ctrl-C or a cancel while it downloads ends the entry it wrote
(`quality.entries.end_plan`: what it downloaded goes); should that end fail, the approval's
lease is marked gone as it leaves (`release`), so no process reads it as downloading and the
router ends the entry at its next tick. Once ready, the plan is the router's: a refusal
leaves it queued (the router ends what it cannot run), and the same plan queued and ready
before this approval is joined and stays the earlier approval's."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from makenomistakes.instances import InstanceService
from makenomistakes.quality import QualityPlanError
from makenomistakes.quality.cache import found_here
from makenomistakes.quality.downloads import Downloads
from makenomistakes.quality.entries import (
    Admitted,
    OwnApproval,
    admit,
    end_plan,
    ready,
    release,
    still_mine,
)
from makenomistakes.quality.plan import QualityCandidate
from makenomistakes.quality.prepare import Claim, Offer, fetch
from makenomistakes.recipes import Recipe
from makenomistakes.router_client import SILENT, router_tune
from makenomistakes.tune.ends import left_words
from makenomistakes.tune.lease import going
from makenomistakes.tune.orphans import LEFT_MARKED, settle
from makenomistakes.tune.store import Entry, QualityQueued, TuneStore, TuneStoreError

Say = Callable[[str], None]
UNANSWERED = (
    "The router did not answer in time ({said}): the quality tune stays queued and the "
    "router's tuner runs it from its queue (`mnm tune --status`)."
)


@dataclass(frozen=True)
class Approved:
    entry: QualityQueued  # as queued: its plan with the files the downloads put on disk
    answer: tuple[int, str] | None  # the router's (a 2xx); None when no router took the ask
    waiting: str | None = None  # why it waits in the queue (the router silent), in words
    replaced: str | None = None  # a waiting speed tune it took the place of, in words


def approve(service: InstanceService, name: str, offered: Offer, now: bool, say: Say) -> Approved:
    """`offered`'s plan queued, held, fetched and made ready, the router asked;
    QualityPlanError says why not (nothing of this approval's left behind)."""
    store = TuneStore(service.settings.expanded_state_dir())
    downloads = Downloads(store)
    wanted = QualityQueued(name=name, now=now, reason="asked", quality=offered.plan)
    admitted: Admitted | None = None
    with going() as token:  # its lease is live while it approves (a router ends it after)
        try:
            told: list[str] = []  # what the settle said: said once the lock is let go, and
            try:  # said whether or not the approval then goes on
                with store.locked():  # the queue's word, the entry and its holds together
                    if (queued := store.queued(name)) is not None:  # whatever the name holds:
                        _settled(service, store, queued, told)  # its mark settled first
                    admitted = admit(store, wanted, token)
                    if not admitted.entry.ready:
                        downloads.hold(admitted.entry.holder, offered.plan)
            finally:
                for line in told:
                    say(line)
            entry = admitted.entry
            if not entry.ready:
                plan = fetch(service, offered, say, _claimer(service, store, downloads, entry))
                entry = ready(store, entry, plan)
        except BaseException as exc:  # a failure, a refusal, Ctrl-C: what it downloaded goes
            if admitted is not None and not admitted.entry.ready:  # its own entry
                _end_own(service, admitted.entry, say)
            if isinstance(exc, (TuneStoreError, OSError)):
                why = f"the quality tune of {name} was not queued: {exc}"
                raise QualityPlanError(why) from exc
            raise
        finally:  # an entry still under this approval (its end failed) holds nothing now
            if admitted is not None and not admitted.entry.ready:
                release(store, admitted.entry)
    fingerprint = entry.quality.fingerprint
    asked = {"action": "tune", "kind": "quality", "now": now, "fingerprint": fingerprint}
    answer = router_tune(service.settings, name, json.dumps(asked).encode())
    if answer is not None and answer[0] == SILENT:  # it may run it already: it stays queued
        return Approved(entry, None, UNANSWERED.format(said=answer[1]), admitted.replaced)
    if answer is not None and answer[0] >= 400:  # the router's refusal, in words
        raise QualityPlanError(_refused(store, entry, answer[1]))
    return Approved(entry, answer, None, admitted.replaced)


def _settled(service: InstanceService, store: TuneStore, queued: Entry, told: list[str]) -> None:
    """The mark of what the name holds, settled by the one rule (`tune.orphans`), whatever its
    kind; what to say of it goes into `told` (said once the lock is let go); QualityPlanError,
    in the settle's own words, for a mark it left."""
    settled = settle(service, store, queued)
    told.extend(settled.news)
    if settled.verdict in LEFT_MARKED:  # its line names the tune whose mark it is (a speed tune's)
        raise QualityPlanError(f"{settled.line} The quality tune approved now was not queued.")


def _end_own(service: InstanceService, entry: QualityQueued, say: Say) -> None:
    """The approval's own entry ends with what it downloaded; one another approval took over
    (or made ready) meanwhile is no longer its own to end."""
    end = end_plan(service, entry, say, OwnApproval(entry.approver))
    if (left := left_words(end, entry.name)) is not None:
        say(left)


def _claimer(
    service: InstanceService, store: TuneStore, downloads: Downloads, entry: QualityQueued
) -> Claim:
    """Before each candidate's download, under the store's lock: the entry still this
    approval's (CancelledError otherwise), then its files found here (the recipe with them),
    or the download claimed as a tune's own (never the instance's current quant)."""
    plan = entry.quality

    def claim(row: QualityCandidate) -> Recipe | None:
        with store.locked():
            still_mine(store, entry)
            found = found_here(service.models_dir, row.recipe)
            if service.files_present(found):
                return found
            if row.quant != plan.current:  # the instance's own stays the user's
                downloads.claim(entry.holder, plan.repo_id, row)
            return None

    return claim


def _refused(store: TuneStore, entry: QualityQueued, said: str) -> str:
    """The router's refusal, and whether the plan stays queued: the router ends a plan it
    cannot run itself; any other refusal (no tuner in an older router, a key, a failure)
    leaves it for the router's tuner, or for a cancel."""
    try:
        stays = store.current(entry) is not None
    except TuneStoreError:
        stays = False
    if not stays:
        return said
    return (
        f"{said} (the quality tune stays queued for the router's tuner; `mnm tune "
        f"{entry.name} --cancel` ends it with what it downloaded)"
    )
