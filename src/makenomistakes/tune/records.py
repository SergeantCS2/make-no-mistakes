"""What the tuning folder holds and what the router's tuner is asked: a queued
tune (`Queued`; `QualityQueued` with its approval's lease, `Approval`), the tune in progress
(`TuneState`), a tune that gave up (`GaveUp`) and `POST /__mnm/tune/<name>`'s body
(`TuneAsk`). `tune.store` reads and writes them behind the folder's lock and re-exports them.

A queued tune carries one mark a live run (or approval) holding it reads at every step
(`closing`; a cancel kept in a cut's last word would be lost to the next request, or to a
trial held for room): `cancelled`, the user's cancel, written when a
live run or approval holds the entry (that run ends it at its next look, nothing runs it
again), or `saving`, written by that run before it saves its result (a cancel from then on is
too late, from any process). `tune.store` makes every transition under the folder's lock, and
`tune.orphans` decides what a mark whose run is gone becomes."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, TypeAdapter

from makenomistakes.quality.plan import QualityPlan
from makenomistakes.recipes.tuned import ScoreMode
from makenomistakes.tune.lease import Lease

DEFAULT_MINUTES = 45
AFTER_QUALITY = "quality tune"  # a speed tune's reason when a quality tune moved the quant

TuneKind = Literal["speed", "quality"]  # The settings for speed, or the quant
Closing = Literal["cancelled", "saving"]  # the run holding an entry is ending it: why


class _Waiting(BaseModel):
    """What every queued tune carries."""

    schema_version: Literal[1] = 1
    name: str
    mode: ScoreMode | None = None  # None: the instance type's default
    minutes: int = Field(default=DEFAULT_MINUTES, ge=1, le=24 * 60)
    now: bool = False  # `--now`: no idle wait for its first trial
    reason: str = "added"  # added, asked, another build, …
    queued_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    closing: Closing | None = None  # the run (or approval) holding it is ending it

    @property
    def cancelled(self) -> bool:
        """The user cancelled it while a live run or approval held it."""
        return self.closing == "cancelled"

    @property
    def saving(self) -> bool:
        """The run holding it saves its result now: a cancel is too late."""
        return self.closing == "saving"


class Queued(_Waiting):
    """A speed tune waiting for an idle card."""

    kind: Literal["speed"] = "speed"

    def owns(self, state: TuneState) -> bool:
        """`state` is this entry's tune in progress."""
        return state.kind == "speed"


class Approval(Lease):
    """The approving call's lease on a quality entry while it downloads (its token going for
    the whole approval, `tune.lease.going`); `gone`: the approval ended without ending its
    entry (that end failed), so it holds nothing for any process (the router ends the entry
    as orphaned)."""

    gone: bool = False

    @property
    def alive(self) -> bool:
        return not self.gone and super().alive


class QualityQueued(_Waiting):
    """A quality tune waiting: the plan the user approved, written by the CLI or the
    web's job into the folder the router reads (never sent over HTTP: the router runs exactly
    what is on disk). Written before its quants download, under the approving
    call's lease (`approver`) while they do (the router leaves it alone, and ends it when that
    process is gone); ready once they are here (no approver). Fields this mnm does not know
    (an earlier shape, a newer mnm's) make it unreadable rather than read as ready."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["quality"] = "quality"
    quality: QualityPlan
    approver: Approval | None = None  # set while its approval downloads its quants

    @property
    def words(self) -> str:
        """`tiny's quality tune (3 quants against Q8_0)`."""
        plan = self.quality
        count, reference = len(plan.measured), plan.reference.quant
        return f"{self.name}'s quality tune ({count} quants against {reference})"

    @property
    def holder(self) -> str:
        """Its name on the downloads ledger: two entries of one instance never share one."""
        return f"{self.name}@{self.queued_at.isoformat()}"

    @property
    def ready(self) -> bool:
        """Its quants are here: the router may run it."""
        return self.approver is None

    def owns(self, state: TuneState) -> bool:
        """`state` is this entry's tune in progress (by its holder: never another entry's)."""
        return state.kind == "quality" and state.holder == self.holder


def _kind(value: object) -> str:
    """An entry's kind; one written by an older mnm has none and is a speed tune."""
    found = value.get("kind") if isinstance(value, dict) else getattr(value, "kind", None)
    return str(found or "speed")


Entry = Queued | QualityQueued
ENTRY: TypeAdapter[Entry] = TypeAdapter(
    Annotated[
        Annotated[Queued, Tag("speed")] | Annotated[QualityQueued, Tag("quality")],
        Discriminator(_kind),
    ]
)


class Marked(StrEnum):
    """What the user's cancel did to an entry a live run or approval holds
    (`TuneStore.mark_cancelled`)."""

    MARKED = "marked"  # cancelled now (or before): its holder ends it at its next look
    TOO_LATE = "too late"  # its run saves its result already
    GONE = "gone"  # nothing of it is queued any more


@dataclass(frozen=True)
class Holds:
    """Who holds a queued tune now and ends it itself (`TuneStore.holder`): a live run of its
    tune, or a live approval downloading a quality plan's quants (which holds no run lease)."""

    by: Literal["run", "approval"]
    pid: int


def same_plan(before: QualityQueued, entry: Entry) -> bool:
    return (
        isinstance(entry, QualityQueued) and entry.quality.fingerprint == before.quality.fingerprint
    )


def same_entry(found: Entry | None, entry: Entry) -> bool:
    """The queued entry is `entry` itself (its kind and when it was queued)."""
    return found is not None and found.kind == entry.kind and found.queued_at == entry.queued_at


class TuneAsk(BaseModel):
    """`POST /__mnm/tune/<name>`'s body: what `mnm tune` asks the router's tuner. A quality
    tune is in the queue already (its plan written there by the asker); the ask names it by
    its fingerprint and nothing else."""

    action: Literal["tune", "cancel"] = "tune"
    mode: ScoreMode | None = None
    minutes: int = Field(default=DEFAULT_MINUTES, ge=1, le=24 * 60)
    now: bool = False
    by: Literal["owner", "nightly"] = "owner"  # the nightly never re-asks one that gave up
    kind: TuneKind = "speed"
    fingerprint: str | None = Field(default=None, max_length=64)  # The plan approved


class GaveUp(BaseModel):
    """A tune that ended without a result: why, on which settings, when."""

    schema_version: Literal[1] = 1
    name: str
    why: str  # `decode readings spread 55% over 3 (the rule: 10%)`
    settings: str  # `measurements.settings_key` of the recipe it ran
    trials: int = 0
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TuneState(BaseModel):
    schema_version: Literal[1] = 1
    name: str
    session: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    mode: ScoreMode
    minutes: int = Field(ge=1)
    settings: str  # `measurements.settings_key` of the recipe it started from
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    running_pid: int | None = None  # the process running it now (either kind: the whole tune)
    running_since: str | None = None  # that process's identity (`run_token`): a pid reused
    running_token: str | None = None  # the run in that process (`tune.lease.going`)
    trial: int = 0  # the trial running (or last run)
    max_trials: int = 24
    seconds_left: float | None = None
    cuts: int = 0  # times an outside request took the card back
    machine: str | None = None  # The box its trials ran on (None: an older state)
    kind: TuneKind = "speed"
    step: str | None = None  # The quality run under way (`Q5_K_M's speed`)
    plan: str | None = None  # The quality plan's fingerprint (a re-asked plan starts over)
    holder: str | None = None  # The queue entry it runs (`QualityQueued.holder`)

    @property
    def lease(self) -> Lease | None:
        """The run holding it now, as a lease (its process and its token)."""
        if not self.running_pid:
            return None
        return Lease(pid=self.running_pid, since=self.running_since, token=self.running_token)

    def run_beside(self, token: str | None) -> int | None:
        """The pid of the live run holding it, unless that run is the caller's own (`token`):
        a run in this very process is another run too."""
        lease = self.lease
        if lease is None or not lease.alive or (token is not None and lease.token == token):
            return None
        return lease.pid

    @property
    def line(self) -> str:
        """`tuning tiny (trial 7 of ≤24, 31 min left)`; a quality tune's `tuning tiny for
        quality (step 3 of 7: Q5_K_M's speed)`."""
        if self.kind == "quality":
            doing = f": {self.step}" if self.step else ""
            return f"tuning {self.name} for quality (step {self.trial} of {self.max_trials}{doing})"
        left = f", {self.seconds_left / 60:.0f} min left" if self.seconds_left is not None else ""
        return f"tuning {self.name} (trial {self.trial} of ≤{self.max_trials}{left})"
