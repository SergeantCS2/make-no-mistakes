"""A set's members as a page or table shows them, and the rules read over them: the verdict
when no sum can be trusted, a swap set's, the start order (`workflows` re-exports these
names)."""

from __future__ import annotations

from dataclasses import dataclass

from makenomistakes.capacity import FitNow, Verdict, VramEstimate, streaming_member
from makenomistakes.instance_types import InstanceError, InstanceStatus
from makenomistakes.recipes.model import Recipe


class WorkflowError(InstanceError):
    """A workflow problem worded for the user (the CLI and the web print it alike)."""


@dataclass(frozen=True)
class Member:
    """One member as a page or table shows it. An instance removed out from under the
    workflow is a member state (`missing`), never a failed page: the user still sees the
    workflow, can remove it, and can stop the members that exist."""

    role: str
    name: str
    status: InstanceStatus | None  # None when the instance is gone
    verdict: Verdict
    estimate: VramEstimate | None = None

    @property
    def missing(self) -> bool:
        return self.status is None

    @property
    def running(self) -> bool:
        return self.status is not None and self.status.running

    @property
    def recipe(self) -> Recipe:
        if self.status is None:
            raise WorkflowError(f"the {self.role} '{self.name}' is gone")
        return self.status.recipe


GONE = Verdict(level=FitNow.UNKNOWN, note="This instance was removed; `mnm add` it again.")


def docker_error(members: list[Member]) -> str | None:
    """The first member's Docker error: the listing's one line about it."""
    return next((m.status.error for m in members if m.status and m.status.error), None)


def _why_unknown(members: list[Member]) -> Verdict | None:
    """The verdict when no sum can be trusted: a member gone, Docker not answering (nothing
    is guessed to be stopped), or a waiting member without an estimate (an unreadable GPU is
    `capacity.fit_together`'s to say)."""
    gone = [m for m in members if m.missing]
    if gone:
        names = ", ".join(f"{m.name} ({m.role})" for m in gone)
        return Verdict(level=FitNow.UNKNOWN, note=f"Removed instance: {names}.")
    error = docker_error(members)
    if error is not None:
        return Verdict(level=FitNow.UNKNOWN, note=f"Docker could not be asked: {error}")
    for m in members:
        if not m.running and (m.estimate is None or m.estimate.bytes is None):
            basis = m.estimate.basis if m.estimate else "no estimate"
            return Verdict(level=FitNow.UNKNOWN, note=f"{m.name}: {basis}.")
    return None


def _taking_turns_together(members: list[Member]) -> Verdict:
    """A swap workflow never needs its members side by side: each must fit once its
    group-mates are parked, which is each member's own verdict. An *on request* member is
    stuck here too: a request through the router would park the outsider holding the card,
    but a workflow start parks group-mates only."""
    stuck = next((m for m in members if m.verdict.warns), None)
    if stuck is not None:
        return Verdict(level=FitNow.NO_FIT, note=f"{stuck.name}: {stuck.verdict.note}")
    return Verdict(
        level=FitNow.TAKES_TURNS,
        note="The members take turns on the GPU: each loads when asked, parking the others.",
    )


def start_order(members: list[Member]) -> list[Member]:
    """The members that are not running, largest estimate first (what cannot fit fails before
    the small ones take its room), the one that streams its weights last: it takes the VRAM
    that is left (`capacity.streaming_member`, the member the verdict priced at its floor)."""
    waiting = [m for m in members if not m.running]
    streams = streaming_member([(m.name, m.estimate) for m in waiting if m.estimate])
    return sorted(
        waiting,
        key=lambda m: (m.name == streams, -((m.estimate.bytes if m.estimate else None) or 0)),
    )
