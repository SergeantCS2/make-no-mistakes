"""Which imported ComfyUI workflows mnm keeps, and why: a run of one going, a film or a song
that the user continues whose clips or song come from one, and a studio stage picked to be
played by one. `mnm clean imports` (and the web's Clean up) leaves them, and `mnm comfyui
workflow rm` (and the web's Remove) refuses them, saying why. Without that, a song waiting at
its lyrics review with its stage on `workflow:<name>` would lose its workflow to `mnm clean
imports`, and every later `mnm song new` would be refused."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from makenomistakes.pipeline.short_film import workflow_of as film_workflow
from makenomistakes.runs import RunKind, RunRecord, RunsRead
from makenomistakes.studio import stages


@dataclass(frozen=True)
class InUse:
    """The workflows in use, each with why; `unread`: the runs or the studio's stage picks
    could not be read (why), so any workflow may be one and every workflow is kept."""

    named: dict[str, str] = field(default_factory=dict)
    unread: str = ""

    def why(self, name: str) -> str | None:
        return self.named.get(name) or self.unread or None


def runs_using(records: Iterable[RunRecord]) -> dict[str, str]:
    """Each workflow a run uses now: a run of it going, or a film or a song going, waiting at
    a review or interrupted (`kept_for_owner`) whose clips or song come from it."""
    used: dict[str, str] = {}
    for run in records:
        if run.kind is RunKind.WORKFLOW and run.going:
            used.setdefault(run.target, f"its run {run.id} is going")
        elif run.kind is RunKind.PIPELINE and run.kept_for_owner:
            name = film_workflow(run.settings.get("video", "")) or stages.workflow_of(
                run.settings.get("song_instance")
            )
            if name:
                why = f"the {run.target} run {run.id} uses it (going, waiting or interrupted)"
                used.setdefault(name, why)
    return used


def in_use(runs: RunsRead, state_dir: Path) -> InUse:
    """`runs_using`, and the studio's stage picks (`mnm studio stage song fast=workflow:…`).
    A runs folder the system will not list keeps every workflow: any run may use one (read as
    no run in use, `mnm clean imports` would remove a waiting film's workflow and `mnm comfyui
    workflow rm` would refuse nothing)."""
    named = runs_using(runs.records)
    unread = f"{runs.unsure}, so a run may use it" if runs.unsure else ""
    try:
        picks = stages.read_picks(state_dir).picks
    except stages.StudioError as exc:
        why = f"the studio's stage picks cannot be read, so it may be one ({exc})"
        return InUse(named, unread or why)
    for stage, tiers in picks.items():
        for tier, pick in tiers.items():
            if (name := stages.workflow_of(pick)) is not None:
                named.setdefault(
                    name,
                    f"the {stage} stage's {tier} tier is played by it (`mnm studio stage {stage} "
                    f"{tier}=` picks another)",
                )
    return InUse(named, unread)


class WorkflowInUseError(RuntimeError):
    """A workflow mnm keeps, refused in words."""


def refuse_if_used(name: str, runs: RunsRead, state_dir: Path) -> None:
    """`mnm comfyui workflow rm` and the web's Remove: a workflow in use stays, said why."""
    why = in_use(runs, state_dir).why(name)
    if why is not None:
        raise WorkflowInUseError(f"{name} is in use: {why}. Nothing was removed.")
