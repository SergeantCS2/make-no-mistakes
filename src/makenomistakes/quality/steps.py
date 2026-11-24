"""A quality tune's steps, replayed from the tune's trials log: the
reference's base run, then each candidate's speed trial and, but for the reference, its
KL-divergence run. Shared by the session that runs them and the finish that reads them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from makenomistakes.instances import InstanceService
from makenomistakes.measurements import settings_key
from makenomistakes.quality import WORK_FOLDER
from makenomistakes.quality.plan import QualityCandidate, QualityPlan
from makenomistakes.recipes import Recipe
from makenomistakes.tune.trial_types import Outcome, TrialResult

Stage = Literal["reference", "speed", "quality"]
Status = Literal["tuned", "cut", "held", "failed", "skipped", "cancelled"]
DONE = {Outcome.MEASURED, Outcome.CRASHED, Outcome.REFUSED, Outcome.NOISY}
NOISY_TRIES = 2  # a speed trial whose readings spread runs once more; the last one counts


@dataclass(frozen=True)
class Step:
    stage: Stage
    row: QualityCandidate

    @property
    def words(self) -> str:
        return {
            "reference": f"{self.row.quant}, the reference",
            "speed": f"{self.row.quant}'s speed",
            "quality": f"{self.row.quant}'s quality",
        }[self.stage]


def steps_of(plan: QualityPlan) -> list[Step]:
    steps = [Step("reference", plan.reference)]
    for row in plan.candidates:
        steps.append(Step("speed", row))
        if row.quant != plan.reference.quant:
            steps.append(Step("quality", row))
    return steps


def _whole(done: list[TrialResult], step: Step) -> list[TrialResult]:
    """The step's whole results in the session, oldest first (a cut or held one is not)."""
    return [
        r for r in done
        if r.candidate.quant == step.row.quant and r.candidate.stage == step.stage
        and r.outcome in DONE
    ]  # fmt: skip


def last_of(done: list[TrialResult], step: Step) -> TrialResult | None:
    found = _whole(done, step)
    return found[-1] if found else None


def _settled(done: list[TrialResult], step: Step) -> bool:
    """A step with its result: a spread speed trial only once it has run `NOISY_TRIES` times
    (something else may have used the card the first time)."""
    found = _whole(done, step)
    if not found:
        return False
    noisy = step.stage == "speed" and found[-1].outcome is Outcome.NOISY
    return not noisy or len(found) >= NOISY_TRIES


def next_step(steps: list[Step], done: list[TrialResult], base_here: bool) -> Step | None:
    """The first step without its result; the reference again when its base file went. A
    quant whose speed trial crashed or was refused does not serve here and is never scored."""
    reference = last_of(done, steps[0])
    if reference is not None and reference.outcome is Outcome.MEASURED and not base_here:
        return steps[0]
    for step in steps:
        timed = last_of(done, Step("speed", step.row)) if step.stage == "quality" else None
        if timed is not None and timed.outcome in {Outcome.CRASHED, Outcome.REFUSED}:
            continue
        if not _settled(done, step):
            return step
    return None


def work_folder(service: InstanceService, name: str) -> Path:
    return service.models_dir / WORK_FOLDER / name


@dataclass(frozen=True)
class Ended:
    """How a quality tune ended, for the tuner's `TuneEnd`."""

    status: Status
    line: str


def plan_moved(recipe: Recipe, plan: QualityPlan) -> bool:
    """The instance changed since the plan (a `mnm set`, a re-plan): its plan no longer holds."""
    return settings_key(recipe) != plan.settings


def refusal(service: InstanceService, recipe: Recipe, plan: QualityPlan) -> str | None:
    """Why the router cannot run `plan` now, in words: the instance moved since it was planned,
    or a quant's files are no longer on disk."""
    name = recipe.name
    if plan_moved(recipe, plan):
        return (
            f"{name} changed since its quality tune was planned: `mnm tune {name} --quality` "
            "plans it again"
        )
    missing = [row.quant for row in plan.measured if not service.files_present(row.recipe)]
    if missing:
        return f"{name}'s quality tune is missing files of {', '.join(missing)}"
    return None
