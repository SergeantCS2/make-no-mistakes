"""The finish of a quality tune: what each quant measured, the choice
(`quality.choose`), a reading per quant and the recipe saved at the chosen quant with its
record (`Recipe.quality`). Nothing moves when the instance changed since the plan or no quant
met the floor. The plan's end (`quality.entries.end_plan`) is the caller's, whatever happens
here."""

from __future__ import annotations

import contextlib
from datetime import datetime
from statistics import median

from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.machine import planned_here, stamped
from makenomistakes.quality.choose import Measured, choose, fastest
from makenomistakes.quality.plan import QualityCandidate, QualityPlan
from makenomistakes.quality.readings import write
from makenomistakes.quality.steps import Stage, Step, last_of, plan_moved
from makenomistakes.recipes import Recipe
from makenomistakes.recipes.model import OWNER_FIELDS
from makenomistakes.recipes.quality import QualityTuned
from makenomistakes.tune.trial_types import Outcome, TrialResult


class QualityMovedError(InstanceError):
    """The instance changed since its quality tune was planned: nothing is saved."""


def measured_rows(plan: QualityPlan, done: list[TrialResult]) -> list[Measured]:
    """Each quant the plan measured: its scores (the reference's from its base run) and its
    median decode at the serving plan (a spread trial's median too, said where it is kept)."""
    rows = []
    for row in plan.measured:
        scoring: Stage = "reference" if row.quant == plan.reference.quant else "quality"
        scored = last_of(done, Step(scoring, row))
        measured = scored is not None and scored.outcome is Outcome.MEASURED
        figures = scored.quality if measured and scored else None
        timed = last_of(done, Step("speed", row)) if row.serves else None
        spoke = timed is not None and timed.outcome in {Outcome.MEASURED, Outcome.NOISY}
        decode = median(timed.decode) if spoke and timed and timed.decode else None
        noisy = bool(timed and timed.outcome is Outcome.NOISY)
        build = scored.build if scored else None
        rows.append(Measured(row.quant, figures, decode, row.serves, noisy, build))
    return rows


def finish(
    service: InstanceService,
    plan: QualityPlan,
    done: list[TrialResult],
    made_for: datetime | None = None,
) -> tuple[bool, str, Recipe | None]:
    """(kept a quant, the line, the recipe saved); the plan's end (`end_plan`) is the
    caller's, whatever happens here. `made_for`: the queue entry's `queued_at`, stamped on
    the record (`QualityTuned.made_for`)."""
    name = plan.reference.recipe.name
    reference = last_of(done, Step("reference", plan.reference))
    if reference is None or reference.outcome is not Outcome.MEASURED:
        why = reference.why if reference is not None and reference.why else "it never ran"
        return (
            False,
            (
                f"The quality tune of {name} could not measure its reference "
                f"{plan.reference.quant} ({why}): nothing changed."
            ),
            None,
        )
    rows = measured_rows(plan, done)
    by_quant = {r.quant: r for r in plan.measured}
    for measured in rows:
        if (scored := measured.scored) is not None:
            with contextlib.suppress(Exception):  # a reading lost never fails the tune
                write(service, by_quant[measured.quant].recipe, scored, measured.build)
    chosen = choose(rows, plan.floor)
    if chosen is None:
        fast = fastest(rows)
        read = (
            f" (the fastest, {fast.quant}, read {fast.decode:.1f}{_spread(fast)})"
            if fast and fast.decode
            else ""
        )
        return (
            False,
            (
                f"The quality tune of {name}: none of the {len(rows)} quants measured reached the "
                f"{plan.floor:g} tok/s floor{read}: nothing changed (`mnm tune {name} --quality "
                "--floor N` sets another for one tune)."
            ),
            None,
        )
    row = by_quant[chosen.quant]
    scored = chosen.scored
    assert scored is not None  # `choose` keeps only a quant with its figures
    quality = QualityTuned(
        quant=chosen.quant,
        was=plan.current,
        figures=scored,
        floor=plan.floor,
        use=plan.use,
        measured=len(rows),
        build=chosen.build,
        made_for=made_for,
    )
    try:
        saved = save_choice(service, plan, row, quality)
    except InstanceError as exc:
        return (
            False,
            f"The quality tune of {name} chose {chosen.quant} but saved nothing: {exc}",
            None,
        )
    moved = (
        f"kept {chosen.quant}"
        if chosen.quant == plan.current
        else f"moved from {plan.current} (it loads {chosen.quant} from its next start)"
    )
    return (
        True,
        (
            f"Quality-tuned {name}: {quality.summary}{_spread(chosen)}; {moved}, {len(rows)} "
            f"quants measured on {scored.text}{_under_floor(rows, chosen, plan.floor)}."
        ),
        saved,
    )


def _under_floor(rows: list[Measured], chosen: Measured, floor: float) -> str:
    """The more faithful quants the floor left out, with what they read (a kept quant never
    hides a better one that read, say, 14.9 tok/s)."""
    drift = chosen.figures.drift if chosen.figures is not None else 0.0
    slower = [
        m for m in rows
        if m.serves and m.figures is not None and m.decode is not None and m.decode < floor
        and m.figures.drift < drift
    ]  # fmt: skip
    if not slower:
        return ""
    read = ", ".join(f"{m.quant} {m.decode:.1f} tok/s{_spread_word(m)}" for m in slower)
    return f" ({read}: more faithful, under the {floor:g} tok/s floor)"


def _spread_word(row: Measured) -> str:
    return " (spread)" if row.noisy else ""


def _spread(row: Measured) -> str:
    """A kept or quoted decode whose three readings spread past the tuner's 10%."""
    return (
        " (its speed readings spread: something else may have used the card; another "
        "`mnm tune --quality` measures it again)"
        if row.noisy
        else ""
    )


def save_choice(
    service: InstanceService, plan: QualityPlan, row: QualityCandidate, quality: QualityTuned
) -> Recipe:
    """The recipe at the chosen quant (its serving plan as planned, the user's own fields
    as they are now), or the same recipe with its record; refused when it moved since."""
    name = row.recipe.name
    here = planned_here(service.settings, service.memory())

    def swap(current: Recipe) -> Recipe:
        if plan_moved(current, plan):
            raise QualityMovedError(
                f"{name} changed since its quality tune was planned (a `mnm set` or a re-plan): "
                f"`mnm tune {name} --quality` plans it again"
            )
        if row.quant == current.quant:
            return current.model_copy(update={"quality": quality})
        fresh = row.recipe.model_copy(
            update={
                "display_name": current.display_name,
                "swap_group": current.swap_group,
                **{key: getattr(current, key) for key in OWNER_FIELDS},  # as they are now
                "evicted": None,
                "quality": quality,
                "tuned": None,  # measured on the old quant's files: the speed tune runs again
            }
        )
        return stamped(fresh, here)

    return service.change(name, swap)
