"""A quality tune's plan in words: one wording for `mnm tune --quality` and
the web's Optimise for quality page."""

from __future__ import annotations

from dataclasses import dataclass

from makenomistakes.quality.plan import QualityCandidate, QualityPlan
from makenomistakes.quality.prepare import GIB, Offer
from makenomistakes.quality.texts import TEXTS

FLOOR_SETTING = {"chat": "MNM_FLOOR_CHAT", "agents": "MNM_FLOOR_AGENT"}


@dataclass(frozen=True)
class PlanRow:
    """One quant the plan measures, as the table shows it."""

    quant: str
    role: str  # reference, current, candidate
    size: str
    here: bool  # on disk already
    decode: str  # the planner's decode at its serving plan, "-" when it does not serve


def role_of(plan: QualityPlan, row: QualityCandidate) -> str:
    roles = []
    if row.quant == plan.reference.quant:
        roles.append("reference" if row.serves else "reference (measured only)")
    if row.quant == plan.current:
        roles.append("current")
    return ", ".join(roles) or "candidate"


def plan_rows(offered: Offer) -> list[PlanRow]:
    planned = {f.quant: f.decode for f in offered.facts}
    rows = []
    for row in offered.plan.measured:
        decode = planned.get(row.quant) if row.serves else None
        rows.append(
            PlanRow(
                quant=row.quant,
                role=role_of(offered.plan, row),
                size=f"{row.size_bytes / GIB:.1f} GiB" if row.size_bytes else "-",
                here=row.quant in offered.on_disk,
                decode=f"~{decode:.0f} tok/s" if decode else "-",
            )
        )
    return rows


def plan_words(offered: Offer) -> list[str]:
    """What the tune measures, what it keeps, what it downloads and deletes."""
    plan = offered.plan
    text = TEXTS[plan.text]
    use = "coding" if plan.use == "agents" else "chat"
    measures = (
        f"Each quant is scored against {plan.reference.quant} on {text.words}, {plan.chunks} "
        "chunks of 512 tokens (KL divergence, the same top token, perplexity), and its decode is "
        f"measured at its serving plan. The most faithful quant at or over {plan.floor:g} tok/s is "
        f"kept (the {use} floor, {FLOOR_SETTING[plan.use]}; `--floor` sets another for this tune); "
        "when none reaches it, nothing changes."
    )
    fetch = []
    if offered.fetch_bytes:
        fetch.append(f"downloads {offered.fetch_bytes / GIB:.1f} GiB")
    if not offered.image_here:
        fetch.append(f"pulls {offered.image}")
    got = f"It {' and '.join(fetch)}" if fetch else "Every quant is on disk already"
    free = (
        f"; the models disk has {offered.free_bytes / GIB:.1f} GiB free"
        if offered.free_bytes is not None
        else ""
    )
    disk = (
        f"{got}; the reference's log-probabilities take {offered.base_bytes / GIB:.1f} GiB while "
        f"it runs{free}."
    )
    after = (
        "The quants not chosen stay on disk (`--keep`)."
        if plan.keep
        else (
            "After, the quants it downloaded and does not keep are deleted, and the one "
            f"{plan.reference.recipe.name} moves from, each where no other instance uses it "
            "(`--keep` keeps them)."
        )
    )
    said = [measures, f"{disk} {after}"]
    if offered.busy:
        said.append(
            f"{offered.busy}: the quants are planned around it, so one that fits once more of "
            "the card is free may be missing here."
        )
    return said
