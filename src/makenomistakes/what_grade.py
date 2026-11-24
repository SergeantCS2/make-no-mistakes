"""`mnm what`: every hit graded against this box by the planner, exactly as its `mnm add`
line would plan it (`plan.make_plan` on the remote header, then `Plan.choose`: Balanced, the
agent's context for a coder, a curated row's quant). When Balanced's quant does not fit, the
quants below it that the plan runs are tried, most faithful first (`Plan.smaller_quants`), and
the row names the first that fits; when nothing fits, the row is the planner's refusal in its
own words, its verdict by the refusal's kind (`plan_types`: no room, a read that failed, a
type the model cannot be added as).

Each hit is one guarded unit on a thread of its own (`GRADERS` at a time): a repo the Hub
refuses, a source that times out or a model of another type is a row of its own, never why
the table is not shown. Nothing downloads: the planner reads headers over ranges."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum

from makenomistakes.add_intent import AddIntent
from makenomistakes.agent_profile import agent_context
from makenomistakes.config import Settings
from makenomistakes.fit import Budget, Profile, ProfileName
from makenomistakes.hub.facts import Modality
from makenomistakes.model_uses import row_for
from makenomistakes.plan import make_plan
from makenomistakes.plan_types import (
    NoRoomError,
    Plan,
    PlanError,
    SourceReadError,
    TypeRefusedError,
)
from makenomistakes.what_sources import Hit

GRADERS = 4  # hits planned at once: each is a few Hub calls and a ranged header read


class Verdict(StrEnum):
    """What the planner said of a hit, set once by `grade`."""

    FITS = "fits"  # Balanced (or the curated row's quant) fits
    SMALLER = "a smaller quant"  # Balanced's quant does not fit; a smaller one does
    NO_FIT = "does not fit"  # no quant fits this box (`NoRoomError`, or every quant tried)
    REFUSED = "refused"  # the planner refuses the model: nothing here runs it
    UNREAD = "cannot read"  # the source or the file could not be read (`SourceReadError`)
    OTHER_TYPE = "another type"  # a model of another type than the use asked: left out


FITTING = frozenset({Verdict.FITS, Verdict.SMALLER})
# the planner's refusals by kind, the most specific first (`TypeRefusedError` is apart: its
# row names the model's own type)
_REFUSALS: tuple[tuple[type[PlanError], Verdict], ...] = (
    (NoRoomError, Verdict.NO_FIT),
    (SourceReadError, Verdict.UNREAD),
    (PlanError, Verdict.REFUSED),
)


@dataclass(frozen=True)
class Graded:
    """One hit as the planner sees it on this box. `pick` is set exactly when the verdict
    fits; `refusal` is the planner's words for why not (for `SMALLER`: why not Balanced's)."""

    hit: Hit
    title: str
    verdict: Verdict
    pick: Profile | None = None  # what `mnm add` would save
    params_b: float | None = None
    refusal: str | None = None
    repo_id: str | None = None  # the repo the planner read (its quants' measured scores)

    def __post_init__(self) -> None:
        if (self.pick is not None) != (self.verdict in FITTING):
            raise ValueError(f"a {self.verdict.name} row with pick={self.pick!r}")

    @property
    def fits(self) -> bool:
        return self.verdict in FITTING

    @property
    def smaller(self) -> bool:
        return self.verdict is Verdict.SMALLER

    @property
    def size(self) -> str:
        """Its parameters as a row shows them (`8.2B`, `35B`, `270M`), empty when unread."""
        if self.params_b is None:
            return ""
        if self.params_b < 1:
            return f"{self.params_b * 1000:.0f}M"
        return f"{self.params_b:.1f}B" if self.params_b < 10 else f"{self.params_b:.0f}B"

    @property
    def intent(self) -> AddIntent:
        """What its `mnm add` adds: the agent profile for a coder, a curated row's quant or
        the smaller one that fits, a curated row's name."""
        quant = self.pick.quant if self.smaller and self.pick is not None else self.hit.quant
        agents = row_for(self.hit.use).for_agents
        return AddIntent(self.hit.reference, agents, quant, self.hit.name)

    @property
    def add_line(self) -> str:
        return self.intent.cli()

    @property
    def plan_href(self) -> str:
        return self.intent.href()


Planner = Callable[[str, Settings, Budget, Modality | None, str | None], Plan]


def plan_hit(
    reference: str,
    settings: Settings,
    budget: Budget,
    model_type: Modality | None,
    quant: str | None,
) -> Plan:
    """The plan `mnm add <reference> [--type] [--quant]` makes on this budget."""
    return make_plan(reference, settings, budget, model_type=model_type, quant=quant)


def grade(hit: Hit, settings: Settings, budget: Budget, planner: Planner = plan_hit) -> Graded:
    """One hit: planned, its Balanced pick (or the first smaller quant that fits), or why
    not, the verdict by the kind of the planner's refusal. Whatever one hit raises is that
    hit's row."""
    try:
        return _graded(hit, settings, budget, planner)
    except TypeRefusedError as exc:  # a coder search's embedding repo: no coder, left out
        return Graded(hit, hit.shown, Verdict.OTHER_TYPE, refusal=f"a {exc.modality.label} model")
    except PlanError as exc:
        verdict = next(v for kind, v in _REFUSALS if isinstance(exc, kind))
        return Graded(hit, hit.shown, verdict, refusal=str(exc))
    except Exception as exc:  # a source's own error class: one hit's failure is its row
        return Graded(hit, hit.shown, Verdict.UNREAD, refusal=f"could not be planned: {exc}")


def _graded(hit: Hit, settings: Settings, budget: Budget, planner: Planner) -> Graded:
    row = row_for(hit.use)
    plan = planner(hit.reference, settings, budget, row.plans_as, hit.quant)
    facts = plan.facts
    title, params, repo = facts.display_name, facts.params_b, facts.repo_id
    if not row.serves(facts.modality):
        refusal = f"a {facts.modality.label} model"
        return Graded(hit, title, Verdict.OTHER_TYPE, params_b=params, refusal=refusal)
    ctx = agent_context(plan, None) if row.for_agents else None
    try:
        pick = plan.choose(ProfileName.BALANCED, hit.quant, ctx)
        return Graded(hit, title, Verdict.FITS, pick, params, repo_id=repo)
    except PlanError as exc:
        refused = exc
    if hit.quant is None:  # a curated row's quant is the row's: never another
        for quant in plan.smaller_quants():
            try:
                pick = plan.choose(ProfileName.BALANCED, quant, ctx)
                return Graded(hit, title, Verdict.SMALLER, pick, params, str(refused), repo)
            except PlanError as exc:
                refused = exc  # the smallest tried says the least it needs
    return Graded(hit, title, Verdict.NO_FIT, params_b=params, refusal=str(refused), repo_id=repo)


def grade_all(
    hits: list[Hit],
    settings: Settings,
    budget: Budget,
    planner: Planner = plan_hit,
    graders: int = GRADERS,
) -> list[Graded]:
    """Every hit graded, in the hits' order (curated, then each source's)."""
    if not hits:
        return []
    with ThreadPoolExecutor(max_workers=min(graders, len(hits))) as pool:
        return list(pool.map(lambda h: grade(h, settings, budget, planner), hits))
