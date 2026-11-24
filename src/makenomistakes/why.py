"""`mnm why <name>`: an instance's plan in words. What it runs as saved; why the planner
picks what it picks for this box today (its profile's own reason, what a re-plan keeps of the
instance's own: its quant, a coding instance's context; how it fits and where; its speed in its
kind's unit; the other profiles); what a re-plan now would change (`mnm replan`'s own words,
`replan_words`); and what more room would change, planned for real on the box's budget grown
(`Budget.with_more`). Read-only: it plans and saves nothing, reading the model's header from
its source as `mnm replan` does. The instance page shows the same words.

So it never speaks in tok/s for voices and music, never calls a kept quant "your explicit
choice", never describes a re-plan as what the instance runs, and never plans more room by
editing the plan's budget (which would leave its correction, a pool's RAM and vLLM's split
behind)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from makenomistakes.busy_card import replan_note
from makenomistakes.fit.budget import Budget
from makenomistakes.fit.estimate import FitLevel
from makenomistakes.fit.profiles import Profile, ProfileName
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.model_meta import serving_context, serving_slots
from makenomistakes.plan_budget import budget_now
from makenomistakes.plan_types import NoSuchQuantError, Plan, PlanError, base_profile
from makenomistakes.recipes import Recipe
from makenomistakes.recipes.cards import ONE_CARD_ENGINES
from makenomistakes.replan import (
    Replanned,
    choice,
    keeps_context,
    keeps_quant,
    plan_again,
    replan,
)
from makenomistakes.replan_words import replan_lines

GIB = 2**30
EXTRA_GIB = 4
FITS = {  # said inside other words' parentheses too: no parentheses of their own
    FitLevel.COMFORTABLE: "comfortably",
    FitLevel.TIGHT: "tightly",
    FitLevel.OFFLOAD: "only with part of it in RAM",
    FitLevel.NO_FIT: "not at all",
}
TIGHT_MEANS = "under a quarter of the room left"  # the fit sentence says what tight means
NOW_HEADER = "A plan for the box as it is now would change:"


@dataclass(frozen=True)
class Why:
    """The plan in words: `runs` one line; each list a paragraph's sentences; `problem`: why
    it could not be planned now (the source, the card), said instead of the reasons."""

    name: str
    runs: str
    reasons: list[str] = field(default_factory=list)
    today: list[str] = field(default_factory=list)
    more: list[str] = field(default_factory=list)
    problem: str | None = None

    @property
    def lines(self) -> list[str]:
        """Every sentence in order, for a page or a file."""
        said = [self.runs, *self.reasons, *self.today, *self.more]
        return [*said, self.problem] if self.problem else said


def explain(service: InstanceService, name: str, extra_gib: int = EXTRA_GIB) -> Why:
    """`name`'s plan in words; InstanceError only for a name that is no instance."""
    old = service.get(name)
    runs = runs_line(old)
    try:
        budget = budget_now(service)  # PlanError: no card read, no free RAM read
        plan = plan_again(service, old, budget)
        profile = choice(old, plan)  # what a re-plan gives it: its quant, an agent's context
    except (PlanError, InstanceError) as exc:
        return Why(name, runs, problem=problem_words(old, exc))
    done, today = today_lines(service, name, budget, plan)
    reasons = [
        *framing(budget, done),
        *profile_reasons(old, plan, profile),
        *others(old, plan, profile),
    ]
    return Why(name, runs, reasons, today, more_room(service, old, budget, profile, extra_gib))


def runs_line(recipe: Recipe) -> str:
    """What it runs as saved, in one line: the context one request gets, and how many at once
    (`model_meta`'s, vLLM's `max_model_len` included)."""
    per = serving_context(recipe)
    slots = serving_slots(recipe)
    context = ""
    if per:
        each = f" each for {slots} at once" if slots > 1 else ""
        context = f", {per:,} tokens of context{each}"
    chosen = (  # `--quant`, `--ctx`, `--for-agents`: the custom profile is no word for it
        "its own settings"
        if recipe.profile is ProfileName.CUSTOM
        else f"the {recipe.profile.value} profile"
    )
    return f"{recipe.name} runs {recipe.repo_id} at {recipe.quant} ({chosen}{context})."


def problem_words(old: Recipe, exc: Exception) -> str:
    """Why it could not be planned now, and the next step where there is one: a quant the
    repo no longer lists is planned again with one it has."""
    said = f"It could not be planned now: {exc}"
    if isinstance(exc, NoSuchQuantError):
        said += f" `mnm replan {old.name} --any-quant` plans it with a quant the repo has now."
    return said


def place(budget: Budget) -> str:
    """Where a plan on this budget puts the model."""
    if budget.no_card:
        return "in the RAM"
    if budget.pool:
        return "in the unified memory"
    return "on the cards" if len(budget.cards) > 1 else "on the card"


def framing(budget: Budget, done: Replanned | None) -> list[str]:
    """What shapes the lines below, said first: other programs holding the card (as
    `mnm replan` says it), and a re-plan that differs from what the instance runs (else the
    lines read as the instance's own when they are a re-plan's)."""
    said = [busy] if (busy := replan_note(budget)) is not None else []
    if done is not None and (done.changed or done.withheld is not None):
        said.append(
            "These lines are the plan for the box as it is now; the instance runs its saved "
            "settings until a re-plan saves this one (Now, below)."
        )
    return said


def profile_reasons(old: Recipe, plan: Plan, profile: Profile) -> list[str]:
    """The instance's own reasons, how it fits and where, and how fast in its kind's unit."""
    fit = profile.fit
    said = [*own_reasons(old, plan, profile), *fit_lines(old, plan.budget, profile)]
    if fit.cpu_moe_layers:
        said.append(f"The experts of {fit.cpu_moe_layers} layers stay in RAM, read from there.")
    said.append(f"Speed by the planner's estimate: {profile.speed_summary}.")
    return said


def fit_lines(old: Recipe, budget: Budget, profile: Profile) -> list[str]:
    """How it fits against the room its planner judged (`Fit.room_bytes`: llama.cpp keeps
    WSL's margin out of it, the others judge the free VRAM), in the card's own terms (a learned
    correction prices estimates up: both figures are scaled by it, as nvidia-smi would read
    them), then what that room is made of beyond the card's free figure: mnm's other running
    models counted free, the margin where this planner kept it (never one rule for every
    engine: only llama.cpp keeps it), the correction."""
    where = place(budget)
    ratio = budget.correction
    fit = profile.fit
    room = fit.room_bytes or budget.vram_free_bytes
    size = f"{fit.gpu_bytes * ratio / GIB:.1f} GiB {where}"
    if profile.ram_bytes:
        size += f" + {profile.ram_bytes / GIB:.1f} GiB in RAM"
    level = fit.level
    fits = f"{FITS[level]} ({TIGHT_MEANS})" if level is FitLevel.TIGHT else FITS[level]
    said = [f"It fits {fits}: {size}, of the {room * ratio / GIB:.1f} GiB a plan may use."]
    basis = []
    others = [name for name in budget.held_by if name != old.name]  # its own room is its own
    if others:
        verb, they = ("holds", "it were") if len(others) == 1 else ("hold", "they were")
        basis.append(
            f"counts what {', '.join(others)} {verb} as free (planned as if {they} stopped)"
        )
    kept = (budget.vram_free_bytes - room) * ratio  # the margin, where this planner kept one
    if kept > GIB / 64:
        cards = max(len(budget.cards), 1)
        top = "the card's top" if cards == 1 else f"each of the {cards} cards' tops"
        basis.append(f"keeps {kept / cards / GIB:.1f} GiB free at {top} (WSL spills past it)")
    if basis:
        said.append(f"That room {'; '.join(basis)}.")
    if ratio != 1.0:
        said.append(f"Sizes are this box's own: estimates x{ratio:.2f}, as its readings measured.")
    return said


def own_reasons(old: Recipe, plan: Plan, profile: Profile) -> list[str]:
    """Its profile's own reason, then what a re-plan keeps that the profile alone would not
    pick: its quant (the file it downloaded, or a quality tune's choice) and a coding
    instance's context. An instance added with its own settings (the custom profile, sized from
    balanced: `plan_types.base_profile`) is said with what a re-plan keeps of them, by the one
    rule `replan.choice` applies (`keeps_quant`, `keeps_context`)."""
    base = base_profile(old.profile)
    named = plan.profiles.get(base)
    if named is None:
        return []
    title = base.value
    said = []
    if old.profile is ProfileName.CUSTOM:
        kept = [
            w
            for w, held in (("its quant", keeps_quant(old)), ("its context", keeps_context(old)))
            if held
        ]
        keeps = (
            f"keeps {' and '.join(kept)} and sizes the rest as the {title} profile would"
            if kept
            else f"keeps neither and takes the {title} profile's pick"
        )
        said.append(
            f"It was added with its own settings (`--quant`, `--ctx` or `--for-agents`): a "
            f"re-plan {keeps}."
        )
    said.append(f"The {title} profile: {named.why}")
    if profile.quant != named.quant:
        said.append(_kept_quant(old, named, title))
    if keeps_context(old) and profile.ctx and profile.ctx != named.ctx:
        said.append(
            f"Its context, {profile.ctx:,} tokens, stays when it is planned again (a coding "
            f"instance's context is what its clients need); the {title} profile alone would "
            f"take {named.ctx:,}."
        )
    return said


def _kept_quant(old: Recipe, named: Profile, title: str) -> str:
    if old.quality is not None and old.quality.quant == old.quant:  # a quality tune chose it
        return (
            f"It keeps {old.quant}, which a quality tune chose (`mnm show {old.name}` has its "
            f"scores); the {title} profile alone would pick {named.quant} on this box now."
        )
    return (
        f"It keeps {old.quant}, the file it downloaded; the {title} profile alone would pick "
        f"{named.quant} on this box now (`mnm replan {old.name} --any-quant` takes it)."
    )


def others(old: Recipe, plan: Plan, chosen: Profile) -> list[str]:
    """The other profiles, one sentence each: never the instance's own, nor one that picks what
    a re-plan gives it anyway (a custom profile's settings a re-plan does not keep fold onto
    balanced's pick)."""
    said = []
    for name, profile in plan.profiles.items():
        if name in (old.profile, ProfileName.CUSTOM):
            continue
        if (profile.quant, profile.ctx) == (chosen.quant, chosen.ctx):
            continue
        ctx = f" at {profile.ctx:,} context" if profile.ctx else ""
        said.append(
            f"{name.value.capitalize()} would run {profile.quant}{ctx} "
            f"(it fits {FITS[profile.fit.level]}, {profile.speed_summary})."
        )
    return said


def today_lines(
    service: InstanceService, name: str, budget: Budget, plan: Plan
) -> tuple[Replanned | None, list[str]]:
    """What a plan for the box as it is now would change, in `mnm replan`'s own words; its tune
    said as what a re-plan would do (`mnm replan`'s note says it was done)."""
    try:
        done = replan(service, name, budget, keep_if_fits=True, made=plan)
    except InstanceError as exc:
        return None, [str(exc)]
    said = [line for _tone, line in replan_lines(replace(done, tune_note=None), name, NOW_HEADER)]
    tuned = done.old.tuned
    if done.tune_note is not None and tuned is not None:
        said.append(
            f"A re-plan keeps its tune ({tuned.summary}): it still fits."
            if done.new.tuned is not None
            else f"A re-plan would drop its tune ({tuned.summary}): it no longer fits the new "
            "plan, and the next idle card tunes it again."
        )
    if done.changed:
        said.append(f"`mnm replan {name}` saves it.")
    return done, said


def more_room(
    service: InstanceService, old: Recipe, budget: Budget, profile: Profile, extra_gib: int
) -> list[str]:
    """What `extra_gib` more of the place it plans in would change, planned for real on the
    box's budget grown (`Budget.with_more`), never the plan's own narrowed one (a vLLM split
    plan's is one card's). On several cards an engine that runs on one card (sd.cpp,
    whisper.cpp, audio.cpp: `ONE_CARD_ENGINES`) is planned on the first card as `mnm add` plans
    it; for the others which card gets the room is a placement question, said not worked out."""
    where = place(budget)
    if len(budget.cards) > 1:
        if old.engine not in ONE_CARD_ENGINES:
            return [
                f"{len(budget.cards)} cards: which card more room would go to is a placement "
                "question, not worked out here."
            ]
        budget = budget.one_card()
        where = place(budget)
    try:
        bigger = plan_again(service, old, budget.with_more(extra_gib * GIB))
        then = choice(old, bigger)
    except (PlanError, InstanceError) as exc:
        return [f"With {extra_gib} GiB more {where} it could not be planned: {exc}"]
    moved = differences(profile, then)
    said = [
        f"With {extra_gib} GiB more {where}: {', '.join(moved)}."
        if moved
        else f"{extra_gib} GiB more {where} would change nothing: it fits as this plan wants."
    ]
    named = bigger.profiles.get(base_profile(old.profile))
    if named is not None and named.quant != then.quant:  # the profile's own pick, named
        ctx = f" at {named.ctx:,} context" if named.ctx else ""
        said.append(
            f"There the {base_profile(old.profile).value} profile alone would pick "
            f"{named.quant}{ctx}."
        )
    return said


def differences(now: Profile, then: Profile) -> list[str]:
    """What moves between two plans of the instance, each with what it is now, the fit first."""
    said = []
    if then.fit.level is not now.fit.level:
        said.append(f"it fits {FITS[then.fit.level]} (now {FITS[now.fit.level]})")
    if then.quant != now.quant:
        said.append(f"{then.quant} (now {now.quant})")
    if then.ctx and then.ctx != now.ctx:
        said.append(f"{then.ctx:,} tokens of context (now {now.ctx:,})")
    if then.fit.cpu_moe_layers != now.fit.cpu_moe_layers:
        said.append(
            f"the experts of {then.fit.cpu_moe_layers} layers in RAM (now {now.fit.cpu_moe_layers})"
        )
    if then.speed_summary != now.speed_summary:
        said.append(f"{then.speed_summary} (now {now.speed_summary})")
    return said
