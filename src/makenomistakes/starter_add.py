"""One way to add a starter item, for the guide, `mnm starter` and the web's Get started:
what `mnm add <reference> --name <name>` saves, the agent profile for a coder. Nothing is
downloaded here; the caller downloads (with its own progress) right after each add, so a
later item that cannot be planned never leaves an earlier one saved without its files."""

from __future__ import annotations

from collections.abc import Callable

from makenomistakes.agent_profile import agent_context, requested_type
from makenomistakes.config import Settings
from makenomistakes.fit import Budget, Profile, ProfileName
from makenomistakes.hub.facts import Modality
from makenomistakes.instances import InstanceService
from makenomistakes.plan import Plan, make_plan
from makenomistakes.plan_budget import planning_budget
from makenomistakes.recipes import Recipe
from makenomistakes.starter import Item

PlanFn = Callable[[str, Settings, Budget | None, str | None, Modality | None], Plan]


def plan_item(service: InstanceService, item: Item, planner: PlanFn = make_plan) -> Plan:
    reference = f"{item.reference}:{item.quant}" if item.quant else item.reference
    wanted = requested_type(Modality(item.kind) if item.kind else None, item.for_agents)
    return planner(reference, service.settings, planning_budget(service), None, wanted)


def chosen_profile(plan: Plan, item: Item) -> Profile:
    """The one row a starter saves: Balanced at the item's quant (the reference's tag) and,
    for a coder, the agent context. What is shown is this row, never `mnm fit`'s three rows
    (UD-Q4_K_XL on each) beside a saved UD-Q4_K_M."""
    ctx = agent_context(plan, None) if item.for_agents else None
    return plan.choose(ProfileName.BALANCED, None, ctx, None)


def add_item(
    service: InstanceService, item: Item, plan: Plan | None = None, planner: PlanFn = make_plan
) -> Recipe:
    """Plan (unless given the plan), choose its row, save. Raises PlanError/InstanceError."""
    service.check_name(item.name)
    plan = plan or plan_item(service, item, planner)
    return service.add(plan, chosen_profile(plan, item), item.name, upscaler=True)


def saved_words(profile: Profile, plan: Plan) -> str:
    """`Saved: UD-Q4_K_M, ctx 32,768, every layer on the card (the experts of 30 of 40
    layers in RAM), ~19 tok/s`: the plan a starter saved, in one line."""
    parts = [profile.quant]
    if profile.ctx:
        parts.append(f"ctx {profile.ctx:,}")
    blocks = plan.header.block_count if plan.header is not None else None
    fit = profile.fit
    if blocks and fit.cpu_moe_layers >= blocks and fit.gpu_layers < blocks:
        parts.append(f"every expert in RAM, {fit.gpu_layers} of {blocks} layers' rest on the card")
    elif blocks and fit.cpu_moe_layers:
        parts.append(
            f"every layer on the card (the experts of {fit.cpu_moe_layers} of {blocks} layers "
            "in RAM)"
        )
    elif blocks and fit.gpu_layers and fit.gpu_layers < blocks:
        parts.append(f"{fit.gpu_layers} of {blocks} layers on the card, the rest in RAM")
    parts.append(profile.vram_words)
    if profile.speed_summary:
        parts.append(profile.speed_summary)
    return "Saved: " + ", ".join(parts) + "."
