"""The RAM floor as the planner applies it (`fit.ram_floor`): to the smallest quant before
any profile is weighed, and to every quant a profile or an explicit `--quant` lands on, at the
plan's own context. Setup's ladder (`ladder.py`) reads the same floor at 4,096 tokens, so a
quant the ladder calls short is never planned. The ladder prices the text model, as the large
models are added: counting Bolt's vision projector, it had called UD-IQ3_XXS "0.5 GB short"
while `mnm add … --type chat` planned the text model with 1.7 GB to spare."""

from __future__ import annotations

from typing import TYPE_CHECKING

from makenomistakes.engine_args import KvType
from makenomistakes.fit.ram_floor import Basis, Floor, floor_of, floor_words
from makenomistakes.hardware.fingerprint import SHARED_MEMORY

if TYPE_CHECKING:
    from makenomistakes.fit.profiles import Planner
    from makenomistakes.hub.facts import ModelFacts
    from makenomistakes.hub.variants import GgufVariant


class RamFloorError(ValueError):
    """Not even the smallest quant fits this machine's RAM and VRAM together."""


def floor_at(planner: Planner, variant: GgufVariant, ctx: int) -> Floor:
    """The variant's RAM floor at `ctx`: against the machine's whole RAM and VRAM when the
    probe read them, else what is free."""
    from makenomistakes.fit.profiles import ProfileName  # noqa: PLC0415 - it imports this

    estimate = planner.profile(ProfileName.CUSTOM, variant, ctx, KvType.F16, "").estimate
    budget = planner.budget
    ram = budget.ram_total_bytes or budget.ram_free_bytes  # free: a budget built by hand
    vram = budget.vram_total_bytes or budget.vram_free_bytes
    if budget.device.memory in SHARED_MEMORY:  # One pool, counted once
        vram = 0
    disk = variant.engram.size_bytes if variant.engram else 0
    basis = (
        Basis.FREE
        if budget.ram_total_bytes is None
        else Basis.GIVEN
        if budget.elsewhere  # `--vram`: `explicit_budget`
        else Basis.RAM
        if budget.no_card
        else Basis.BOX
    )
    margin, pool = budget.margin, budget.pool
    return floor_of(variant.quant, estimate, disk, ram, vram, margin=margin, pool=pool, basis=basis)


def holds(planner: Planner, variant: GgufVariant, ctx: int) -> bool:
    """Whether the machine holds the variant at `ctx`; always for a budget that does not know
    the machine's RAM (a budget built by hand)."""
    return planner.budget.ram_total_bytes is None or floor_at(planner, variant, ctx).fits


def check_floor(planner: Planner, facts: ModelFacts, variants: list[GgufVariant], ctx: int) -> None:
    """RamFloorError, with the arithmetic, when not even the smallest quant fits."""
    if planner.budget.ram_total_bytes is None:
        return
    smallest = min(variants, key=lambda v: v.memory_bytes)
    floor = floor_at(planner, smallest, ctx)
    if not floor.fits:
        raise RamFloorError(floor_words(facts.family_name, floor, None))


def check_pick(planner: Planner, facts: ModelFacts, variant: GgufVariant, ctx: int) -> None:
    """An explicit quant or context (`Plan.choose`): refused, with the arithmetic and the
    smallest quant that holds, when this machine cannot hold it."""
    if planner.budget.ram_total_bytes is None:
        return
    floor = floor_at(planner, variant, ctx)
    if floor.fits:
        return
    fitting = sorted(
        (v for v in facts.gguf_variants if holds(planner, v, ctx)), key=lambda v: v.memory_bytes
    )
    if not fitting:
        raise RamFloorError(floor_words(facts.family_name, floor, None))
    raise RamFloorError(floor_words(facts.family_name, floor, None, fitting[-1].quant))
