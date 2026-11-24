"""The lines a plan says about how its card is priced: the device row and its basis
(`fit.devices.device_note`), and, for a `--vram` plan of another machine, that its RAM-held
experts are priced at the default rate, fitted on an RTX 4070 (this box's readings say
nothing of that one)."""

from __future__ import annotations

from collections.abc import Iterable

from makenomistakes.backend import label
from makenomistakes.fit.budget import Budget
from makenomistakes.fit.cpu_plan import CPU_SOURCE, in_ram
from makenomistakes.fit.cpu_rates import CPU_EXPERT_RATE, EXPERT_TOUCH_SECONDS
from makenomistakes.fit.devices import device_note
from makenomistakes.fit.estimate import FitLevel
from makenomistakes.fit.profiles import Profile
from makenomistakes.hardware.fingerprint import MemoryKind


def pool_words(budget: Budget) -> str | None:
    """A plan made in RAM (the CPU backend, or Vulkan on Mesa's software device) or in
    a unified pool, as the budget line says it; None for a card."""
    if budget.cpu is not None:
        threads = f", {budget.cpu.threads} threads" if budget.cpu.threads else ""
        numa = f", --numa {budget.cpu.numa}" if budget.cpu.numa else ""
        held = " and what mnm's running models hold," if budget.held_by else ""
        return (
            f"{budget.vram_free_bytes / 2**30:.1f} GiB of RAM for models (what is free{held} "
            f"less {budget.margin / 2**30:.0f} GiB for the system), on the processor: the CPU "
            "backend"
            f"{threads}{numa}"
        )
    if in_ram(budget):  # the software device
        return (
            f"{budget.vram_free_bytes / 2**30:.1f} GiB of RAM as Mesa's software device's memory "
            "(Vulkan without a card: slow)"
        )
    if budget.pool:  # A unified card (a GB10) or an APU's GTT
        total = budget.ram_total_bytes
        of = f" of the {total / 10**9:.0f} GB" if total else ""
        what = "unified memory" if budget.device.memory is MemoryKind.UNIFIED else "APU's GTT"
        return (
            f"{budget.vram_free_bytes / 2**30:.1f} GiB{of} {what} for models (what is free less "
            f"{budget.margin / 2**30:.0f} GiB kept for the system): every model planned whole, "
            "nothing offloaded"
        )
    return None


def backend_tail(budget: Budget) -> str:
    """`; backend: CUDA 13` after a plan's budget line; empty for a plan that did not
    read the box (`--vram`)."""
    return f"; backend: {label(budget.backend)}" if budget.backend is not None else ""


def device_lines(budget: Budget, profiles: Iterable[Profile] = ()) -> list[str]:
    here = budget.vram_total_bytes is not None  # a `--vram` budget plans another machine
    if in_ram(budget):
        rate = f"{budget.gpu_bandwidth / 1e9:.0f} GB/s"
        if budget.ram_note:  # The RAM platform's row, or this box's probe
            return [f"Priced as: {budget.device.name or 'this CPU'}: {budget.ram_note}."]
        return [
            f"Priced as: {budget.device.name or 'this CPU'}: {rate} read a token "
            f"(estimated from {CPU_SOURCE}): `mnm try --speed` and `mnm tune` measure this box."
        ]
    lines = [f"Priced as: {device_note(budget.device, here=here)}."]
    listed = list(profiles)
    if here and budget.ram_note and any(_in_ram(p) for p in listed):
        lines.append(f"{budget.ram_note}.")  # The RAM platform's row, or this box's
    if not here and any(p.fit.cpu_moe_layers for p in listed):
        lines.append(
            "RAM-held experts priced at an RTX 4070 machine's rate "
            f"({CPU_EXPERT_RATE / 1e9:g} GB/s, {EXPERT_TOUCH_SECONDS * 1e6:g} us a used expert): "
            "a plan for another machine has no readings of its RAM."
        )
    return lines


def _in_ram(profile: Profile) -> bool:
    """The plan keeps weights in RAM: experts (`--n-cpu-moe`, a placement's) or whole blocks."""
    placed = profile.placement is not None and bool(profile.placement.in_ram)
    return bool(profile.fit.cpu_moe_layers) or placed or profile.fit.level is FitLevel.OFFLOAD
