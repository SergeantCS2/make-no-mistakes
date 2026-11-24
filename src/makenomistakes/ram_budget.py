"""The CPU backend's budget: the RAM as the one pool every model is planned in, priced
by the box's RAM platform until it measures. What `plan.live_budget` and the planning
budget on a box whose backend is the processor's both build (from the snapshot the
planning budget was handed, never probing the box again)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from makenomistakes.backend import Backend
from makenomistakes.fit import Budget
from makenomistakes.fit.cpu_plan import platform_budget
from makenomistakes.hardware.ram_platforms import platform_of
from makenomistakes.machine_facts import MachineFactsError, load

if TYPE_CHECKING:
    from makenomistakes.config import Settings
    from makenomistakes.hardware.fingerprint import MachineFingerprint
    from makenomistakes.hardware.probe import MemoryInfo


def in_ram_budget(settings: Settings, box: MachineFingerprint, memory: MemoryInfo) -> Budget:
    """The RAM's budget on `box`: its platform's rate until `mnm doctor --deep` measures one."""
    total, free = memory.total_bytes or 0, memory.available_bytes or 0
    budget = platform_budget(platform_of(box), total, free, box.cpu, ram_gbps(settings))
    return budget.model_copy(update={"backend": Backend.CPU, "arch": box.arch})


def ram_gbps(settings: Settings) -> float | None:
    """The RAM bandwidth `mnm doctor --deep` measured over every core on this box
    (`MachineFacts.every_core_gbps`); None before it did, or for a one-thread probe saved
    by an older mnm or an implausible one (the raw figure would price a CPU plan at a
    one-thread probe's rate)."""
    try:
        found = load(settings.expanded_state_dir())
    except MachineFactsError:
        return None
    return found.every_core_gbps if found is not None else None
