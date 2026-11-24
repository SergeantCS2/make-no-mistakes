"""The RAM as one more place a model sits: an instance on the processor's image (the
CPU backend: a box with no card, or `MNM_BACKEND=cpu`) holds its room in RAM, every other on
its card, whatever box it runs on. So room is judged per place, as on several cards: the
snapshot carries the RAM beside the cards (`MemorySnapshot.ram` at `RAM_INDEX`),
`card_room.placed` puts a processor instance's estimate there, and every per-card rule
(fit now, room-making, a set's fit-together, a load's parking) judges it on the RAM and
parks for it only what sits in the RAM. A separator thus never loads unjudged beside a
running song on a box without a card, and a card's instance is never judged on the RAM (as
one box-wide switch would judge it).

The RAM's room is MemTotal less what the box's system keeps (its RAM platform's margin, the
4 GiB `OS_RAM_MARGIN` before a box is recorded), its use MemTotal less MemAvailable. A plan in
RAM loads its weights with `--load-mode none` (`fit.budget.pool_load`), since a mapped file's
pages count as available."""

from __future__ import annotations

from typing import TYPE_CHECKING

from makenomistakes.backend import Backend
from makenomistakes.backend_now import recipe_backend_or_none, recorded_box
from makenomistakes.fit.ram_floor import OS_RAM_MARGIN
from makenomistakes.hardware.probe import RAM_INDEX, Gpu, MemoryInfo, MemorySnapshot
from makenomistakes.hardware.ram_platforms import platform_of

if TYPE_CHECKING:
    from makenomistakes.config import Settings
    from makenomistakes.recipes import Recipe

RAM_NAME = "RAM"


def ram_margin(settings: Settings) -> int:
    """What the box's system keeps of its RAM: its RAM platform's row (a server keeps 8 GiB),
    else 4 GiB while no box is recorded."""
    box = recorded_box(settings.expanded_state_dir())
    return OS_RAM_MARGIN if box is None else platform_of(box).margin_bytes


def ram_pool(memory: MemoryInfo | None, margin: int) -> Gpu | None:
    """The RAM as a place models sit; None when MemAvailable could not be read."""
    if memory is None or memory.available_bytes is None or not memory.total_bytes:
        return None
    usable = max(memory.total_bytes - margin, 0)
    used = min(max(memory.total_bytes - memory.available_bytes, 0), usable)
    return Gpu(name=RAM_NAME, vram_total_bytes=usable, vram_used_bytes=used, index=RAM_INDEX)


def with_ram(snapshot: MemorySnapshot, margin: int) -> MemorySnapshot:
    """`snapshot` with its RAM read as a place (unchanged when it already carries one)."""
    if snapshot.ram is not None:
        return snapshot
    return snapshot.model_copy(update={"ram": ram_pool(snapshot.memory, margin)})


def runs_on_processor(recipe: Recipe) -> bool:
    """The instance's image is the processor's: its room is the RAM's."""
    return recipe_backend_or_none(recipe) is Backend.CPU


__all__ = ["RAM_INDEX", "ram_margin", "ram_pool", "runs_on_processor", "with_ram"]
