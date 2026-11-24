"""The RAM floor: whether this machine can hold a model at all, said with
the arithmetic. Whatever the cards do not hold sits in RAM, so the weights (less an Engram
table, which stays on disk and is read as needed), the cache and buffers, and a margin for
everything else the machine runs must fit in the RAM and the VRAM together. A model that
does not is refused before any profile is weighed, with what it needs, what this machine
has, the smallest quant that would fit, and the RAM a box with these cards would need.

RAM and disk are said in GB (as RAM and disks are sold, and as a box is sized), the
cards in GiB (as nvidia-smi and every other line of mnm says them).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from makenomistakes.fit.estimate import GIB, MemoryEstimate
from makenomistakes.hardware.fingerprint import UNIFIED_MARGIN, MemoryKind

# What the rest of the machine keeps while a model runs: the OS and (under WSL) the VM's own
# kernel and page cache, Docker, the router and the web UI, llama.cpp's own host buffers
# beside the weights. 4 GiB: an RTX 4070 machine's WSL with every mnm service up and no model
# loaded held 2.6 GiB (the doctor's RAM line), plus room to breathe; a
# floor, not a plan, so it errs towards running.
OS_RAM_MARGIN = 4 * GIB
GB = 10**9


def margin_for(memory: MemoryKind) -> int:
    """What the system keeps beside the models: 4 GiB, and 8 on a unified pool (the
    GB10 hangs rather than OOM-kills, `hardware.unified_pool`)."""
    return UNIFIED_MARGIN if memory is MemoryKind.UNIFIED else OS_RAM_MARGIN


# RAM sizes boxes are built with (DDR4/DDR5 in two or four DIMMs, then workstation boards).
BOX_RAM_GB = (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048)


class Basis(Enum):
    """Where a floor's RAM and VRAM figures came from, which its words name (else
    `mnm fit --vram …` says "this machine has 15 GB of RAM", `--ram`'s default of 14 GiB, on
    a box with 25 GB)."""

    BOX = "box"  # the live probe: the RAM and VRAM the box has
    RAM = "ram"  # the live probe of a box without a card: its RAM alone
    GIVEN = "given"  # `--vram` (and `--ram`, 14 GiB unless given): another machine's
    FREE = "free"  # a budget that knows only what is free now


@dataclass(frozen=True)
class Floor:
    """One quant's need against what this machine has (RAM + VRAM)."""

    quant: str
    weights_bytes: int  # what RAM and the cards hold (no Engram table)
    cache_bytes: int  # KV, recurrent state, compute buffers and the runtime
    disk_bytes: int  # an Engram table, read from disk as needed
    ram_bytes: int  # the machine's RAM (under WSL the VM's)
    vram_bytes: int  # the cards' VRAM (0 on a unified pool: the RAM is the pool)
    margin_bytes: int = OS_RAM_MARGIN  # kept for the system (`margin_for`: 8 GiB on a pool)
    pool: bool = False  # One pool, said as such (unified memory, an APU's GTT)
    basis: Basis = Basis.BOX

    @property
    def need_bytes(self) -> int:
        return self.weights_bytes + self.cache_bytes + self.margin_bytes

    @property
    def fits(self) -> bool:
        return self.need_bytes <= self.ram_bytes + self.vram_bytes

    @property
    def box_ram_gb(self) -> int | None:
        """The smallest RAM a box with these cards runs it with; None beyond 2 TB."""
        short = self.need_bytes - self.vram_bytes
        return next((gb for gb in BOX_RAM_GB if gb * GB >= short), None)


def floor_of(  # noqa: PLR0913 - the floor's figures, then how it is said
    quant: str, estimate: MemoryEstimate, disk_bytes: int, ram: int, vram: int,
    *, margin: int = OS_RAM_MARGIN, pool: bool = False, basis: Basis = Basis.BOX,
) -> Floor:  # fmt: skip
    """On a pool (`SHARED_MEMORY`) the caller passes `vram=0`: the pool is the RAM, once."""
    cache = estimate.total_bytes - estimate.weights_bytes
    return Floor(quant, estimate.weights_bytes, cache, disk_bytes, ram, vram, margin, pool, basis)


def floor_words(name: str, floor: Floor, smallest: Floor | None, largest: str | None = None) -> str:
    """`DeepSeek-V4.1-Flash UD-IQ1_S needs about 127 GB in RAM and on the card …`; with
    `largest` (an explicit pick) the largest quant this machine holds instead."""
    disk = f", and {_gb(floor.disk_bytes)} more stay on disk" if floor.disk_bytes else ""
    where = (
        "in the unified memory"
        if floor.pool
        else "in RAM"
        if floor.basis is Basis.RAM
        else "in RAM and on the card"
    )
    need, ram = _gb(floor.need_bytes), _gb(floor.ram_bytes)
    if need == ram:  # 131 GB against 131 GB reads as a fit; a decimal tells them apart
        need, ram = f"{floor.need_bytes / GB:.1f} GB", f"{floor.ram_bytes / GB:.1f} GB"
    lines = [
        f"{name} {floor.quant} needs about {need} {where} "
        f"({_gb(floor.weights_bytes)} of weights, {_gb(floor.cache_bytes)} of cache and "
        f"buffers, {_gb(floor.margin_bytes)} kept for the system{disk}); "
        f"{_has(floor, ram)}.",
        f"The smallest quant that would fit: {smallest.quant if smallest else 'none'}.",
    ]
    if largest is not None:
        lines[-1] = f"The largest quant this machine holds at this context: {largest}."
        return " ".join(lines)
    if smallest is None:
        box = floor.box_ram_gb
        lines.append(
            f"A box with {box} GB of {_box_has(floor)} runs it."
            if box is not None
            else "No box of RAM this tool knows runs it."
        )
    return " ".join(lines)


def _box_has(floor: Floor) -> str:
    if floor.pool:
        return "unified memory"
    return "RAM" if floor.basis is Basis.RAM else "RAM and these cards"


def _has(floor: Floor, ram: str) -> str:
    """What the floor was weighed against, said as what it is."""
    has = (
        f"{ram} of unified memory"
        if floor.pool
        else f"{ram} of RAM and {floor.vram_bytes / GIB:.0f} GiB of VRAM"
    )
    match floor.basis:
        case Basis.GIVEN:
            return (
                f"the machine `--vram` describes has {has} (`--ram`, 14 GiB unless given)"
                if floor.pool
                else f"the machine `--vram` describes has {ram} of RAM (`--ram`, 14 GiB "
                f"unless given) and {floor.vram_bytes / GIB:.0f} GiB of VRAM"
            )
        case Basis.FREE:
            return f"{has} {'is' if floor.pool else 'are'} free now"
        case Basis.RAM:
            return f"this machine has {ram} of RAM and no card"
        case _:
            return f"this machine has {has}"


def _gb(value: int) -> str:
    return f"{value / GB:.0f} GB" if value >= 10 * GB else f"{value / GB:.1f} GB"
