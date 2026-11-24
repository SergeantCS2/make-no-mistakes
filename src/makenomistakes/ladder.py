"""The ladder of large models: what each known large model needs in RAM and VRAM
together, and whether this box holds it. Setup's first section and the doctor's RAM line say
it without the network: the figures are the RAM floor (`fit.ram_floor`) worked out once
from the large repos' headers and kept here as constants; when the planner changes, each row
is planned again from those headers and updated.

A row is the floor, not a plan: weights not on a card, the cache and buffers at 4,096 tokens
and the 4 GiB kept for the system must fit the RAM and the VRAM together. Holding a model is
not running it fast; `mnm fit <link>` plans the speed. The planner holds every quant it lands
on to the same floor at the plan's own context (`fit.plan_floor`): a row this box falls
short of is never planned here. RAM and disk in GB (as they are sold),
the cards in GiB (as nvidia-smi says them), as the floor's own refusal words them.
"""

from __future__ import annotations

from dataclasses import dataclass

from makenomistakes.fit.ram_floor import BOX_RAM_GB, GB, OS_RAM_MARGIN
from makenomistakes.hardware.fingerprint import UNIFIED_MARGIN

GIB = 2**30
FLOOR_CTX = 4096  # the context the floor is checked at (the planner's smallest rung)


@dataclass(frozen=True)
class Rung:
    """One model at one quant: what it needs, what it downloads, what stays on disk."""

    model: str  # the name a person reads
    reference: str  # what `mnm fit`/`mnm add` take
    quant: str
    need_gb: float  # RAM + VRAM: the RAM floor at FLOOR_CTX (weights + cache + the margin)
    download_gb: float
    disk_gb: float = 0.0  # an Engram table, read from disk as needed

    @property
    def need_bytes(self) -> int:
        return int(self.need_gb * GB)

    @property
    def label(self) -> str:
        return f"{self.model} {self.quant}"


GLM47 = "unsloth/GLM-4.7-Flash-GGUF"
BOLT = "unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF"
MISTRAL4 = "unsloth/Mistral-Small-4-119B-2603-GGUF"
DS4 = "unsloth/DeepSeek-V4-Flash-0731-GGUF"
DS41 = "unsloth/DeepSeek-V4.1-Flash-1017-GGUF"

# Smallest need first. Read from the Hub's headers on 2026-10-26; each row's
# quant is the one chosen for that model, or its smallest. Each row is the text model, as the
# large starters and the pages add them (`--type chat`): the vision projector a repo also
# ships is no part of it (Bolt's 1.7 GB and Mistral-Small-4's 0.9 GB would make Bolt
# UD-IQ3_XXS read "0.5 GB short" on an RTX 4070 box while `mnm add … --type chat` plans it
# within the same floor).
LADDER: tuple[Rung, ...] = (
    Rung("GLM-4.7-Flash", GLM47, "Q4_K_M", 23.1, 18.3),
    Rung("Nemotron-3.5-Bolt-74B", BOLT, "UD-Q2_K_XL", 33.6, 28.8),
    Rung("Nemotron-3.5-Bolt-74B", BOLT, "UD-IQ3_XXS", 36.4, 31.6),
    Rung("Mistral-Small-4-119B", MISTRAL4, "UD-IQ1_M", 37.0, 32.3),
    Rung("Nemotron-3.5-Bolt-74B", BOLT, "UD-Q4_K_XL", 55.0, 50.2),
    Rung("DeepSeek-V4-Flash-0731", DS4, "UD-IQ1_S", 87.2, 82.5),
    Rung("DeepSeek-V4.1-Flash", DS41, "UD-IQ1_S", 127.0, 233.0, disk_gb=110.6),
    Rung("DeepSeek-V4.1-Flash", DS41, "UD-Q2_K_XL", 204.3, 334.8, disk_gb=135.2),
)


# A unified pool keeps 8 GiB for its system, not the floor's 4 (`fit.ram_floor`): the
# rungs' figures carry the 4, so a pool adds the difference
POOL_EXTRA = UNIFIED_MARGIN - OS_RAM_MARGIN


def holds(rung: Rung, ram_bytes: int, vram_bytes: int, pool: bool = False) -> bool:
    """`pool`: one unified pool, its size as `ram_bytes` and `vram_bytes` 0: the
    memory counted once, with the pool's firmer margin."""
    need = rung.need_bytes + (POOL_EXTRA if pool else 0)
    return need <= ram_bytes + vram_bytes


def pool_words(pool_bytes: int) -> str:
    """`this box has 128 GB of unified memory: the cards and the RAM are one pool`."""
    return f"this box has {pool_bytes / GB:.0f} GB of unified memory: the card and the RAM are one"


def box_words(ram_bytes: int, vram_bytes: int, cards: int) -> str:
    """`this box has 128 GB of RAM + 64 GiB of VRAM (4 cards): 196 GB together`."""
    where = f" ({cards} cards)" if cards > 1 else ""
    together = (ram_bytes + vram_bytes) / GB
    return (
        f"this box has {ram_bytes / GB:.0f} GB of RAM + {vram_bytes / GIB:.0f} GiB of VRAM"
        f"{where}: {together:.0f} GB together"
    )


def verdict(rung: Rung, ram_bytes: int, vram_bytes: int, pool: bool = False) -> str:
    """`needs about 127 GB: holds it` / `needs about 127 GB: 89 GB short (128 GB of RAM with
    these cards holds it)`; an Engram table's disk part said after. `pool`: as `holds`."""
    extra = POOL_EXTRA if pool else 0
    need = f"needs about {(rung.need_bytes + extra) / GB:.0f} GB"
    disk = f", and {rung.disk_gb:.0f} GB on disk" if rung.disk_gb else ""
    if holds(rung, ram_bytes, vram_bytes, pool):
        return f"{need}{disk}: holds it"
    short = (rung.need_bytes + extra - ram_bytes - vram_bytes) / GB
    want = rung.need_bytes + extra
    box = next((gb for gb in BOX_RAM_GB if gb * GB + vram_bytes >= want), None)
    what = "of unified memory" if pool else "of RAM with these cards"
    with_ram = f"; {box} GB {what} holds it" if box else ""
    return f"{need}{disk}: {max(short, 0.1):.1f} GB short{with_ram}"


def rows(ram_bytes: int, vram_bytes: int, pool: bool = False) -> list[tuple[str, str]]:
    """(model and quant, verdict) for every rung, smallest need first."""
    return [(r.label, verdict(r, ram_bytes, vram_bytes, pool)) for r in LADDER]


def more_ram_buys(ram_bytes: int, vram_bytes: int, ram_max_bytes: int) -> list[Rung]:
    """The rungs this box misses now and would hold with `ram_max_bytes` of RAM (under WSL:
    what `.wslconfig` could give from the PC's RAM)."""
    return [
        r
        for r in LADDER
        if not holds(r, ram_bytes, vram_bytes) and holds(r, ram_max_bytes, vram_bytes)
    ]


@dataclass(frozen=True)
class CheapBox:
    """A server box of used parts: CPU, RAM and board, with one card; what it holds is the
    rungs its RAM and card hold together (`holds`), never a speed. Its system keeps 8 GiB of
    the RAM, not the rungs' 4 (`SERVER_EXTRA`)."""

    name: str  # "$2,000-2,500"
    parts: str
    ram_gb: int
    card_gib: int
    cost: str
    expert_gbps: float  # the RAM platform's default until measured (`hardware.ram_platforms`)

    @property
    def ram_bytes(self) -> int:
        return self.ram_gb * GB

    @property
    def card_bytes(self) -> int:
        return self.card_gib * GIB

    def held(self) -> list[Rung]:
        held = self.ram_bytes + self.card_bytes
        return [r for r in LADDER if r.need_bytes + SERVER_EXTRA <= held]

    def words(self) -> str:
        """`$2,000-2,500: EPYC 7532 + H12SSL-i + 8 x 32 GB DDR4 (256 GB) with a 24 GiB card
        ($1,880-2,470 used [est]): holds … ; experts priced at 70 GB/s until measured`."""
        largest = self.held()[-1].label if self.held() else "none of the rungs"
        return (
            f"{self.parts} ({self.ram_gb} GB) with a {self.card_gib} GiB card, {self.cost} "
            f"[est]: holds up to {largest}; experts in RAM priced at {self.expert_gbps:g} GB/s "
            "until it measures"
        )


# a server board's system keeps 8 GiB (`hardware.ram_platforms.SERVER_MARGIN`), the rungs 4
SERVER_EXTRA = 4 * GIB
# CPU, RAM and board used, prices as read on 2026-10-26 (GPUs apart: a used 3090 is
# $1,000-1,500).
CHEAP_BOXES: tuple[CheapBox, ...] = (
    CheapBox("~$1,000", "EPYC 7532 + H12SSL-i + 8 x 8 GB DDR4", 64, 24, "$1,030-1,180", 70.0),
    CheapBox(
        "~$2,000-2,500", "EPYC 7532 + H12SSL-i + 8 x 32 GB DDR4", 256, 24, "$1,880-2,470", 70.0
    ),
    CheapBox(
        "~$4,000",
        "EPYC 7532 + H12SSL-i or ROMED8-2T + 8 x 64 GB DDR4",
        512,
        24,
        "$3,230-4,150",
        70.0,
    ),
)


def box_rows() -> list[tuple[str, str]]:
    """(the box's price, what it holds) for setup's ladder and the GUIDE."""
    return [(b.name, b.words()) for b in CHEAP_BOXES]
