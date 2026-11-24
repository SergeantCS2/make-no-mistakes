"""The box mnm runs on, as one typed record: the cards by uuid, the CPU, the RAM, the
board, the platform and the arch, with two short ids. `id` hashes only what stays put while
the box does (the cards' uuids, the CPU's model, the RAM in whole GB, the board, WSL or
native, the arch); free memory, times, the driver and the links' speeds never enter it, and
neither does `MNM_GPU` (every listed card counts, chosen or not). `cards_id` is the cards'
part alone. Nothing here reads the machine: `machine_reads` does, through its seams.

A field that could not be read is None ("unknown" in words), never an exception; a card
whose memory nvidia-smi answers `[N/A]` (a DGX Spark's GB10) is kept with `memory_bytes`
None and `memory` unified, its memory read as the machine's RAM. The backend
(`backend.decide` over the cards and `MNM_BACKEND`, filled by `machine.read_now`) is
recorded beside it, outside the id: the same box may change backend."""

from __future__ import annotations

import hashlib
import json
import math
import re
from enum import StrEnum

from pydantic import BaseModel, Field

from makenomistakes.private_ids import AMD_ID

GIB = 2**30
SCHEMA_VERSION = 1
VECTOR_UNITS = ("avx2", "avx512f", "amx_tile", "sve")  # what llama.cpp's CPU paths use
ID_LENGTH = 12
HYPER_V_BOARD = ("Microsoft Corporation", "Virtual Machine")  # WSL's VM, not the PC's board


class Vendor(StrEnum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"  # the device table names Arc cards; nothing reads them yet


class MemoryKind(StrEnum):
    DISCRETE = "discrete"
    UNIFIED = "unified"  # the card shares the machine's RAM (GB10; Strix Halo later)
    CARVE_OUT = "carve-out"  # an iGPU's slice of the RAM set in firmware (Radeon 780M)


# the kinds that plan in the machine's RAM: the floor and the busy rule count the pool once
SHARED_MEMORY = frozenset({MemoryKind.UNIFIED, MemoryKind.CARVE_OUT})
# What a unified pool (the GB10) keeps beside its models, firmer than the
# 4 GiB a box with a card keeps (`fit.ram_floor.OS_RAM_MARGIN`): the box hangs rather than
# OOM-kills when the pool runs out (`hardware.unified_pool`). An APU's carve-out keeps the
# 4 GiB: its GTT is capped by the kernel, and no hang was reported there.
UNIFIED_MARGIN = 8 * 2**30


class CardId(BaseModel):
    vendor: Vendor = Vendor.NVIDIA
    # what follows the card (`stable_id_of`): NVIDIA's uuid, else its slot, else its name
    stable_id: str
    uuid: str | None = None
    bus_id: str | None = None
    name: str
    memory_bytes: int | None = None  # None: unified memory, read as RAM
    memory: MemoryKind = MemoryKind.DISCRETE
    compute_cap: str | None = None  # NVIDIA's "8.9"
    gfx_target: str | None = None  # AMD's "gfx1201"
    driver_major: int | None = None
    lanes_max: int | None = None
    pcie_gen_max: int | None = None
    index: int = 0  # nvidia-smi's, for words only: an index moves with the slots
    render_node: str | None = None  # AMD's `/dev/dri/renderD128`: what ROCm maps

    @property
    def short(self) -> str:
        """`RTX 4070` for `NVIDIA GeForce RTX 4070`, `Radeon AI PRO R9700` for AMD's."""
        return re.sub(r"^(NVIDIA\s+|AMD\s+)?(GeForce\s+)?", "", self.name).strip() or self.name

    @property
    def link(self) -> str | None:
        """`x16 Gen4` at most; None where nvidia-smi says `[N/A]` (WSL)."""
        if self.lanes_max is None:
            return None
        return f"x{self.lanes_max}" + (f" Gen{self.pcie_gen_max}" if self.pcie_gen_max else "")

    def words(self) -> str:
        if self.memory is MemoryKind.UNIFIED or self.memory_bytes is None:
            return f"{self.short} (unified memory, read as RAM)"
        gfx = f" {self.gfx_target}" if self.gfx_target else ""  # AMD's
        if self.memory is MemoryKind.CARVE_OUT:
            pool = round(self.memory_bytes / GIB)
            return f"{self.short}{gfx} ({pool} GB of GTT, shared with the RAM)"
        size = f"{self.short} {round(self.memory_bytes / GIB)} GB{gfx}"
        return f"{size} ({self.link})" if self.link else size


def stable_id_of(
    uuid: str | None, bus_id: str | None, name: str, index: int, unique_id: str | None = None
) -> str:
    """A card's uuid follows it from slot to slot (AMD's `unique_id`); without one (an
    older nvidia-smi, an AMD card older than GFX9) its slot, else its name and place."""
    if uuid:
        return uuid
    if unique_id:
        return f"{AMD_ID}{unique_id}"
    return f"bus:{bus_id}" if bus_id else f"name:{name}#{index}"


class CpuPrint(BaseModel):
    model: str | None = None
    vendor: str | None = None  # cpuinfo's `vendor_id`: AuthenticAMD, GenuineIntel
    cores: int | None = None  # physical
    threads: int | None = None
    sockets: int | None = None
    numa_nodes: int | None = None
    # Intel's hybrid cores (`/sys/devices/cpu_core`, `cpu_atom`): {"performance": 8,
    # "efficient": 16}; empty on a CPU of one kind
    core_kinds: dict[str, int] = Field(default_factory=dict)
    vector: list[str] = Field(default_factory=list)  # of VECTOR_UNITS, as /proc/cpuinfo has
    # The groups of cores sharing one L3 cache (sysfs, else lscpu's L3 instances): an
    # AMD CCD each from Zen 3 on, two a CCD on Zen 1 and 2 (`hardware.ram_platforms`)
    l3_groups: int | None = None

    @property
    def short(self) -> str:
        """`Ryzen 7 7700` for `AMD Ryzen 7 7700 8-Core Processor`."""
        if not self.model:
            return "an unknown CPU"
        text = re.sub(r"\((R|TM|tm|r)\)", "", self.model)
        text = re.sub(r"\b(AMD|Intel|CPU|Processor)\b", "", text)
        text = re.sub(r"\b\d+-Cores?\b|@.*$", "", text)
        return " ".join(text.split()) or self.model

    def words(self, vcpus: bool = False) -> str:
        """`Ryzen 9 9950X 16c`; under WSL (`vcpus`) `Ryzen 7 7700 8 vCPUs`: the VM shows
        its vCPUs as half as many cores of two threads."""
        if vcpus and self.threads:
            return f"{self.short} {self.threads} vCPUs"
        if self.cores:
            return f"{self.short} {self.cores}c"
        return f"{self.short} {self.threads}t" if self.threads else self.short


class RamPrint(BaseModel):
    total_bytes: int  # MemTotal (the VM's under WSL)
    channels: int | None = None  # from `ram_slots` where readable (never under WSL)
    speed_mts: int | None = None
    kind: str | None = None  # DDR5
    bandwidth_gbps: float | None = None  # `mnm doctor --deep` on this box, when measured

    @property
    def sold_gb(self) -> int:
        """The GB the RAM was bought as: MemTotal is always a little under it (the kernel
        keeps some), so 23.5 GiB reads 24, and a few MiB more or less never moves it."""
        return math.ceil(self.total_bytes / GIB - 0.01)

    def words(self) -> str:
        text = f"{self.total_bytes / GIB:.1f} GiB RAM"
        return f"{text} {self.channels}ch" if self.channels else text


class BoardPrint(BaseModel):
    vendor: str | None = None
    name: str | None = None
    product: str | None = None

    @property
    def virtual(self) -> bool:
        return (self.vendor, self.product) == HYPER_V_BOARD

    def words(self) -> str | None:
        """`ASUS PRIME X670-P`; None when nothing was readable or it is WSL's virtual board."""
        if self.virtual or not (self.vendor or self.name):
            return None
        vendor = (self.vendor or "").split()[0].rstrip(",.") if self.vendor else ""
        return " ".join(w for w in (vendor, self.name or self.product or "") if w)


class PlatformPrint(BaseModel):
    kind: str  # "wsl" or "native" (`hardware.platform.Kind`)
    distro: str = ""  # Ubuntu 24.04.1 LTS
    kernel: str | None = None  # recorded, never identity: it moves with every update
    arch: str  # `uname -m`: the engine images' architecture

    def words(self) -> str:
        if self.kind == "wsl":
            return "WSL"
        return f"native {self.distro}".strip() if self.distro else "native Linux"


class MachineFingerprint(BaseModel):
    schema_version: int = SCHEMA_VERSION
    id: str
    cards_id: str
    cards: list[CardId]
    driver_version: str | None = None
    cpu: CpuPrint
    ram: RamPrint
    board: BoardPrint
    platform: PlatformPrint
    # `backend.Backend`'s value as decided when the box was read (its words, never its
    # identity); None when `MNM_BACKEND` named one these cards cannot run, or in an older record
    backend: str | None = None

    @property
    def arch(self) -> str:
        return self.platform.arch

    def words(self) -> str:
        """`RTX 3090 24 GB (x16 Gen4), Ryzen 9 7950X 16c, 62.5 GiB RAM 2ch, ProArt X670E,
        native`."""
        wsl = self.platform.kind == "wsl"
        parts = [cards_words(self.cards), self.cpu.words(vcpus=wsl), self.ram.words()]
        if (board := self.board.words()) is not None:
            parts.append(board)
        parts.append(self.platform.words())
        return ", ".join(parts)


def fingerprint(
    cards: list[CardId],
    driver: str | None,
    cpu: CpuPrint,
    ram: RamPrint,
    board: BoardPrint,
    platform: PlatformPrint,
) -> MachineFingerprint:
    """The record with its two ids (the one way to build one: the ids follow the fields)."""
    keys = sorted(card.stable_id for card in cards)
    identity = {
        "arch": platform.arch,
        "platform": platform.kind,
        "cards": keys,
        "cpu": cpu.model,
        "ram_gb": ram.sold_gb,
        "board": [board.vendor, board.name, board.product],
    }
    return MachineFingerprint(
        id=_short_hash(identity),
        cards_id=_short_hash({"cards": keys}),
        cards=sorted(cards, key=lambda c: c.index),
        driver_version=driver,
        cpu=cpu,
        ram=ram,
        board=board,
        platform=platform,
    )


def _short_hash(value: object) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:ID_LENGTH]


def cards_words(cards: list[CardId]) -> str:
    """`RTX 4070 12 GB (x16 Gen4)`; `2 x RTX 3090 24 GB`; `no card`."""
    if not cards:
        return "no card"
    counted: dict[str, int] = {}
    for card in cards:
        counted[card.words()] = counted.get(card.words(), 0) + 1
    return " + ".join(w if n == 1 else f"{n} x {w}" for w, n in counted.items())


def differences(old: MachineFingerprint, new: MachineFingerprint) -> list[str]:
    """What changed from `old` to `new`, card by card: `the cards (RTX 4070 gone, RTX 3090
    new)`, `the RAM (24 → 32 GB)`, `the CPU (…)`, `the board (…)`, `the platform (WSL →
    native)`, `the architecture (…)`. Empty when nothing a user would name did."""
    found = []
    cards = _card_changes(old.cards, new.cards)
    if cards:
        found.append(f"the cards ({', '.join(cards)})")
    if old.ram.sold_gb != new.ram.sold_gb:
        before, after = old.ram.total_bytes / GIB, new.ram.total_bytes / GIB
        found.append(f"the RAM ({before:.1f} → {after:.1f} GiB)")
    if old.cpu.model != new.cpu.model:
        found.append(f"the CPU ({old.cpu.short} → {new.cpu.short})")
    if old.board != new.board and not (old.board.virtual and new.board.virtual):
        found.append(f"the board ({_board(old)} → {_board(new)})")
    if old.platform.kind != new.platform.kind:
        found.append(f"the platform ({old.platform.words()} → {new.platform.words()})")
    if old.arch != new.arch:
        found.append(f"the architecture ({old.arch} → {new.arch})")
    return found


def _board(found: MachineFingerprint) -> str:
    return found.board.words() or "unknown"


def _card_changes(old: list[CardId], new: list[CardId]) -> list[str]:
    before = {c.stable_id: c for c in old}
    after = {c.stable_id: c for c in new}
    gone = [c for k, c in before.items() if k not in after]
    came = [c for k, c in after.items() if k not in before]
    found = []
    for card in gone:  # the same model in its place: another card of that name
        twin = next((c for c in came if c.short == card.short), None)
        if twin is not None:
            came.remove(twin)
            found.append(f"{card.short} replaced by another {card.short}")
        else:
            found.append(f"{card.short} gone")
    found += [f"{c.short} new" for c in came]
    for key, card in after.items():
        was = before.get(key)
        if was is None:
            continue
        if was.bus_id and card.bus_id and was.bus_id != card.bus_id:
            found.append(f"{card.short} moved from {was.bus_id} to {card.bus_id}")
        elif was.link and card.link and was.link != card.link:
            found.append(f"{card.short}'s link {was.link} → {card.link}")
    return found
