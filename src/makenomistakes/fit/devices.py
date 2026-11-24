"""The device table: which row prices a card. `device_for` takes
the row whose name words match the card's name most specifically ("RTX 4070 Ti SUPER" beats
"RTX 4070 Ti" beats "RTX 4070"; a word never matches inside another, so "A100" is not "A1000"),
then the variant whose memory matches the card's (an RTX 3060 12 GB and 8 GB differ); else the
family's default by compute capability or gfx target; else the vendor's default. Each says
which in `chosen_by` and `source`, and a plan says it in one line (`device_note`).

A default row carries the 4070's efficiencies and today's fallback figures (400 GB/s, 25
TFLOPS): a card the table does not know plans as it did before the table.
`DeviceRow.with_learned` overlays what this box's own readings taught (`machine_facts`;
never another box's)."""

from __future__ import annotations

import re
from typing import Protocol

from makenomistakes.fit.device_row import (
    FOUR_070_SOURCE,
    ChosenBy,
    DeviceRow,
)
from makenomistakes.fit.device_table import ROWS
from makenomistakes.hardware.fingerprint import MemoryKind, Vendor

DEFAULT_GPU_BANDWIDTH = 400 * 10**9
DEFAULT_GPU_TFLOPS = 25.0
GIB = 2**30

# NVIDIA's compute capability → its family (llama.cpp's and the images' floors follow it,
# `hardware.cards`); AMD's gfx target prefix → its family.
_NVIDIA_FAMILIES = {
    "6.0": "pascal-dc",
    "6.1": "pascal",
    "7.0": "volta",
    "7.5": "turing",
    "8.0": "ampere-dc",
    "8.6": "ampere",
    "8.7": "orin",
    "8.9": "ada",
    "9.0": "hopper",
    "10.0": "blackwell-dc",
    "10.3": "blackwell-dc",
    "12.0": "blackwell",
    "12.1": "gb10",
}
_AMD_FAMILIES = (
    ("gfx1151", "strix-halo"),
    ("gfx115", "rdna3.5"),
    ("gfx120", "rdna4"),
    ("gfx110", "rdna3"),
    ("gfx103", "rdna2"),
    ("gfx101", "rdna1"),
    ("gfx908", "cdna1"),
    ("gfx90a", "cdna2"),
    ("gfx94", "cdna3"),
    ("gfx906", "gcn5"),
)
_UNIFIED_FAMILIES = {"gb10", "orin"}
_CARVE_OUT_FAMILIES = {"strix-halo", "rdna3.5"}  # APUs that plan in their GTT

TABLE: tuple[DeviceRow, ...] = ROWS


class CardLike(Protocol):
    """What `device_for` reads of a card (`hardware.probe.Gpu`)."""

    @property
    def name(self) -> str: ...
    @property
    def vram_total_bytes(self) -> int: ...
    @property
    def compute_cap(self) -> str | None: ...


def words(text: str) -> str:
    """A name as space-separated lowercase words: `NVIDIA A100-SXM4-80GB` → `nvidia a100 sxm4
    80gb`, so a needle matches whole words only."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _specificity(row: DeviceRow, name: str) -> int:
    """How much of the name the row's best alternative matches (0: none)."""
    padded = f" {name} "
    best = 0
    for needle in row.match:
        found = words(needle)
        if found and f" {found} " in padded:
            best = max(best, len(found))
    return best


def _vendor_of(name: str) -> Vendor:
    if re.search(r"\b(amd|radeon|instinct)\b", name):
        return Vendor.AMD
    if re.search(r"\b(intel|arc)\b", name):
        return Vendor.INTEL
    return Vendor.NVIDIA


def device_for(
    card: CardLike | str, *, compute: str | None = None, gfx: str | None = None
) -> DeviceRow:
    """The row that prices this card: by name, then its variant by memory; else the family's
    default (compute capability for NVIDIA, gfx target for AMD); else the vendor's."""
    if isinstance(card, str):
        name, vram, cap = card, None, compute
    else:
        name, vram, cap = card.name, card.vram_total_bytes or None, card.compute_cap or compute
        gfx = gfx or getattr(card, "gfx_target", None)  # an AMD card's
    found = words(name)
    scored = [(s, row) for row in TABLE if (s := _specificity(row, found))]
    if scored:
        top = max(s for s, _ in scored)
        best = [row for s, row in scored if s == top]
        return _variant(best, vram)
    return family_default(_vendor_of(found), cap, gfx, name)


def _variant(rows: list[DeviceRow], vram: int | None) -> DeviceRow:
    """The row whose memory is the card's (nvidia-smi's total, to the GB), else the first."""
    if vram is not None and len(rows) > 1:
        gb = round(vram / GIB)
        exact = [r for r in rows if r.memory_gb is not None and abs(r.memory_gb - gb) <= 1]
        if exact:
            return exact[0]
    return rows[0]


def family_default(
    vendor: Vendor, compute: str | None, gfx: str | None, name: str = ""
) -> DeviceRow:
    """A card the table does not name: its family's row if the compute capability or gfx
    target says which, else its vendor's, each with the 4070's efficiencies."""
    family = _family_of(vendor, compute, gfx)
    label = name.strip() or f"an unknown {vendor.value.upper()} card"
    if family is None:
        return DeviceRow(
            match=(),
            vendor=vendor,
            family="unknown",
            bandwidth=DEFAULT_GPU_BANDWIDTH,
            tflops=DEFAULT_GPU_TFLOPS,
            source=f"vendor default: {label} is not in the table, so 400 GB/s is assumed",
            name=label,
            compute=compute,
            gfx=gfx,
            chosen_by=ChosenBy.VENDOR,
        )
    said = f"compute {compute}" if vendor is Vendor.NVIDIA else gfx
    return DeviceRow(
        match=(),
        vendor=vendor,
        family=family,
        memory=_memory_of(family),
        bandwidth=DEFAULT_GPU_BANDWIDTH,
        tflops=DEFAULT_GPU_TFLOPS,
        source=(
            f"family default ({family}, {said}): {label} is not in the table, so 400 GB/s "
            "is assumed"
        ),
        name=label,
        compute=compute,
        gfx=gfx,
        chosen_by=ChosenBy.FAMILY,
    )


def _memory_of(family: str) -> MemoryKind:
    if family in _UNIFIED_FAMILIES:
        return MemoryKind.UNIFIED
    return MemoryKind.CARVE_OUT if family in _CARVE_OUT_FAMILIES else MemoryKind.DISCRETE


def name_for_pci(
    device_id: int, vram: int | None = None, vendor: Vendor = Vendor.AMD
) -> str | None:
    """The card's name from its PCI device id (the AMD probe's word for a name when
    neither amd-smi nor lspci names one card), the variant by memory where several share the
    id (the RX 7900 XT and XTX); None when no row carries it."""
    rows = [r for r in TABLE if r.vendor is vendor and device_id in r.pci_ids]
    if not rows:
        return None
    name = re.sub(r"\s\d+ GB$", "", _variant(rows, vram).name)  # the size is the card's own
    return name if name.lower().startswith(vendor.value) else f"AMD {name}"


def _family_of(vendor: Vendor, compute: str | None, gfx: str | None) -> str | None:
    if vendor is Vendor.NVIDIA and compute:
        return _NVIDIA_FAMILIES.get(compute.strip())
    if vendor is Vendor.AMD and gfx:
        return next((fam for prefix, fam in _AMD_FAMILIES if gfx.startswith(prefix)), None)
    return None


def priced_words(row: DeviceRow, *, here: bool = True) -> str:
    """How the plan's card is priced, in one line: the row, its bandwidth and its basis;
    `here` False for a `--vram` plan of another machine (no reading of it can correct it)."""
    gbps = f"{row.bandwidth / 1e9:,.0f} GB/s"
    name = row.name or "this card"
    if row.learned is not None:
        launch = " and launch term" if row.learned.launch_seconds is not None else ""
        return (
            f"{name}: {gbps}, decode efficiency{launch} learned on this box "
            f"({row.efficiency:.2f} from {row.learned.readings} readings)"
        )
    if row.efficiency_source == FOUR_070_SOURCE:
        return f"{name}: {gbps}, measured on {FOUR_070_SOURCE}"
    later = (
        "no reading on this box yet; `mnm try --speed` and `mnm tune` correct it"
        if here
        else "a plan for another machine, priced from the table alone"
    )
    if row.chosen_by is ChosenBy.VENDOR:
        return (
            f"{name}: priced as an unknown card ({gbps} assumed, the 4070's efficiencies): {later}"
        )
    if row.chosen_by is ChosenBy.FAMILY:
        return (
            f"{name}: priced from the {row.family} family's default ({gbps} assumed, the "
            f"4070's efficiencies): {later}"
        )
    return f"{name}: {gbps}, priced from its family row ({row.family}): {later}"


def device_note(row: DeviceRow, *, here: bool = True) -> str:
    """The line under a plan; `here` False for a `--vram` plan of another card."""
    note = priced_words(row, here=here)
    if row.memory is MemoryKind.UNIFIED:
        note += "; unified memory, its row from documentation until a box of it runs"
    if row.vendor is Vendor.AMD and not row.measured:  # No AMD card has run mnm yet
        note += "; unmeasured on this card"
        if row.memory is MemoryKind.CARVE_OUT:
            note += " (an APU: planned in its GTT, the same memory as the RAM)"
    return note
