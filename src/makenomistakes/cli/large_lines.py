"""The lines a large model's plan adds: the Engram table kept on disk, the
quants mnm refuses (an MXFP8 Engram table llama.cpp misreads), the draft heads mnm does not
run, and whether the installed llama.cpp image carries the model's architecture."""

from __future__ import annotations

from rich.markup import escape

from makenomistakes.engine_support import architecture_line
from makenomistakes.fit.device_row import RTX_4070, UNMEASURED, DeviceRow
from makenomistakes.fit.estimate import GIB
from makenomistakes.fit.profiles import Profile
from makenomistakes.fit.ram_floor import GB
from makenomistakes.hub import ModelFacts
from makenomistakes.hub.variants import DraftHead
from makenomistakes.plan_large import MXFP8_REFUSED, refused_variants
from makenomistakes.plan_types import Plan
from makenomistakes.supervisor import LocalImage


def engram_line(profile: Profile) -> str | None:
    """`UD-IQ1_S: its Engram table (111 GB) stays on disk, read as needed …`."""
    engram = profile.variant.engram
    if engram is None:
        return None
    return (
        f"{profile.quant}: its Engram table ({engram.size_bytes / GB:.0f} GB, "
        f"{engram.file.rsplit('/', 1)[-1]}) is downloaded with it and stays on disk, read as "
        "needed (`--lazy-mode on`): never in RAM or on the card."
    )


def download_words(profile: Profile) -> str:
    """What `add` adds to its download line for a model with an Engram table."""
    engram = profile.variant.engram
    if engram is None:
        return ""
    return f", {engram.size_bytes / GB:.0f} GB of it on disk, read as needed"


def refused_line(facts: ModelFacts) -> str | None:
    refused = refused_variants(facts)
    if not refused:
        return None
    names = ", ".join(v.quant for v in refused)
    return f"Not offered: {names}: {MXFP8_REFUSED}."


def head_lines(heads: list[DraftHead]) -> list[str]:
    """The draft heads mnm does not run, each as unused."""
    return [
        f"  {'Draft head':<15} {escape(h.file)} ({h.size_bytes / GIB:.2f} GiB, {h.kind}: unused)"
        for h in heads
    ]


def plan_lines(plan: Plan, profiles: list[Profile], image: LocalImage | None) -> list[str]:
    """Every line above for a llama.cpp plan, the architecture's first."""
    if plan.header is None:
        return []
    lines = [architecture_line(plan.header.architecture, image)]
    lines += [engram_line(p) for p in _distinct(profiles)]
    if plan.header.engram_layers:  # only a model with Engram has quants refused for it
        lines.append(refused_line(plan.facts))
    return [line for line in lines if line is not None]


def _distinct(profiles: list[Profile]) -> list[Profile]:
    seen: set[str] = set()
    kept = []
    for p in profiles:
        if p.quant not in seen:
            seen.add(p.quant)
            kept.append(p)
    return kept


PER_CARD_WORDS = "estimated per card; a box reading replaces it"


def placement_lines(profiles: list[Profile], device: DeviceRow = RTX_4070) -> list[str]:
    """What each card holds, per profile placed across cards (the same placement once):
    `balanced (UD-IQ1_S), estimated per card …:` then `  GPU0: blocks 0-8 + experts of 0-1
    (9.2 GiB)` per card and the RAM's line."""
    lines: list[str] = []
    seen: list[object] = []
    for profile in profiles:
        placement = profile.placement
        if placement is None or placement in seen:
            continue
        seen.append(placement)
        lines.append(f"{profile.name.value} ({profile.quant}), {PER_CARD_WORDS}:")
        lines += [f"  {line}" for line in placement.lines()]
        lines.append(f"  {hop_words(device)}")
    return lines


def hop_words(device: DeviceRow) -> str:
    """The crossing the placement weighed, said as what it is."""
    said = f"each crossing between cards priced at {device.hop_seconds * 1e3:.2f} ms"
    if device.hop_source == UNMEASURED:
        return f"{said} ({UNMEASURED}: a box reading replaces it)"
    return f"{said} ({device.hop_source})"
