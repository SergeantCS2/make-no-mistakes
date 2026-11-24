"""`mnm doctor --deep`'s RAM line: the bandwidth measured now and saved for the
planner (`machine_facts`), which prices experts kept in RAM at no more than it;
every physical core's figure and one thread's beside it, and whole blocks kept in RAM priced
at the first x `LLAMA_RAM_SHARE`."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from makenomistakes.doctor_types import Check, Status
from makenomistakes.fit.cpu_rates import LLAMA_RAM_SHARE
from makenomistakes.hardware.ram_bandwidth import PROBE, RamProbeError, RamReading, measure_all
from makenomistakes.hardware.ram_platforms import RamPlatform
from makenomistakes.machine_facts import (
    PLAUSIBLE_RAM_GBPS,
    MachineFacts,
    MachineFactsError,
    load_file,
    platform_here,
    plausible,
    update,
    with_ram,
)

NAME = "RAM bandwidth"


def ram_check(
    state_dir: Path, measured: Callable[[], float | RamReading] = measure_all, pool: bool = False
) -> Check:
    """`measured` gives every core's reading (`measure_all`), or a bare figure (an older
    probe's, one thread: saved as such). `pool`: a unified box, where the probe reads
    the pool the models sit in and no expert or block is ever kept "in RAM": the rates those
    are priced at mean nothing there and are never said."""
    try:
        found = measured()
    except (MemoryError, RamProbeError) as exc:  # a worker that died, too
        why = (
            str(exc)
            if isinstance(exc, RamProbeError)
            else "the RAM for the copies could not be had"
        )
        return Check(
            name=NAME,
            status=Status.WARN,
            detail=f"not measured: {why}",
            remedy="Run `mnm doctor --deep` again with fewer models loaded.",
        )
    reading = found if isinstance(found, RamReading) else None
    gbps = found.gbps if isinstance(found, RamReading) else found
    if gbps <= 0:
        return Check(name=NAME, status=Status.WARN, detail="not measured: the clock read 0")
    thread = reading.one_thread_gbps if reading is not None else None
    try:
        facts = update(state_dir, with_ram(gbps, thread, PROBE if reading is not None else None))
    except (MachineFactsError, OSError) as exc:
        return Check(name=NAME, status=Status.WARN, detail=f"{gbps:.1f} GB/s, not saved: {exc}")
    if not plausible(gbps):
        return Check(
            name=NAME,
            status=Status.WARN,
            detail=f"{gbps:.1f} GB/s read + write, not saved: under {PLAUSIBLE_RAM_GBPS:g} GB/s "
            "the probe could not measure the RAM (another program copying at the same time, "
            "memory short enough that the system swapped, or a virtual machine slow to give "
            f"pages){'' if pool else '; ' + _priced(facts, state_dir)}",
            remedy="Run `mnm doctor --deep` again with fewer programs running.",
        )
    if reading is None:
        how = f"{gbps:.1f} GB/s read + write (a 1 GiB copy, best of 3)"
    else:
        cores = f"{reading.workers} core{'s' * (reading.workers != 1)}"
        how = (
            f"{gbps:.1f} GB/s read + write over {cores} (best of 3), one thread "
            f"{reading.one_thread_gbps:.1f}"
        )
        if not pool:
            how += f"; whole blocks kept in RAM priced at {gbps * LLAMA_RAM_SHARE:.1f} GB/s"
    if pool:
        return Check(
            name=NAME,
            status=Status.OK,
            detail=f"{how}; saved; this is the unified pool the "
            "models are planned in whole (nothing is kept in RAM beside it), and plans read its "
            "speed from the card's row until this box's readings refit it",
        )
    return Check(name=NAME, status=Status.OK, detail=f"{how}; saved; {_priced(facts, state_dir)}")


def _priced(facts: MachineFacts, state_dir: Path | None = None) -> str:
    """How plans price RAM-held experts now: on a box whose RAM platform is known the
    platform's words, else as before."""
    platform = _platform(state_dir) if state_dir is not None else None
    if platform is not None:
        text = f"plans: RAM-held experts {facts.priced(platform).words}"
        if facts.stale_experts:
            text += (
                f" (the {facts.expert_gbps:g} GB/s learned before mnm's expert model changed is "
                "not used until `mnm calibration` learns it again)"
            )
        return f"{text}; this box: {platform.words()}"
    rate = f"plans price experts kept in RAM at {facts.expert_rate / 1e9:.1f} GB/s"
    if facts.learned_experts:
        return f"{rate}, learned from this box's readings (`mnm calibration`)"
    if facts.stale_experts:
        return (
            f"{rate}: the {facts.expert_gbps:g} GB/s learned before mnm's expert model changed "
            "is not used until `mnm calibration` learns it again from the readings"
        )
    if facts.capped:
        return f"{rate}, the RAM's bandwidth (until `mnm calibration` learns from readings)"
    return f"{rate} (an RTX 4070 machine's figure, until `mnm calibration` learns this box's)"


def _platform(state_dir: Path) -> RamPlatform | None:
    try:
        file = load_file(state_dir)
    except MachineFactsError:
        return None
    return platform_here(file) if file is not None else None
