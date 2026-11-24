"""The doctor on a unified-memory box: a DGX Spark's GB10, or an APU whose
GTT is its memory (Strix Halo and the GEEKOM boxes, whose own lines `doctor_amd` keeps).

- **The pool line** replaces "VRAM in use": `128 GB unified, 97 GB available: models are
  planned against it whole`, the firm margin said, and on a GB10 what CUDA processes hold
  (nvidia-smi's `--query-compute-apps`, the one memory figure it gives there). Busy is what the
  pool holds beyond the system's margin (other processes' memory is the pool's pages); said,
  never told to close (mnm plans around it).
- **The card line** on a GB10: unified memory, the driver; a WARN on a 590 driver, on one
  forum post's word (590.48.01 broke memory accounting, 580.126.09 fixed it: reported,
  unverified), and below 580 (the CUDA 13 image needs that branch).
- **One Spark at a time:** a model across two Sparks over their ConnectX-7 link (vLLM over Ray,
  llama.cpp's RPC) is out of scope.

Written from documentation; not yet run on a Spark or a Strix Halo.
"""

from __future__ import annotations

from makenomistakes.doctor_types import Check, Status
from makenomistakes.fit.budget import BUSY_CARD_BYTES
from makenomistakes.fit.devices import device_for
from makenomistakes.fit.ram_floor import OS_RAM_MARGIN
from makenomistakes.hardware.fingerprint import UNIFIED_MARGIN
from makenomistakes.hardware.probe import CommandRunner, Gpu, HardwareReport, chosen_cards
from makenomistakes.hardware.unified_pool import compute_apps_bytes

GB = 10**9
GIB = 2**30
GPU_ROW = "NVIDIA GPU"
VRAM_ROW = "VRAM in use"
POOL_ROW = "Unified memory"
SPARKS_ROW = "Several Sparks"
CUDA13_BRANCH = 580  # `server-cuda13`'s NVIDIA_REQUIRE_CUDA names the 580 branch
REPORTED_BAD = 590  # one forum post: 590.48.01 broke the memory accounting on a GB10


def pool_card(report: HardwareReport) -> Gpu | None:
    """The chosen card that plans in the machine's RAM, if any."""
    if not isinstance(report.gpus, list):
        return None
    return next((c for c in chosen_cards(report.gpus, report.chosen) if shared(c)), None)


def shared(card: Gpu) -> bool:
    return card.unified or card.carve_out


def unified_checks(
    report: HardwareReport, checks: list[Check], runner: CommandRunner | None = None
) -> list[Check]:
    """`checks` with the pool's lines on a unified box; unchanged elsewhere."""
    card = pool_card(report)
    if card is None:
        return checks
    rows: list[Check] = []
    for check in checks:
        if check.name == VRAM_ROW:
            rows.append(pool_check(report, card, runner))
        elif check.name == GPU_ROW and card.unified:
            rows.append(gb10_check(card))
            rows.append(sparks_check())
        else:
            rows.append(check)
    return rows


def pool_check(report: HardwareReport, card: Gpu, runner: CommandRunner | None = None) -> Check:
    memory = report.memory
    if memory is None or memory.available_bytes is None:
        return Check(
            name=POOL_ROW,
            status=Status.WARN,
            detail="could not read /proc/meminfo: the pool is the RAM, and nothing is planned "
            "without it",
        )
    margin = UNIFIED_MARGIN if card.unified else OS_RAM_MARGIN
    kind = "unified" if card.unified else "shared with the APU (its GTT)"
    used = memory.total_bytes - memory.available_bytes
    detail = (
        f"{memory.total_bytes / GB:.0f} GB {kind}, {memory.available_bytes / GB:.0f} GB "
        f"available: models are planned against it whole, {margin / GIB:.0f} GiB kept for the "
        "system"
    )
    if card.unified:
        detail += " (the box hangs rather than refusing when the pool runs out)"
        held = compute_apps_bytes(runner)
        if held is not None:
            detail += f"; {held / GIB:.1f} GiB held by CUDA processes"
    beyond = used - margin
    if beyond > BUSY_CARD_BYTES:
        return Check(
            name=POOL_ROW,
            status=Status.WARN,
            detail=f"{detail}; {used / GIB:.1f} GiB of the pool in use, {beyond / GIB:.1f} GiB "
            "of it beyond the system's share",
            remedy="Models started now have that much less of the pool; a model added "
            "meanwhile was planned around it: `mnm replan --all` when more of it is free.",
        )
    return Check(name=POOL_ROW, status=Status.OK, detail=detail)


def gb10_check(card: Gpu) -> Check:
    driver = card.driver_version or "?"
    detail = (
        f"{card.name} (compute {card.compute_cap or '?'}), unified memory, driver {driver}, "
        f"priced: {device_for(card).source_word}"
    )
    major = _major(card.driver_version)
    if major is not None and major < CUDA13_BRANCH:
        return Check(
            name=GPU_ROW,
            status=Status.WARN,
            detail=detail,
            remedy=f"llama.cpp's CUDA 13 image (the GB10's, compute 12.1) needs the "
            f"{CUDA13_BRANCH} driver or newer: update DGX OS (`sudo apt update && sudo apt "
            "full-upgrade`), then reboot.",
        )
    if major is not None and major >= REPORTED_BAD:
        return Check(
            name=GPU_ROW,
            status=Status.WARN,
            detail=detail,
            remedy=f"Driver {major} on a GB10: one forum post reports that 590.48.01 broke the "
            "memory accounting and 580.126.09 fixed it (reported, unverified). If `mnm vram` "
            "or the pool line reads wrong, DGX OS's 580 driver is the one NVIDIA's own pages "
            "use.",
        )
    return Check(name=GPU_ROW, status=Status.OK, detail=detail)


def sparks_check() -> Check:
    return Check(
        name=SPARKS_ROW,
        status=Status.OK,
        detail="mnm plans and runs on this Spark alone; a model across two Sparks over their "
        "ConnectX-7 link (vLLM over Ray, llama.cpp's RPC) is not supported yet",
    )


def _major(version: str | None) -> int | None:
    head = (version or "").split(".", 1)[0]
    return int(head) if head.isdigit() else None
