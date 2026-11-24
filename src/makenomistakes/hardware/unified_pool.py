"""A unified card's memory: the DGX Spark's GB10 shares the machine's RAM,
and nvidia-smi answers `[N/A]` for `memory.total` and `memory.used` (NVIDIA's known-issues
page: "Memory-Usage: Not Supported"). The probe keeps such a card at 0 bytes; here every
reader of a card's memory gets the pool instead:

- **total** is MemTotal less `UNIFIED_MARGIN`: the part of the pool mnm may plan in. The box
  hangs rather than OOM-kills when the pool runs out (a cgroup's `MemoryMax` does not count GPU
  allocations; one serving framework at 0.85 of the memory took a box down, 0.65 was safe:
  berkorbay's gist), so the margin is firm, 8 GiB against the 4 GiB
  (`OS_RAM_MARGIN`) a box with a card keeps for its system.
- **used** is MemTotal less MemAvailable: the pool in use by everything, the engines' CUDA
  allocations among it (they are the RAM's pages). nvidia-smi's
  `--query-compute-apps=used_memory` is not used first: that sum counts only CUDA processes
  (llama-swap #782) and leaves out what the system and other programs hold in the same pool,
  so on a box that hangs rather than refuses it would overstate what is free. It stands in
  only where /proc/meminfo has no MemAvailable.

So free is MemAvailable less the margin, which is what `plan.unified_budget` plans against, and
Fits now, room, the router's loads, `mnm vram` and the peak sampler read the same pool
without knowing it is one. Planned from documentation; no Spark has run it.
"""

from __future__ import annotations

from makenomistakes.hardware.fingerprint import UNIFIED_MARGIN
from makenomistakes.hardware.probe import (
    MIB,
    CommandRunner,
    Gpu,
    MemoryInfo,
    ProbeError,
)

GIB = 2**30
COMPUTE_APPS_QUERY = "--query-compute-apps=used_memory"
COMPUTE_APPS_FORMAT = "--format=csv,noheader,nounits"


def pooled(
    gpus: list[Gpu] | ProbeError, memory: MemoryInfo | None, runner: CommandRunner | None = None
) -> list[Gpu] | ProbeError:
    """`gpus` with each unified card's memory read as the pool's; the others unchanged."""
    if isinstance(gpus, ProbeError) or not any(g.unified for g in gpus) or memory is None:
        return gpus
    return [pool_card(g, memory, runner) if g.unified else g for g in gpus]


def pool_card(gpu: Gpu, memory: MemoryInfo, runner: CommandRunner | None = None) -> Gpu:
    total = max(memory.total_bytes - UNIFIED_MARGIN, 0)
    if memory.available_bytes is not None:
        used = memory.total_bytes - memory.available_bytes
    else:
        used = compute_apps_bytes(runner) or 0
    return gpu.model_copy(
        update={"vram_total_bytes": total, "vram_used_bytes": min(max(used, 0), total)}
    )


def compute_apps_bytes(runner: CommandRunner | None) -> int | None:
    """What CUDA processes hold now (`nvidia-smi --query-compute-apps=used_memory`, MiB a
    line); None when it cannot be read."""
    if runner is None:
        return None
    code, out, _ = runner.run(["nvidia-smi", COMPUTE_APPS_QUERY, COMPUTE_APPS_FORMAT])
    if code != 0:
        return None
    return parse_compute_apps(out)


def parse_compute_apps(output: str) -> int:
    """The sum of the lines' MiB; `[N/A]` and blank lines count nothing."""
    total = 0
    for line in output.splitlines():
        cell = line.strip()
        if cell.isdigit():
            total += int(cell) * MIB
    return total
