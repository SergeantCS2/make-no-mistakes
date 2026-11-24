"""A box without a card (the CPU backend): every model planned in RAM. The RAM is the
one pool, like a unified card's (`plan.unified_budget`): what is free is MemAvailable less the
system's margin (`OS_RAM_MARGIN`), nothing beside it to offload to, the RAM floor counting it
once, and every profile by quant as on a card (Quality, Balanced, Speed). The recipe renders
`-ngl 0` (every block in RAM: `recipes.llamacpp`), `-t` the physical cores and `--numa
distribute` on a box of several NUMA nodes (`--numa distribute` took a 4-node Rome from 1.96
to 4.69 tok/s; one node needs nothing).

**Decode speed** is the weights a token reads over the rate llama.cpp's CPU path reads them at:
the box's measured bandwidth x `LLAMA_RAM_SHARE` (else `CPU_BANDWIDTH`), capped by the
cores (`CPU_CORE_RATE` a physical core) when there are few, plus a cost per block
(`CPU_LAUNCH_SECONDS`). Both constants were fitted on 4 vCPUs of a shared Xeon with no card,
with the official `server` image: Qwen3-0.6B Q4_K_M and Qwen3-1.7B Q4_K_M decoding with every
block in RAM."""

from __future__ import annotations

from makenomistakes.fit.budget import Budget
from makenomistakes.fit.cpu_rates import CPU_BANDWIDTH, LLAMA_RAM_SHARE, CpuSide
from makenomistakes.fit.device_row import DeviceRow
from makenomistakes.fit.ram_floor import OS_RAM_MARGIN
from makenomistakes.hardware.fingerprint import CpuPrint, MemoryKind, Vendor
from makenomistakes.hardware.ram_platforms import RamPlatform, cpu_note

# Bytes a second one physical core's share of llama.cpp's CPU decode reads (quantised dot
# products, one thread a core), and the fixed cost per block per token, fitted to two readings
# on 4 vCPUs of a shared Xeon (`CPU_SOURCE`, `mnm try --speed`, 400 tokens, 4 threads, the
# official `server` image of 2026-10-27): Qwen3-0.6B Q4_K_M 35.9 tok/s (301.4 MB streamed a
# token, 28 blocks) and Qwen3-1.7B Q4_K_M 16.1 tok/s (916.8 MB, 28 blocks): 615.4 MB more in
# 34.3 ms more is 17.96 GB/s over 4 cores, and 11.1 ms left over 28 blocks is 0.40 ms a block.
CPU_CORE_RATE = 4.49 * 10**9
CPU_LAUNCH_SECONDS = 0.40e-3
CPU_TFLOPS_PER_CORE = 0.05  # dense fp32 a core, roughly: only picture plans read it (none here)
CPU_SOURCE = "a 4-vCPU Xeon, the official `server` image"


def decode_rate(cpu: CpuPrint, ram_gbps: float | None, unmeasured: float = CPU_BANDWIDTH) -> float:
    """Bytes a second a decode reads weights at: the RAM's share, capped by the cores;
    `unmeasured` without a probe (the RAM platform's row, else the RTX 4070 machine's)."""
    ram = ram_gbps * 1e9 * LLAMA_RAM_SHARE if ram_gbps else float(unmeasured)
    cores = cpu.cores or cpu.threads
    return min(ram, cores * CPU_CORE_RATE) if cores else ram


def cpu_row(cpu: CpuPrint, rate: float) -> DeviceRow:
    """The CPU as the device the speed model prices: its rate already the effective one."""
    vendor = Vendor.AMD if (cpu.vendor or "").startswith("AuthenticAMD") else Vendor.INTEL
    cores = cpu.cores or cpu.threads or 1
    return DeviceRow(
        match=(),
        vendor=vendor,
        family="cpu",
        memory=MemoryKind.UNIFIED,  # one pool: the floor and the busy rule count it once
        bandwidth=int(rate),
        tflops=cores * CPU_TFLOPS_PER_CORE,
        efficiency=1.0,
        iq_grid_efficiency=1.0,
        launch_seconds=CPU_LAUNCH_SECONDS,
        context_bytes=0,  # no CUDA context: the runtime's own memory is the margin's
        source=CPU_SOURCE,
        efficiency_source=CPU_SOURCE,
        name=cpu.short,
    )


def cpu_budget(
    total_bytes: int,
    available_bytes: int,
    cpu: CpuPrint,
    ram_gbps: float | None = None,
    unmeasured: float = CPU_BANDWIDTH,
    margin: int = OS_RAM_MARGIN,
) -> Budget:
    """The RAM as the one pool every model is planned in (`Budget.cpu` marks it), with the
    RAM platform's rate without a probe (`unmeasured`) and what its system keeps (`margin`)."""
    rate = decode_rate(cpu, ram_gbps, unmeasured)
    row = cpu_row(cpu, rate)
    return Budget(
        ram_margin=margin,
        vram_free_bytes=max(available_bytes - margin, 0),
        vram_total_bytes=None,  # no card: no busy-card note, no correction from a card
        ram_free_bytes=0,
        ram_total_bytes=total_bytes,
        gpu_bandwidth=int(rate),
        gpu_compute=row.tflops * 1e12,
        device=row,
        cpu=CpuSide(threads=cpu.cores or cpu.threads, numa_nodes=cpu.numa_nodes),
    )


def platform_budget(
    platform: RamPlatform,
    total_bytes: int,
    available_bytes: int,
    cpu: CpuPrint,
    probe: float | None,
) -> Budget:
    """The CPU backend's budget on this box's RAM platform: its row's rate until the
    probe measures one, what its system keeps, and the rate said in the plan's notes."""
    budget = cpu_budget(
        total_bytes, available_bytes, cpu, probe, platform.layer_gbps * 1e9, platform.margin_bytes
    )
    note = cpu_note(budget.gpu_bandwidth, probe, platform.words())
    return budget.model_copy(update={"ram_note": note})


# Mesa's software device (lavapipe) decodes at about this share of the same image's CPU
# backend: Qwen3-0.6B Q4_K_M, 2.1 tok/s against 45-51 on 4 vCPUs. Only the Vulkan check
# without a card plans on it (`backend.SOFTWARE_SEAM`); a real card has its own row.
SOFTWARE_VULKAN_SHARE = 0.045
SOFTWARE_FAMILY = "lavapipe"  # the software device's row; one of `fit.budget.IN_RAM_FAMILIES`


def in_ram(budget: Budget) -> bool:
    """A plan made in RAM: the CPU backend, or Vulkan on Mesa's software device."""
    return budget.cpu is not None or budget.device.family == SOFTWARE_FAMILY


def software_vulkan_budget(total_bytes: int, available_bytes: int, cpu: CpuPrint) -> Budget:
    """Vulkan without a card (lavapipe): the RAM as the device's memory, every block "on the
    card" (`-ngl` as a full fit), at the software device's rate."""
    base = cpu_budget(total_bytes, available_bytes, cpu)
    rate = base.gpu_bandwidth * SOFTWARE_VULKAN_SHARE
    row = base.device.model_copy(
        update={"bandwidth": int(rate), "family": SOFTWARE_FAMILY, "name": "Mesa's software device"}
    )
    return base.model_copy(update={"gpu_bandwidth": int(rate), "device": row, "cpu": None})
