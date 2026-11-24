"""What the fingerprint reads beyond what the probe already has, all without sudo: the CPU
from `/proc/cpuinfo` (its model, physical cores, threads, sockets and the vector units in its
flags), the NUMA nodes from `/sys/devices/system/node/online`, the board from DMI
(`/sys/class/dmi/id/{board_vendor,board_name,product_name}`, world-readable; empty under WSL
or on a board that hides them) and the arch (`uname -m`). Every file goes through one
`FileReader` under `ROOT` (another root can stand in); nothing is run.

The cards and MemTotal come from the probe the caller already made (`HardwareReport`,
`MemorySnapshot`), the RAM's channels from `box_facts` where the doctor or setup read them,
the bandwidth from `machine_facts`. A box whose cards could not be read is `Unread`, never a
box without cards: nvidia-smi absent on a native box with no NVIDIA card on the PCI bus is
"no NVIDIA card", while a failing nvidia-smi (or one gone under WSL, or a card on the bus
without one) says why and nothing is recorded."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from makenomistakes.hardware import platform as platforms
from makenomistakes.hardware.box_facts import BoxFacts
from makenomistakes.hardware.fingerprint import (
    VECTOR_UNITS,
    BoardPrint,
    CardId,
    CpuPrint,
    MachineFingerprint,
    MemoryKind,
    PlatformPrint,
    RamPrint,
    fingerprint,
    stable_id_of,
)
from makenomistakes.hardware.numa_nodes import cpuinfo_cores, parse_cpu_list
from makenomistakes.hardware.probe import (
    NO_NVIDIA_SMI,
    FileReader,
    Gpu,
    MemoryInfo,
    ProbeError,
    read_text_or_empty,
)

ROOT = Path("/")
CPUINFO = "proc/cpuinfo"
NODES = "sys/devices/system/node/online"
CPUS = "sys/devices/system/cpu"
KERNEL = "proc/sys/kernel/osrelease"
CORE_KINDS = {"performance": "sys/devices/cpu_core/cpus", "efficient": "sys/devices/cpu_atom/cpus"}
DMI = "sys/class/dmi/id"
PCI = "sys/bus/pci/devices"
# what boards write where they name nothing ("To be filled by O.E.M." is matched apart)
FILLER = frozenset({"default string", "system product name", "system manufacturer"})
_RANGE = re.compile(r"^(\d+)(?:-(\d+))?$")
# An Arm CPU's cpuinfo has no `model name`, only its parts' MIDR numbers (Linux's
# `arch/arm64/include/asm/cputype.h`); the ones a box mnm plans for is likely to carry. The
# fingerprint's id must come from cpuinfo alone: lscpu's model, which only the doctor and setup
# read, would give the same box another id from one command to the next.
ARM_PARTS = {
    "0xd0c": "Neoverse-N1",
    "0xd40": "Neoverse-V1",
    "0xd49": "Neoverse-N2",
    "0xd4f": "Neoverse-V2",
    "0xd84": "Neoverse-V3",
    "0xd85": "Cortex-X925",
    "0xd87": "Cortex-A725",
    "0xd8e": "Neoverse-N3",
}


@dataclass(frozen=True)
class Unread:
    """The box could not be read well enough to name it: nothing is recorded."""

    reason: str


def read_fingerprint(
    cards: list[Gpu] | ProbeError,
    memory: MemoryInfo | None,
    box: BoxFacts | None = None,
    bandwidth_gbps: float | None = None,
    root: Path | None = None,
    read_text: FileReader = read_text_or_empty,
    arch: str | None = None,
) -> MachineFingerprint | Unread:
    where = root or ROOT
    listed = _cards(cards, where)
    if isinstance(listed, Unread):
        return listed
    if memory is None:
        return Unread("/proc/meminfo could not be read")
    here = platforms.current()
    slots = box.slots if box is not None else None
    ram = RamPrint(
        total_bytes=memory.total_bytes,
        channels=slots.channels if slots is not None else None,
        speed_mts=min((d.speed_mts for d in slots.dimms if d.speed_mts), default=None)
        if slots is not None
        else None,
        kind=next((d.kind for d in slots.dimms if d.kind), None) if slots is not None else None,
        bandwidth_gbps=bandwidth_gbps,
    )
    return fingerprint(
        cards=[_card(g) for g in listed],
        driver=next((g.driver_version for g in listed if g.driver_version), None),
        cpu=_cpu(parse_cpuinfo(read_text(where / CPUINFO)), box, where, read_text),
        ram=ram,
        board=read_board(where, read_text),
        platform=PlatformPrint(
            kind=here.kind.value,
            distro=here.distro.label,
            kernel=read_text(where / KERNEL).strip() or None,
            arch=arch or os.uname().machine,
        ),
    )


def _cards(cards: list[Gpu] | ProbeError, root: Path) -> list[Gpu] | Unread:
    if isinstance(cards, list):
        return cards
    absent = cards.message == NO_NVIDIA_SMI or cards.message.endswith(": not found")
    if not absent:
        return Unread(f"the cards could not be read: {cards.message}")
    if platforms.current().is_wsl:  # WSL's nvidia-smi comes with the Windows driver
        return Unread("nvidia-smi is missing under WSL: the Windows NVIDIA driver is not reached")
    if platforms.nvidia_pci_devices(root / PCI):
        return Unread("an NVIDIA card is on the PCI bus, but nvidia-smi is not installed")
    if platforms.amd_pci_devices(root / PCI):  # Sysfs shows no card the driver runs
        return Unread("an AMD card is on the PCI bus, but the amdgpu driver does not run it")
    return []  # a box without a card


def _card(gpu: Gpu) -> CardId:
    major = (gpu.driver_version or "").split(".")[0]
    memory = MemoryKind.DISCRETE
    if gpu.unified:
        memory = MemoryKind.UNIFIED
    elif gpu.carve_out:
        memory = MemoryKind.CARVE_OUT
    return CardId(
        vendor=gpu.vendor,
        stable_id=stable_id_of(gpu.uuid, gpu.bus_id, gpu.name, gpu.index, gpu.unique_id),
        uuid=gpu.uuid,
        bus_id=gpu.bus_id,
        name=gpu.name,
        memory_bytes=None if gpu.unified else gpu.vram_total_bytes,
        memory=memory,
        compute_cap=gpu.compute_cap,
        gfx_target=gpu.gfx_target,
        render_node=gpu.render_node,
        driver_major=int(major) if major.isdigit() else None,
        lanes_max=gpu.lanes_max,
        pcie_gen_max=gpu.pcie_gen_max,
        index=gpu.index,
    )


def _cpu(found: CpuPrint, box: BoxFacts | None, root: Path, read_text: FileReader) -> CpuPrint:
    """cpuinfo's reading, lscpu's (`box_facts`) for the cores where cpuinfo says nothing, the
    NUMA nodes and Intel's core kinds from sysfs. The model is cpuinfo's only (`ARM_PARTS` on
    Arm): it is the id's, and lscpu is read by the doctor and setup alone."""
    topology = box.cpu if box is not None else None
    numa = parse_node_list(read_text(root / NODES))
    kinds = {
        k: _kind_cores(read_text(root / path), root, read_text) for k, path in CORE_KINDS.items()
    }
    if numa is None and topology is not None:
        numa = topology.numa_nodes
    groups = l3_groups(root)
    if groups is None and topology is not None:
        groups = topology.l3_groups
    return found.model_copy(
        update={
            "l3_groups": groups,
            "cores": found.cores or (topology.cores if topology and topology.cores else None),
            "numa_nodes": numa,
            "core_kinds": {k: n for k, n in kinds.items() if n} if all(kinds.values()) else {},
        }
    )


def _kind_cores(listed: str, root: Path, read_text: FileReader) -> int | None:
    """One kind of Intel core's physical cores: the kernel lists its logical CPUs (a
    performance core's two threads both), so they are counted by cpuinfo's (socket, core id)
    as the tuner counts them (else an i9-14900K reads "16 performance cores"); the logical
    count where cpuinfo does not name them all."""
    cpus = parse_cpu_list(listed)
    if not cpus:
        return None
    known = cpuinfo_cores(read_text(root / "proc" / "cpuinfo"))
    if not all(c in known for c in cpus):
        return len(cpus)
    return len({known[c] for c in cpus})


def l3_groups(root: Path) -> int | None:
    """The distinct L3 caches the CPUs share (`cpu*/cache/index3/shared_cpu_list`); None where
    sysfs lists none (a VM that hides its caches)."""
    try:
        lists = {
            p.read_text(encoding="utf-8").strip()
            for p in (root / CPUS).glob("cpu[0-9]*/cache/index3/shared_cpu_list")
        }
    except OSError:
        return None
    return len(lists) or None


def parse_cpuinfo(text: str) -> CpuPrint:
    """x86's `model name`, `physical id`, `core id` and `flags`; ARM's `Features` (`sve`)."""
    model: str | None = None
    vendor: str | None = None
    flags: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    sockets: set[str] = set()
    threads = 0
    parts: list[str] = []
    for block in text.split("\n\n"):
        lines = (line.partition(":") for line in block.splitlines())
        fields = {k.strip(): v.strip() for k, _, v in lines}
        if "processor" not in fields:
            continue
        threads += 1
        model = model or fields.get("model name") or None
        vendor = vendor or fields.get("vendor_id") or None
        part = ARM_PARTS.get(fields.get("CPU part", "").lower()) if not model else None
        if part and fields.get("CPU implementer", "").lower() == "0x41" and part not in parts:
            parts.append(part)
        flags |= set((fields.get("flags") or fields.get("Features") or "").split())
        if "core id" in fields:
            socket = fields.get("physical id", "0")
            pairs.add((socket, fields["core id"]))
            sockets.add(socket)
    return CpuPrint(
        model=model or (" + ".join(parts) if parts else None),
        vendor=vendor or ("ARM" if parts else None),
        cores=len(pairs) or None,
        threads=threads or None,
        sockets=len(sockets) or None,
        vector=[unit for unit in VECTOR_UNITS if unit in flags],
    )


def parse_node_list(text: str) -> int | None:
    """`0`, `0-3`, `0-1,4-5` (the kernel's list format: NUMA nodes, CPUs) as a count; None
    when empty or unreadable."""
    count = 0
    for part in text.strip().split(","):
        found = _RANGE.match(part.strip())
        if found is None:
            return None
        first, last = int(found.group(1)), int(found.group(2) or found.group(1))
        count += last - first + 1
    return count or None


def read_board(root: Path, read_text: FileReader = read_text_or_empty) -> BoardPrint:
    """DMI's names; a value a board leaves as filler ("To be filled by O.E.M.") is unknown."""

    def value(name: str) -> str | None:
        text = read_text(root / DMI / name).strip()
        return None if not text or text.lower() in FILLER or "o.e.m" in text.lower() else text

    return BoardPrint(
        vendor=value("board_vendor"), name=value("board_name"), product=value("product_name")
    )
