"""The CPU as the threads stage reads it: physical cores from `/proc/cpuinfo`, and the
CCDs as the groups of cores sharing one L3 cache
(`/sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list`). A box that says nothing is one
core group with as many cores as logical CPUs.

Under WSL the cores are the VM's vCPUs, every one of them: Hyper-V shows them as pairs of
threads on half as many cores (`core id` 0-3 for 8 vCPUs), which describes the VM's
topology, not the CPU's; each vCPU is a whole logical processor of the host to llama.cpp
(else an 8-vCPU VM reads "4 cores" and the tuner tries 4 threads)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from makenomistakes.hardware import platform as platforms
from makenomistakes.hardware.cpu_families import ccds_of_groups
from makenomistakes.hardware.numa_nodes import cpuinfo_cores, parse_cpu_list


@dataclass(frozen=True)
class CpuLayout:
    cores: int = 0  # physical cores; 0: unknown
    groups: int = 1  # L3 groups (an AMD CCD each)
    # An Intel hybrid CPU's performance cores (`/sys/devices/cpu_core/cpus`, counted
    # as cores); None on a CPU of one kind
    performance: int | None = None

    def choices(self) -> list[int]:
        """The thread counts the stage tries: all cores, two fewer, one CCD's worth, and on a
        hybrid Intel CPU its performance cores alone (the efficient cores can hold a
        decode's threads back)."""
        if self.cores <= 0:
            return []
        found = [self.cores]
        if self.cores - 2 >= 2:
            found.append(self.cores - 2)
        if self.groups >= 2:
            found.append(max(self.cores // self.groups, 1))
        if self.performance and 0 < self.performance < self.cores:
            found.append(self.performance)
        return sorted(set(found), reverse=True)


def read_layout(root: Path = Path("/"), wsl: bool | None = None) -> CpuLayout:
    """`wsl` None: the platform mnm runs on (`hardware.platform.current`)."""
    under_wsl = platforms.current().is_wsl if wsl is None else wsl
    cores = _physical_cores(root / "proc" / "cpuinfo", vcpus=under_wsl)
    groups = ccds_of_groups(_model(root), _l3_groups(root / "sys/devices/system/cpu"))
    return CpuLayout(cores or os.cpu_count() or 0, groups, performance_cores(root))


def _model(root: Path) -> str | None:
    try:
        text = (root / "proc" / "cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "model name":
            return value.strip()
    return None


def performance_cores(root: Path) -> int | None:
    """An Intel hybrid CPU's performance cores: the logical CPUs the kernel lists under
    `cpu_core`, counted as cores by `/proc/cpuinfo`; None on a CPU of one kind."""
    try:
        listed = (root / "sys/devices/cpu_core/cpus").read_text(encoding="utf-8")
        (root / "sys/devices/cpu_atom/cpus").read_text(encoding="utf-8")
        text = (root / "proc" / "cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return None
    cpus = parse_cpu_list(listed) or []
    known = cpuinfo_cores(text)
    found = {known[c] for c in cpus if c in known}
    return len(found) or None


def _physical_cores(path: Path, vcpus: bool = False) -> int:
    """The physical cores; with `vcpus` (WSL) every logical CPU."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return 0
    pairs: set[tuple[str, str]] = set()
    logical = 0
    for block in text.split("\n\n"):
        fields = dict(
            (k.strip(), v.strip())
            for k, _, v in (line.partition(":") for line in block.splitlines())
        )
        if "processor" not in fields:
            continue
        logical += 1
        if "core id" in fields:
            pairs.add((fields.get("physical id", "0"), fields["core id"]))
    return logical if vcpus else len(pairs) or logical


def _l3_groups(folder: Path) -> int:
    try:
        lists = {
            p.read_text(encoding="utf-8").strip()
            for p in folder.glob("cpu[0-9]*/cache/index3/shared_cpu_list")
        }
    except OSError:
        return 1
    return max(len(lists), 1)
