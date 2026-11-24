"""The CPU as the doctor and setup say it: its name, cores and threads, NUMA nodes and
CCDs. `lscpu` first (run with `LC_ALL=C`: its labels are translated otherwise), then sysfs
(`/sys/devices/system/node/node*`, and the tuner's own reading of cores and L3 groups,
`tune.cpus`). A CCD is a group of cores sharing one L3 cache (lscpu's `L3: … (N instances)`):
llama.cpp's threads on one CCD keep their cache, which is why the tuner tries one CCD's worth.
A box with several NUMA nodes (a workstation, or a BIOS that splits one socket) reads RAM
fastest from its own node; the `numa=` setting (`numa_setting`) places each instance.

Under WSL lscpu's sockets x cores x threads describe the VM (8 vCPUs read as 4 cores of 2
threads), not the CPU: the line names the vCPUs instead, and they are the cores the tuner
tries (`tune.cpus`)."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

from makenomistakes.hardware import platform as platforms
from makenomistakes.hardware.cpu_families import ccds_of_groups
from makenomistakes.hardware.probe import CommandRunner
from makenomistakes.tune.cpus import read_layout

LSCPU = ["env", "LC_ALL=C", "lscpu"]
_INSTANCES = re.compile(r"\((\d+) instances?\)")


class CpuTopology(BaseModel):
    model: str | None = None
    sockets: int | None = None
    cores: int = 0  # physical cores; 0: unknown
    threads: int | None = None
    numa_nodes: int = 1
    ccds: int = 1  # L3 groups, halved on Zen 1 and 2 (two core complexes a CCD)
    l3_groups: int | None = None  # lscpu's L3 instances as read
    source: str = "sysfs"
    vcpus: bool = False  # WSL: `threads` are the VM's vCPUs, `cores` its topology only

    @property
    def usable_cores(self) -> int:
        """The cores llama.cpp's threads run on: every vCPU under WSL."""
        return (self.threads or self.cores) if self.vcpus else self.cores

    def words(self) -> str:
        """`AMD Ryzen 9 9950X: 16 cores, 32 threads, 1 NUMA node, 2 CCDs`; under WSL
        `AMD Ryzen 7 7700 8-Core Processor (8 vCPUs by .wslconfig): 1 NUMA node, …`."""
        parts = []
        if self.vcpus:
            name = self.model or "the CPU"
            head = f"{name} ({self.usable_cores} vCPUs by .wslconfig): "
            nodes = f"{self.numa_nodes} NUMA node{'s' if self.numa_nodes != 1 else ''}"
            return head + f"{nodes}, {self.ccds} CCD{'s' if self.ccds != 1 else ''} (L3 groups)"
        if self.cores:
            parts.append(f"{self.cores} cores")
        if self.threads:
            parts.append(f"{self.threads} threads")
        if self.sockets and self.sockets > 1:
            parts.append(f"{self.sockets} sockets")
        parts.append(f"{self.numa_nodes} NUMA node{'s' if self.numa_nodes != 1 else ''}")
        parts.append(f"{self.ccds} CCD{'s' if self.ccds != 1 else ''} (L3 groups)")
        head = f"{self.model}: " if self.model else ""
        return head + ", ".join(parts)


def parse_lscpu(text: str) -> CpuTopology | None:
    """`lscpu`'s English lines; None when they say nothing it reads."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields.setdefault(key.strip(), value.strip())
    cores_per = _int(fields.get("Core(s) per socket"))
    sockets = _int(fields.get("Socket(s)")) or 1
    if cores_per is None and "CPU(s)" not in fields:
        return None
    l3 = _INSTANCES.search(fields.get("L3", "") or fields.get("L3 cache", ""))
    return CpuTopology(
        model=fields.get("Model name") or None,
        sockets=sockets,
        cores=(cores_per or 0) * sockets,
        threads=_int(fields.get("CPU(s)")),
        numa_nodes=_int(fields.get("NUMA node(s)")) or 1,
        ccds=ccds_of_groups(fields.get("Model name"), int(l3.group(1))) if l3 else 1,
        l3_groups=int(l3.group(1)) if l3 else None,
        source="lscpu",
    )


def read_cpu(runner: CommandRunner, root: Path = Path("/"), wsl: bool | None = None) -> CpuTopology:
    """lscpu, else sysfs (never fails: an unread figure stays unknown); `wsl` None: the
    platform mnm runs on."""
    under_wsl = platforms.current().is_wsl if wsl is None else wsl
    code, out, _err = runner.run(LSCPU)
    found = parse_lscpu(out) if code == 0 else None
    if found is not None and found.cores:
        return found.model_copy(update={"vcpus": under_wsl})
    layout = read_layout(root, wsl=under_wsl)
    nodes = _numa_nodes(root / "sys/devices/system/node")
    return CpuTopology(
        model=found.model if found else None,
        cores=layout.cores,
        threads=found.threads if found else None,
        numa_nodes=nodes,
        ccds=layout.groups,
        vcpus=under_wsl,
    )


def _numa_nodes(folder: Path) -> int:
    try:
        return max(sum(1 for _ in folder.glob("node[0-9]*")), 1)
    except OSError:
        return 1


def _int(text: str | None) -> int | None:
    return int(text) if text and text.isdigit() else None
