"""The NUMA nodes and the kernel's memory settings a server board's RAM is read through:
each node's CPUs, physical cores and memory from sysfs (`/sys/devices/system/node/node*/
{cpulist,meminfo}`, `/sys/devices/system/cpu/cpu*/topology/{core_id,physical_package_id}`),
transparent huge pages (`/sys/kernel/mm/transparent_hugepage/enabled`) and automatic NUMA
balancing (`/proc/sys/kernel/numa_balancing`). All world-readable; nothing is run (the official
llama.cpp image has no `numactl`, and mnm never runs it). A file that cannot be read is
unknown (None), never an exception. `ROOT` is where they are read from (another root can stand in
for the box's)."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

from makenomistakes.hardware.probe import FileReader, read_text_or_empty

ROOT = Path("/")
NODES = "sys/devices/system/node"
CPUS = "sys/devices/system/cpu"
CPUINFO = "proc/cpuinfo"
THP = "sys/kernel/mm/transparent_hugepage/enabled"
BALANCING = "proc/sys/kernel/numa_balancing"
_NODE = re.compile(r"^node(\d+)$")
_MEMINFO = re.compile(r"^Node\s+\d+\s+(\w+):\s+(\d+)\s*kB", re.MULTILINE)
_RANGE = re.compile(r"^(\d+)(?:-(\d+))?$")
_CHOSEN = re.compile(r"\[(\w+)\]")


class NumaNode(BaseModel):
    index: int
    cpus: str  # the kernel's list format, as Docker's `--cpuset-cpus` takes it: `0-7,32-39`
    cores: int  # physical cores among them (one thread a core is what llama.cpp's `-t` wants)
    total_bytes: int | None = None
    free_bytes: int | None = None  # MemFree: the page cache a `distribute` wants dropped is not

    def words(self) -> str:
        """`node 1: CPUs 8-15,40-47 (8 cores), 64 GiB, 60 GiB free`."""
        text = f"node {self.index}: CPUs {self.cpus} ({self.cores} cores)"
        if self.total_bytes:
            text += f", {self.total_bytes / 2**30:.0f} GiB"
        if self.free_bytes is not None:
            text += f", {self.free_bytes / 2**30:.0f} GiB free"
        return text


class KernelMemory(BaseModel):
    """The two kernel settings the doctor reads beside the nodes."""

    thp: str | None = None  # always, madvise or never
    numa_balancing: int | None = None  # 1: the kernel moves pages between nodes by itself


def parse_cpu_list(text: str) -> list[int] | None:
    """`0-3,8,10-11` as the CPUs it names; None when it is not the kernel's list format."""
    found: list[int] = []
    for part in text.strip().split(","):
        match = _RANGE.match(part.strip())
        if match is None:
            return None
        first, last = int(match.group(1)), int(match.group(2) or match.group(1))
        found += range(first, last + 1)
    return found or None


def parse_node_meminfo(text: str) -> tuple[int | None, int | None]:
    """A node's MemTotal and MemFree, bytes (its meminfo says kB)."""
    fields = {name: int(kb) * 1024 for name, kb in _MEMINFO.findall(text)}
    return fields.get("MemTotal"), fields.get("MemFree")


def read_nodes(
    root: Path | None = None, read_text: FileReader = read_text_or_empty
) -> list[NumaNode]:
    """Every node sysfs lists, in order; none where sysfs has no node folder (WSL's kernel
    without NUMA, a container that hides it)."""
    where = root or ROOT
    try:
        folders = sorted(
            (int(m.group(1)), p)
            for p in (where / NODES).iterdir()
            if (m := _NODE.match(p.name)) is not None
        )
    except OSError:
        return []
    nodes = []
    for index, folder in folders:
        text = read_text(folder / "cpulist").strip()
        cpus = parse_cpu_list(text) or []
        total, free = parse_node_meminfo(read_text(folder / "meminfo"))
        nodes.append(
            NumaNode(
                index=index,
                cpus=text,
                cores=_cores(where, cpus, read_text),
                total_bytes=total,
                free_bytes=free,
            )
        )
    return nodes


def _cores(root: Path, cpus: list[int], read_text: FileReader) -> int:
    """Physical cores: distinct (package, core) pairs, from sysfs's topology, else
    `/proc/cpuinfo`'s `physical id` and `core id`; the CPUs' count where neither says."""
    pairs = set()
    for cpu in cpus:
        folder = root / CPUS / f"cpu{cpu}" / "topology"
        core = read_text(folder / "core_id").strip()
        if not core:
            break
        pairs.add((read_text(folder / "physical_package_id").strip(), core))
    else:
        return len(pairs)
    known = cpuinfo_cores(read_text(root / CPUINFO))
    found = {known[c] for c in cpus if c in known}
    return len(found) if found and all(c in known for c in cpus) else len(cpus)


def cpuinfo_cores(text: str) -> dict[int, tuple[str, str]]:
    """Each logical CPU's (physical id, core id) as /proc/cpuinfo lists them."""
    found: dict[int, tuple[str, str]] = {}
    for block in text.split("\n\n"):
        fields = {
            k.strip(): v.strip() for k, _, v in (line.partition(":") for line in block.splitlines())
        }
        number = fields.get("processor", "")
        if number.isdigit() and "core id" in fields:
            found[int(number)] = (fields.get("physical id", "0"), fields["core id"])
    return found


def read_kernel(
    root: Path | None = None, read_text: FileReader = read_text_or_empty
) -> KernelMemory:
    where = root or ROOT
    chosen = _CHOSEN.search(read_text(where / THP))
    balancing = read_text(where / BALANCING).strip()
    return KernelMemory(
        thp=chosen.group(1) if chosen else None,
        numa_balancing=int(balancing) if balancing.isdigit() else None,
    )


def node(nodes: list[NumaNode], index: int) -> NumaNode | None:
    return next((n for n in nodes if n.index == index), None)
