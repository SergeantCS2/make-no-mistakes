"""What the doctor and setup read about the box beyond the cards: the CPU's cores, NUMA
nodes and CCDs (`cpu_topology`) and the RAM's DIMMs and channels (`ram_slots`). Kept out of
`probe` (a report every command builds): only `mnm doctor`, the web's Doctor tab and setup's
first section ask for it. `ROOT` is where sysfs is read from (another root can stand in)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from makenomistakes.hardware.cpu_topology import CpuTopology, read_cpu
from makenomistakes.hardware.probe import CommandRunner
from makenomistakes.hardware.ram_slots import RamSlots, read_slots

ROOT = Path("/")


class BoxFacts(BaseModel):
    cpu: CpuTopology | None = None
    slots: RamSlots | None = None
    wsl: bool = False  # the slots are Windows' to read: the VM has none


def read_box(runner: CommandRunner, is_wsl: bool, root: Path | None = None) -> BoxFacts:
    where = root or ROOT
    return BoxFacts(
        cpu=read_cpu(runner, where, wsl=is_wsl),
        slots=None if is_wsl else read_slots(runner, where),
        wsl=is_wsl,
    )
