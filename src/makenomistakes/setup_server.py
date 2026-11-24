"""Setup on a server board: nothing more to install for inference (a native box's steps: the
NVIDIA driver, Docker and the toolkit), but the BIOS settings that decide how fast its RAM
reads, printed in the first section with what sysfs reads of them (the NUMA nodes a socket,
the channels with a DIMM). mnm changes no BIOS setting and runs no `ipmitool`: the fans'
thresholds are a GUIDE note (§25.14)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from makenomistakes.doctor_server import server_board
from makenomistakes.hardware import machine_reads
from makenomistakes.hardware.numa_nodes import read_nodes
from makenomistakes.hardware.ram_platforms import RamPlatform
from makenomistakes.machine_facts import MachineFactsError, load_file, platform_here
from makenomistakes.setup_view import Mark

if TYPE_CHECKING:
    from makenomistakes.setup_guide import Guide

SOURCE = "vendor and review figures"
# The BIOS checklist: (the setting, why)
CHECKLIST: tuple[tuple[str, str], ...] = (
    ("NPS1, or NPS4 with numa=", "NPS4 splits one socket into four nodes: `numa=auto` per "
     "instance (`--numa distribute` took a 4-node Rome from 1.96 to 4.69 tok/s)"),
    ("Every channel, one DIMM each", "eight identical DIMMs, one a channel: 6 of 8 reads 25% "
     "less, and two a channel run slower"),
    ("Above 4G Decoding on", "several large-memory cards need their windows above 4 GiB"),
    ("Risers at Gen3 if the BMC logs errors", "a riser that logs `link degraded` or PCIe "
     "errors runs clean when its slot is forced to Gen3 instead of Auto"),
    ("BIOS for the CPU", "Milan needs 2.0 or later on an H12SSL; Rome on an H11SSL needs a "
     "rev 2.0 board"),
    ("An unlocked CPU", "Dell and Lenovo fuse EPYC and Threadripper PRO parts to their boards "
     "(PSB): buy one sold unlocked"),
)  # fmt: skip


def server_section(guide: Guide) -> None:
    """The checklist, on a server maker's board or a server RAM platform; nothing elsewhere."""
    if guide.platform.is_wsl:
        return
    board = machine_reads.read_board(machine_reads.ROOT)
    platform = _platform(guide)
    if not server_board(board) and (platform is None or not platform.row.server):
        return
    nodes = read_nodes()
    name = board.words() or "This board"
    guide.view.say(
        f"{name} is a server board: nothing more to install for inference (the steps below are "
        "a native box's). The BIOS settings that decide how fast its RAM reads (mnm changes "
        f"none of them; from {SOURCE}):"
    )
    facts = [(item, why) for item, why in CHECKLIST]
    facts.insert(0, ("Read here", _read(platform, len(nodes))))
    guide.view.facts(facts)
    guide.summary.add(Mark.NOTE, "Server board", f"{name}: the BIOS checklist above")


def _read(platform: RamPlatform | None, nodes: int) -> str:
    parts = []
    if platform is not None:
        parts.append(platform.words())
        parts.append(f"DIMMs in {platform.used_channels} of {platform.channels} channels")
    sockets = platform.sockets if platform is not None else 1
    if nodes > 1:
        parts.append(f"{nodes} NUMA nodes on {sockets} socket{'s' * (sockets != 1)}")
    elif nodes == 1:
        parts.append("1 NUMA node (NPS1)")
    return "; ".join(parts) or "nothing readable without sudo"


def _platform(guide: Guide) -> RamPlatform | None:
    try:
        file = load_file(guide.settings.expanded_state_dir())
    except MachineFactsError:
        return None
    return platform_here(file) if file is not None else None


def plain_lines() -> list[str]:
    """`mnm setup --plain`'s lines after its table on a server maker's board (no guided sections
    there): the board, the nodes read, and the checklist; none elsewhere."""
    board = machine_reads.read_board(machine_reads.ROOT)
    if not server_board(board):
        return []
    nodes = len(read_nodes())
    read = f"{nodes} NUMA node{'s' * (nodes != 1)}" if nodes else "no NUMA node readable"
    lines = [
        f"{board.words() or 'This board'} is a server board ({read}): nothing more to install "
        f"for inference. The BIOS settings that decide how fast its RAM reads ({SOURCE}):"
    ]
    lines += [f"  - {item}: {why}" for item, why in CHECKLIST]
    return lines
