"""The doctor's lines about the RAM platform and a server board, after the CPU
and RAM slots lines:

- **RAM platform:** the row this box sits on and how plans price RAM-held experts on it now
  ("priced at 50 GB/s for an EPYC 7532 with 4 CCDs until this box measures"), on every box;
- **RAM channels:** a server or workstation socket with DIMMs in fewer channels than it has
  (6 of 8 reads 25% less), and a CPU whose CCDs read less than its channels (a 4-CCD EPYC);
- **Huge pages:** transparent huge pages `never`;
- **NUMA:** several nodes: how many a socket (NPS), automatic NUMA balancing on (the guides turn
  it off), the page cache `distribute` wants dropped (root's command, never run), two sockets
  (slower than one in mainline llama.cpp), and each instance keeping weights in RAM with no
  `numa` setting (`mnm set <name> numa=auto`);
- **Server board:** on a board whose DMI vendor is a server maker's, the BIOS checklist as far as
  sysfs reads it (above-4G decoding from the cards' memory windows, NPS from the nodes, each
  card's link from the Cards line) and what it cannot (risers forced to Gen3, PSB vendor locks,
  the BMC's fan thresholds: `ipmitool`, a GUIDE note, never run).

Written from documentation; not yet run on a server board."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from makenomistakes.config import Settings
from makenomistakes.doctor_types import Check, Status
from makenomistakes.engine_args import LlamaCppArgs
from makenomistakes.fit.estimate import FitLevel
from makenomistakes.hardware import numa_nodes
from makenomistakes.hardware.cpu_families import Kind
from makenomistakes.hardware.fingerprint import BoardPrint
from makenomistakes.hardware.numa_nodes import KernelMemory, NumaNode, read_kernel, read_nodes
from makenomistakes.hardware.probe import HardwareReport
from makenomistakes.hardware.ram_platforms import RamPlatform
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.machine_facts import MachineFactsError, load_file, platform_here
from makenomistakes.recipes import Recipe
from makenomistakes.store import StoreError

DROP_CACHES = "sync; echo 3 | sudo tee /proc/sys/vm/drop_caches"
NO_BALANCING = "sudo sysctl -w kernel.numa_balancing=0"
FOUR_GIB = 2**32
# The makers of server boards, by DMI's board vendor (Dell's and HPE's own servers, Lenovo's
# ThinkSystem): a list, matched as words at the start
SERVER_VENDORS = (
    "Supermicro",
    "ASRockRack",
    "ASRock Rack",
    "Tyan",
    "Dell",
    "HPE",
    "Hewlett Packard Enterprise",
    "Lenovo",
)
# Gigabyte's server boards by name (MZ32-AR0, MZ73-LM0, ME03-CE0, MC62-G40)
GIGABYTE_SERVER = re.compile(r"^M[CEZ]\d\d-", re.IGNORECASE)
WORKSTATIONS = (Kind.SERVER_DDR4, Kind.SERVER_DDR5, Kind.WORKSTATION_DDR4, Kind.WORKSTATION_DDR5)


@dataclass(frozen=True)
class ServerFacts:
    """What the server lines read: the platform row, the nodes, two kernel settings, the board,
    each card's memory windows (start addresses from sysfs), the saved instances, and how
    plans price RAM-held experts now (`MachineFacts.priced`)."""

    platform: RamPlatform
    priced: str
    nodes: list[NumaNode] = field(default_factory=list)
    kernel: KernelMemory = field(default_factory=KernelMemory)
    board: BoardPrint = field(default_factory=BoardPrint)
    windows: dict[str, list[int]] = field(default_factory=dict)  # card bus id -> BAR starts
    recipes: list[Recipe] = field(default_factory=list)


def server_board(board: BoardPrint) -> bool:
    """A server maker's board, by its DMI vendor (Gigabyte's by its name)."""
    vendor = (board.vendor or "").strip()
    if any(vendor.lower().startswith(v.lower()) for v in SERVER_VENDORS):
        return True
    return vendor.lower().startswith("gigabyte") and bool(GIGABYTE_SERVER.match(board.name or ""))


def server_checks(found: ServerFacts) -> list[Check]:
    platform = found.platform
    checks = [Check(name="RAM platform", status=Status.OK, detail=_platform(found))]
    if platform.kind is Kind.WSL:
        return checks
    checks += _channels(platform)
    if found.kernel.thp == "never":
        checks.append(
            Check(
                name="Huge pages",
                status=Status.WARN,
                detail="transparent huge pages are `never`: a model's weights load and read "
                "through small pages only",
                remedy="`echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled` "
                "(and `transparent_hugepage=madvise` on the kernel command line to keep it)",
            )
        )
    checks += _numa(found)
    if server_board(found.board) or platform.row.server:
        checks.append(_bios(found))
    return checks


def _platform(found: ServerFacts) -> str:
    platform = found.platform
    detail = f"{platform.words()}; {found.priced}"
    if platform.kind is Kind.WSL:
        detail += (
            "; native Linux on the same box is expected to read RAM-held experts 2-4x faster "
            "(GUIDE §27.8)"
        )
    elif platform.kind is not Kind.UNKNOWN:
        detail += f"; a card gets {platform.row.lanes}"
    return detail


def _channels(platform: RamPlatform) -> list[Check]:
    found = []
    if platform.kind in WORKSTATIONS and platform.used_channels < platform.channels:
        share = platform.used_channels / platform.channels
        found.append(
            Check(
                name="RAM channels",
                status=Status.WARN,
                detail=f"DIMMs in {platform.used_channels} of the socket's {platform.channels} "
                f"channels: experts kept in RAM read {share:.0%}"
                " of what every channel would",
                remedy=f"One identical DIMM in each of the {platform.channels} channels (the "
                "board's manual names the slots; never two a channel beside empty ones).",
            )
        )
    if platform.ccd_limited:
        found.append(
            Check(
                name="RAM channels",
                status=Status.WARN,
                detail=f"{platform.cpu}{platform.ccd_words} reads at most about "
                f"{platform.peak_gbps:.0f} GB/s: each CCD reads through one link, fewer than "
                f"its {platform.channels} channels can feed (the CCD rule)",
                remedy="A part with more CCDs fills the channels (an EPYC 7532 has 8); the plans "
                "already price this one's figure.",
            )
        )
    return found


def _numa(found: ServerFacts) -> list[Check]:
    nodes, platform = found.nodes, found.platform
    if len(nodes) < 2:
        return []
    per = len(nodes) // max(platform.sockets, 1)
    detail = (
        f"{len(nodes)} nodes, {per} a socket (NPS{per})"
        if platform.sockets == 1
        else (f"{len(nodes)} nodes on {platform.sockets} sockets")
    )
    detail += "; " + "; ".join(n.words() for n in nodes)
    checks = [
        Check(
            name="NUMA",
            status=Status.OK,
            detail=detail,
            remedy=f"Before starting an instance with numa=distribute, `{DROP_CACHES}` lets its "
            "pages spread over every node (it needs root: mnm never runs it).",
        )
    ]
    if found.kernel.numa_balancing == 1:
        checks.append(
            Check(
                name="NUMA balancing",
                status=Status.WARN,
                detail="the kernel moves pages between nodes by itself (numa_balancing=1), "
                "against llama.cpp's own placement",
                remedy=f"`{NO_BALANCING}` (and `kernel.numa_balancing=0` in /etc/sysctl.d to "
                "keep it).",
            )
        )
    if platform.sockets > 1:
        checks.append(
            Check(
                name="Two sockets",
                status=Status.WARN,
                detail="mainline llama.cpp decodes slower across two sockets than on one "
                "(a published reading: 8.89 → 5.79 tok/s); plans price one socket's RAM",
                remedy="`mnm set <name> numa=node:0` keeps a model that fits one socket's RAM "
                "there; `numa=auto` picks it.",
            )
        )
    unset = [r.name for r in found.recipes if keeps_ram(r) and r.llama_args.numa is None]
    if unset:
        checks.append(
            Check(
                name="NUMA settings",
                status=Status.WARN,
                detail=f"{', '.join(unset)} keep{'s' if len(unset) == 1 else ''} weights in "
                f"RAM across {len(nodes)} nodes with no numa setting",
                remedy=f"`mnm set {unset[0]} numa=auto` (node:N when it fits one node, else "
                "distribute); `mnm tune` tries each.",
            )
        )
    return checks


def keeps_ram(recipe: Recipe) -> bool:
    """A llama.cpp instance with weights in RAM: the CPU backend, experts, or whole blocks."""
    args = recipe.args
    if not isinstance(args, LlamaCppArgs):
        return False
    return args.gpu_layers == 0 or args.experts_in_ram or recipe.fit_level == FitLevel.OFFLOAD


def _bios(found: ServerFacts) -> Check:
    """The BIOS checklist, as far as sysfs reads it."""
    read: list[str] = []
    platform = found.platform
    if found.windows:
        high = [bus for bus, starts in found.windows.items() if any(s >= FOUR_GIB for s in starts)]
        if high:
            read.append("above-4G decoding on (a card's memory window sits above 4 GiB)")
        else:
            read.append(
                "every card's memory window sits below 4 GiB: turn on Above 4G Decoding before "
                "adding cards"
            )
    if len(found.nodes) > 1 and platform.sockets == 1:
        read.append(f"NPS{len(found.nodes)} (set numa= on each instance, or NPS1 in the BIOS)")
    elif found.nodes:
        read.append("NPS1")
    read.append("each card's PCIe generation: the Cards line")
    board = found.board.words() or "this board"
    return Check(
        name="Server board",
        status=Status.OK,
        detail=f"{board}: {'; '.join(read)}",
        remedy="Not readable here: risers that log PCIe errors forced to Gen3; one identical "
        "DIMM a channel; an EPYC or Threadripper PRO sold unlocked (PSB vendor locks); quiet "
        "fans read as stalled by the BMC (`ipmitool sensor thresh`, GUIDE §25.14).",
    )


def card_windows(root: Path, buses: Iterable[str | None]) -> dict[str, list[int]]:
    """Each card's memory windows (its BARs' start addresses) from sysfs's `resource` file;
    a card whose file cannot be read is left out."""
    found: dict[str, list[int]] = {}
    for bus in buses:
        if not bus:
            continue
        name = bus.lower()[-12:] if len(bus) > 12 else bus.lower()  # 00000000:01:00.0
        text = _read(root / "sys/bus/pci/devices" / name / "resource")
        starts = [int(line.split()[0], 16) for line in text.splitlines() if line.split()]
        if any(starts):
            found[bus] = [s for s in starts if s]
    return found


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError):
        return ""


def with_server_lines(
    settings: Settings, report: HardwareReport, checks: list[Check]
) -> list[Check]:
    """`checks` with the platform's and server board's lines after the RAM slots line (else
    after the RAM line); unchanged before a box was read (`machine.json` has none)."""
    try:
        file = load_file(settings.expanded_state_dir())
    except MachineFactsError:
        return checks
    platform = platform_here(file) if file is not None else None
    if file is None or platform is None or file.current is None:
        return checks
    try:
        recipes = InstanceService(settings).all()
    except (InstanceError, StoreError):
        recipes = []
    cards = report.gpus if isinstance(report.gpus, list) else []
    found = ServerFacts(
        platform=platform,
        priced=file.facts().priced(platform).words,
        nodes=read_nodes(),
        kernel=read_kernel(),
        board=file.current.fingerprint.board,
        windows=card_windows(numa_nodes.ROOT, (c.bus_id for c in cards)),
        recipes=recipes,
    )
    names = [c.name for c in checks]
    after = "RAM slots" if "RAM slots" in names else "RAM"
    at = names.index(after) + 1 if after in names else len(checks)
    return [*checks[:at], *server_checks(found), *checks[at:]]
