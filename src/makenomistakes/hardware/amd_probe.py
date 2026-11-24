"""AMD cards, read without ROCm on the host: sysfs first,
`amd-smi` only for what sysfs does not say. Every read goes through a `FileReader` under `ROOT`
(another root can stand in) or a `CommandRunner`; a field that cannot be read is None.

- **The cards:** `/sys/class/drm/card*/device` whose `vendor` reads `0x1002`, one per PCI
  address; memory from the amdgpu driver's files (bytes; the kernel's `driver-misc` docs):
  `mem_info_vram_total`/`_used`, `mem_info_vis_vram_total` (what the CPU sees: smaller than the
  VRAM when Resizable BAR is off), `mem_info_gtt_total`/`_used` (system memory the card maps),
  `gpu_busy_percent`, `unique_id` (GFX9 and newer; a secret like NVIDIA's uuid).
- **An APU** (`mem_info_vram_total` at most 1 GiB beside a GTT many times it: a Strix Halo's
  512 MB carve-out and its 120 GB GTT) plans in its GTT: the card's figures are the carve-out
  plus the GTT (`Gpu.carve_out`), the same memory as the RAM (`plan.carve_out_budget`).
- **The render node:** `/sys/class/drm/renderD*/device` resolving to the same PCI address.
- **The gfx target:** KFD's topology (`/sys/class/kfd/kfd/topology/nodes/*/properties`:
  `gfx_target_version` as major·10000 + minor·100 + stepping, the node's PCI address as
  `domain` and `location_id` = bus << 8 | device << 3 | function: `kfd_topology.c`), else
  `amd-smi static --json`'s `asic.target_graphics_version`.
- **The name:** `amd-smi`'s `asic.market_name`, else lspci's device string, else the PCI id
  in the device table's rows (`fit.devices.name_for_pci`), else "AMD card 1002:<id>".
- **The PCIe link:** never the card's own node, which on an AMD board reads the card's
  internal switch (an RX 6800 XT showed Gen 1: drm/amd #1967): the topmost AMD bridge above
  the card (the switch's upstream port, whose link is the slot's), else the port above it,
  else `amd-smi`'s `bus.max_pcie_width`/`pcie_interface_version`; unknown when none says.

`amd-smi` and `lspci` answers are kept for the process (they do not change while it runs)."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from makenomistakes.hardware.fingerprint import Vendor
from makenomistakes.hardware.probe import (
    NO_NVIDIA_SMI,
    CommandRunner,
    FileReader,
    Gpu,
    ProbeError,
    read_text_or_empty,
)

# Not in the GUIDE: a run without the card can point the probe at a recorded sysfs tree
# to stand an R9700 pair in for it (the engines then run on Mesa's software device)
SYSFS_SEAM = "MNM_AMD_SYSFS"
ROOT = Path(os.environ.get(SYSFS_SEAM) or "/")
DRM = "sys/class/drm"
KFD_NODES = "sys/class/kfd/kfd/topology/nodes"
AMD = "0x1002"
GIB = 2**30
APU_VRAM_MAX = GIB  # an APU's carve-out is at most this (Strix Halo's minimum: 512 MB)
APU_GTT_RATIO = 4  # and its GTT many times it
AMD_SMI_STATIC = ["amd-smi", "static", "--asic", "--bus", "--json"]
LINK_GENS = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6}
_CARD = re.compile(r"^card\d+$")
_RENDER = re.compile(r"^renderD\d+$")
_QUOTED = re.compile(r'"([^"]*)"')
_cache: dict[tuple[str, str], object] = {}
_lock = threading.Lock()
_failed: dict[tuple[str, str], float] = {}  # a failed read: asked again after this
RETRY_SECONDS = 60.0


@dataclass(frozen=True)
class Link:
    lanes: int | None = None
    lanes_max: int | None = None
    gen: int | None = None
    gen_max: int | None = None


@dataclass(frozen=True)
class SmiCard:
    """What `amd-smi static --json` says of one card."""

    bdf: str
    name: str | None = None
    gfx: str | None = None
    lanes_max: int | None = None
    gen_max: int | None = None


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def with_amd(
    nvidia: list[Gpu] | ProbeError,
    runner: CommandRunner,
    root: Path | None = None,
    read_text: FileReader = read_text_or_empty,
) -> list[Gpu] | ProbeError:
    """nvidia-smi's cards and the AMD ones after them; nvidia-smi absent on an AMD box is not
    an error, a failing one still is (the box is not read whole then)."""
    start = max((g.index for g in nvidia), default=-1) + 1 if isinstance(nvidia, list) else 0
    amd = amd_cards(runner, root, read_text, start)
    if not amd:
        return nvidia
    if isinstance(nvidia, ProbeError):
        absent = nvidia.message == NO_NVIDIA_SMI or nvidia.message.endswith(": not found")
        return amd if absent else nvidia
    return [*nvidia, *amd]


def amd_cards(
    runner: CommandRunner,
    root: Path | None = None,
    read_text: FileReader = read_text_or_empty,
    start: int = 0,
) -> list[Gpu]:
    """Every AMD card sysfs shows, in PCI order, numbered from `start`."""
    where = root or ROOT
    devices = _devices(where, read_text)
    if not devices:
        return []
    renders = _render_nodes(where, read_text)
    targets = kfd_targets(where, read_text)
    cards = []
    for k, (bdf, device) in enumerate(sorted(devices.items())):
        smi = _smi(runner, where).get(bdf)
        gfx = targets.get(bdf) or (smi.gfx if smi else None)
        cards.append(
            _card(
                device,
                bdf,
                start + k,
                read_text=read_text,
                root=where,
                runner=runner,
                smi=smi,
                gfx=gfx,
                render=renders.get(bdf),
            )
        )
    return cards


def _devices(root: Path, read_text: FileReader) -> dict[str, Path]:
    """Each AMD card's PCI device folder by its address (a card has one `cardN`)."""
    found: dict[str, Path] = {}
    try:
        entries = sorted((root / DRM).iterdir())
    except OSError:
        return found
    for entry in entries:
        if not _CARD.match(entry.name):
            continue
        device = entry / "device"
        if read_text(device / "vendor").strip().lower() != AMD:
            continue
        found.setdefault(os.path.basename(os.path.realpath(device)), device)
    return found


def _render_nodes(root: Path, read_text: FileReader) -> dict[str, str]:
    found: dict[str, str] = {}
    try:
        entries = sorted((root / DRM).iterdir())
    except OSError:
        return found
    for entry in entries:
        device = entry / "device"
        if _RENDER.match(entry.name) and read_text(device / "vendor").strip().lower() == AMD:
            found[os.path.basename(os.path.realpath(device))] = f"/dev/dri/{entry.name}"
    return found


def _card(  # noqa: PLR0913 - one card's every source
    device: Path,
    bdf: str,
    index: int,
    *,
    read_text: FileReader,
    root: Path,
    runner: CommandRunner,
    smi: SmiCard | None,
    gfx: str | None,
    render: str | None,
) -> Gpu:
    def number(name: str) -> int | None:
        text = read_text(device / name).strip()
        return int(text) if text.isdigit() else None

    vram, gtt = number("mem_info_vram_total") or 0, number("mem_info_gtt_total")
    vram_used, gtt_used = number("mem_info_vram_used") or 0, number("mem_info_gtt_used")
    apu = is_apu(vram, gtt)
    link = None if apu else bridge_link(device, read_text)  # an APU has no slot
    if link is None and smi is not None and smi.lanes_max is not None:
        link = Link(lanes_max=smi.lanes_max, gen_max=smi.gen_max)
    link = link or Link()
    return Gpu(
        name=_name(smi, bdf, read_text(device / "device").strip(), vram, runner, root),
        vram_total_bytes=vram + (gtt or 0) if apu else vram,
        vram_used_bytes=vram_used + (gtt_used or 0) if apu else vram_used,
        index=index,
        bus_id=bdf,
        lanes=link.lanes,
        lanes_max=link.lanes_max,
        pcie_gen=link.gen,
        pcie_gen_max=link.gen_max,
        busy_percent=number("gpu_busy_percent"),
        vendor=Vendor.AMD,
        gfx_target=gfx,
        gtt_total_bytes=gtt,
        gtt_used_bytes=gtt_used,
        vis_vram_total_bytes=number("mem_info_vis_vram_total"),
        render_node=render,
        unique_id=read_text(device / "unique_id").strip() or None,
        carve_out=apu,
    )


def is_apu(vram_total: int, gtt_total: int | None) -> bool:
    """A carve-out of at most 1 GiB beside a GTT many times it."""
    return (
        gtt_total is not None
        and 0 < vram_total <= APU_VRAM_MAX
        and gtt_total >= APU_GTT_RATIO * vram_total
    )


def gfx_name(version: int) -> str | None:
    """KFD's `gfx_target_version` as a target: 120001 → gfx1201, 90010 → gfx90a."""
    if version <= 0:
        return None
    major, minor, step = version // 10000, (version // 100) % 100, version % 100
    return f"gfx{major}{minor:x}{step:x}"


def kfd_targets(root: Path, read_text: FileReader = read_text_or_empty) -> dict[str, str]:
    """Each GPU node's gfx target by its PCI address (CPU nodes read 0 and are skipped)."""
    found: dict[str, str] = {}
    try:
        nodes = sorted((root / KFD_NODES).iterdir())
    except OSError:
        return found
    for node in nodes:
        values: dict[str, int] = {}
        for line in read_text(node / "properties").splitlines():
            key, _, value = line.strip().partition(" ")
            if value.strip().isdigit():
                values[key] = int(value.strip())
        target = gfx_name(values.get("gfx_target_version", 0))
        location = values.get("location_id")
        if target is None or location is None:
            continue
        bdf = (
            f"{values.get('domain', 0):04x}:{(location >> 8) & 0xFF:02x}:"
            f"{(location >> 3) & 0x1F:02x}.{location & 0x7}"
        )
        found[bdf] = target
    return found


def bridge_link(device: Path, read_text: FileReader = read_text_or_empty) -> Link | None:
    """The slot's link: the topmost AMD bridge above the card (its switch's upstream port),
    else the port right above it; None when neither shows a link."""
    path = Path(os.path.realpath(device))
    chosen: Path | None = None
    for parent in path.parents:
        if not (parent / "current_link_speed").exists():
            break
        if read_text(parent / "vendor").strip().lower() == AMD:
            chosen = parent  # keep climbing: the topmost AMD bridge is the slot's
            continue
        chosen = chosen or parent
        break
    return _link(chosen, read_text) if chosen is not None else None


def _link(port: Path, read_text: FileReader) -> Link | None:
    def width(name: str) -> int | None:
        text = read_text(port / name).strip()
        return int(text) if text.isdigit() and int(text) > 0 else None

    def gen(name: str) -> int | None:
        return LINK_GENS.get(read_text(port / name).strip().split(" ")[0])

    link = Link(
        lanes=width("current_link_width"),
        lanes_max=width("max_link_width"),
        gen=gen("current_link_speed"),
        gen_max=gen("max_link_speed"),
    )
    return link if link.lanes is not None or link.lanes_max is not None else None


def _smi(runner: CommandRunner, root: Path) -> dict[str, SmiCard]:
    """`amd-smi static --json` by PCI address, asked once per process; empty without it."""
    key = (str(root), "amd-smi")
    with _lock:
        if key in _cache:
            return _cache[key]  # type: ignore[return-value]
        if _failed.get(key, 0.0) > time.monotonic():
            return {}
    try:
        code, out, _ = runner.run(AMD_SMI_STATIC, timeout=20.0)
    except OSError:
        code, out = 127, ""
    found = parse_amd_smi_static(out) if code == 0 else {}
    _keep(key, found, code == 0)
    return found


def _keep(key: tuple[str, str], found: object, answered: bool) -> None:
    """An answer is kept for the process; a failure (a timeout, a missing tool) only for
    `RETRY_SECONDS`, so a slow amd-smi once never leaves names unread until a restart."""
    with _lock:
        if answered:
            _cache[key] = found
        else:
            _failed[key] = time.monotonic() + RETRY_SECONDS


def parse_amd_smi_static(text: str) -> dict[str, SmiCard]:
    """amd-smi's JSON (`amdsmi_commands.py`'s `static`): a list of cards, or `{"gpu_data":
    [...]}` in newer releases; "N/A" is nothing."""
    try:
        data = json.loads(text)
    except ValueError:
        return {}
    if isinstance(data, dict):
        data = data.get("gpu_data", [])
    found: dict[str, SmiCard] = {}
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        asic, bus = item.get("asic") or {}, item.get("bus") or {}
        bdf = _known(bus.get("bdf"))
        if bdf is None:
            continue
        width = bus.get("max_pcie_width")
        version = (_known(bus.get("pcie_interface_version")) or "").split()  # "Gen 5"
        found[bdf.lower()] = SmiCard(
            bdf=bdf.lower(),
            name=_known(asic.get("market_name")),
            gfx=_known(asic.get("target_graphics_version")),
            lanes_max=width if isinstance(width, int) and width > 0 else None,
            gen_max=int(version[-1]) if version and version[-1].isdigit() else None,
        )
    return found


def _known(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return None if not text or text.upper() == "N/A" else text


def _name(
    smi: SmiCard | None, bdf: str, device_id: str, vram: int, runner: CommandRunner, root: Path
) -> str:
    """amd-smi's market name, else lspci's when it names one card, else the PCI id in the
    device table (several cards share an id: the VRAM tells them apart), else lspci's list."""
    if smi is not None and smi.name:
        return smi.name
    said = _lspci(runner, root, bdf)
    if said and "/" not in said:
        return said
    from makenomistakes.fit.devices import name_for_pci  # noqa: PLC0415 - fit imports hardware

    pci = int(device_id, 16) if re.fullmatch(r"0x[0-9a-fA-F]+", device_id) else None
    found = name_for_pci(pci, vram) if pci is not None else None
    return found or said or f"AMD card 1002:{device_id.removeprefix('0x') or '?'}"


def _lspci(runner: CommandRunner, root: Path, bdf: str) -> str | None:
    """lspci's device string (`Navi 48 [Radeon AI PRO R9700]`) as `AMD Radeon AI PRO R9700`."""
    key = (str(root), f"lspci {bdf}")
    with _lock:
        if key in _cache:
            return _cache[key]  # type: ignore[return-value]
        if _failed.get(key, 0.0) > time.monotonic():
            return None
    try:
        code, out, _ = runner.run(["lspci", "-D", "-mm", "-s", bdf], timeout=10.0)
    except OSError:
        code, out = 127, ""
    fields = _QUOTED.findall(out) if code == 0 else []
    found = lspci_name(fields[2]) if len(fields) > 2 else None
    _keep(key, found, code == 0)
    return found


def lspci_name(device: str) -> str | None:
    """`Navi 48 [Radeon AI PRO R9700]` → `AMD Radeon AI PRO R9700`; the marketing name in the
    brackets when there is one (`Radeon Graphics / 8050S / 8060S` stays a list: `_name`
    asks the device table first)."""
    inside = re.search(r"\[([^\]]+)\]\s*$", device)
    name = (inside.group(1) if inside else device).strip()  # a list keeps its `/`
    if not name:
        return None
    return name if name.lower().startswith("amd") else f"AMD {name}"
