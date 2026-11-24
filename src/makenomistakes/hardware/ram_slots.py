"""The RAM's DIMMs and channels, for the doctor: a mixture of experts keeps its experts
in RAM, and two channels read about twice what one does; on AM5 four DIMMs run slower than
two (two a channel). Read without sudo where Linux allows it, in this order:

- `dmidecode -t memory`: every slot with its size, type, speed and bank (most boxes keep the
  SMBIOS tables root-only, so it usually answers "Permission denied");
- `/sys/devices/system/edac/mc/mc*/dimm*` (or `rank*`): the EDAC driver's view, world-
  readable, loaded on boards with ECC (`size` in MiB, `dimm_location` "csrow 0 channel 1");
- `lshw -short -C memory`: its DIMM rows, when it can read them unprivileged.

Nothing readable is "unknown without sudo", never a guess. Under WSL there are no slots to
read (the VM has no SMBIOS memory devices): the doctor says so instead.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

from makenomistakes.hardware.probe import CommandRunner

DMIDECODE = ["dmidecode", "-t", "memory"]
LSHW = ["lshw", "-short", "-C", "memory"]
GIB = 2**30  # a "32 GB" DIMM holds 32 GiB: DIMMs are sold in binary gigabytes
_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*(GiB|GB|MiB|MB|TB)", re.IGNORECASE)
_SPEED = re.compile(r"(\d+)\s*(MT/s|MHz)", re.IGNORECASE)
_CHANNEL = re.compile(r"CHANNEL\s*([A-Z0-9]+)", re.IGNORECASE)
_SLOT = re.compile(r"(?:DIMM|SLOT)?[_\s-]*([A-H])[_\s-]?\d\b", re.IGNORECASE)
_KIND = re.compile(r"\b(LP)?DDR\d\b")
_UNITS = {"tb": 2**40, "gib": 2**30, "gb": 2**30, "mib": 2**20, "mb": 2**20}  # DIMMs are binary


class Dimm(BaseModel):
    size_bytes: int
    kind: str | None = None  # DDR5
    speed_mts: int | None = None  # what it runs at (configured), else its rated speed
    channel: str | None = None


class RamSlots(BaseModel):
    dimms: list[Dimm]
    empty: int = 0  # slots dmidecode lists without a module
    source: str

    @property
    def channels(self) -> int | None:
        found = {d.channel for d in self.dimms if d.channel}
        return len(found) if found and all(d.channel for d in self.dimms) else None

    def words(self) -> str:
        """`4 DIMMs of 32 GB DDR5 at 5200 MT/s on 2 channels (2 a channel) (dmidecode)`."""
        count = len(self.dimms)
        sizes = {d.size_bytes for d in self.dimms}
        each = (
            f" of {next(iter(sizes)) / GIB:.0f} GB"
            if len(sizes) == 1
            else f" ({sum(d.size_bytes for d in self.dimms) / GIB:.0f} GB)"
        )
        kinds = {d.kind for d in self.dimms if d.kind}
        kind = f" {next(iter(kinds))}" if len(kinds) == 1 else ""
        speeds = {d.speed_mts for d in self.dimms if d.speed_mts}
        speed = f" at {min(speeds)} MT/s" if speeds else ""
        text = f"{count} DIMM{'s' if count != 1 else ''}{each}{kind}{speed}"
        channels = self.channels
        if channels:
            text += f" on {channels} channel{'s' if channels != 1 else ''}"
            if count > channels:
                text += f" ({count // channels} a channel)"
        if self.empty:
            text += f", {self.empty} slot{'s' if self.empty != 1 else ''} free"
        return f"{text} ({self.source})"


def parse_dmidecode(text: str) -> RamSlots | None:
    dimms: list[Dimm] = []
    empty = 0
    for block in text.split("\n\n"):
        if "Memory Device" not in block:
            continue
        fields = {
            k.strip(): v.strip()
            for k, sep, v in (line.partition(":") for line in block.splitlines())
            if sep
        }
        size = fields.get("Size", "")
        if not size or "No Module" in size or size.startswith("0"):
            empty += 1
            continue
        bytes_ = _bytes(size)
        if bytes_ is None:
            continue
        speed = _SPEED.search(fields.get("Configured Memory Speed", "") or "") or _SPEED.search(
            fields.get("Speed", "")
        )
        kind = fields.get("Type")
        dimms.append(
            Dimm(
                size_bytes=bytes_,
                kind=kind if kind and kind not in ("Unknown", "Other") else None,
                speed_mts=int(speed.group(1)) if speed else None,
                channel=_channel(fields.get("Bank Locator", ""), fields.get("Locator", "")),
            )
        )
    return RamSlots(dimms=dimms, empty=empty, source="dmidecode") if dimms else None


def parse_lshw(text: str) -> RamSlots | None:
    dimms = []
    for line in text.splitlines():
        if " memory " not in f" {line} " or "DIMM" not in line:
            continue
        description = line.split("memory", 1)[1]
        size = _SIZE.search(description)
        if size is None:
            continue
        speed = _SPEED.search(description)
        kind = _KIND.search(description)
        dimms.append(
            Dimm(
                size_bytes=_bytes(size.group(0)) or 0,
                kind=kind.group(0) if kind else None,
                speed_mts=int(speed.group(1)) if speed else None,
            )
        )
    return RamSlots(dimms=dimms, source="lshw") if dimms else None


def read_edac(folder: Path) -> RamSlots | None:
    """The EDAC driver's DIMMs (`size` in MiB); a channel per controller and channel."""
    dimms = []
    try:
        entries = sorted(folder.glob("mc[0-9]*/dimm[0-9]*")) or sorted(
            folder.glob("mc[0-9]*/rank[0-9]*")
        )
        for entry in entries:
            size = _read(entry / "size")
            if not size.isdigit() or int(size) == 0:
                continue
            where = re.search(r"channel\s*(\d+)", _read(entry / "dimm_location"))
            kind = _KIND.search(_read(entry / "dimm_mem_type"))
            dimms.append(
                Dimm(
                    size_bytes=int(size) * 2**20,
                    kind=kind.group(0) if kind else None,
                    channel=f"{entry.parent.name}/{where.group(1)}" if where else None,
                )
            )
    except OSError:
        return None
    return RamSlots(dimms=dimms, source="EDAC") if dimms else None


def read_slots(runner: CommandRunner, root: Path = Path("/")) -> RamSlots | None:
    """dmidecode, else EDAC, else lshw; None when none of them says (no sudo is asked)."""
    code, out, _err = runner.run(DMIDECODE)
    found = parse_dmidecode(out) if code == 0 else None
    if found is None:
        found = read_edac(root / "sys/devices/system/edac/mc")
    if found is None:
        code, out, _err = runner.run(LSHW)
        found = parse_lshw(out) if code == 0 else None
    return found


def _bytes(text: str) -> int | None:
    found = _SIZE.search(text)
    if found is None:
        return None
    return int(float(found.group(1)) * _UNITS[found.group(2).lower()])


def _channel(bank: str, locator: str) -> str | None:
    for text in (bank, locator):
        found = _CHANNEL.search(text)
        if found:
            return found.group(1).upper()
    found = _SLOT.search(locator)
    return found.group(1).upper() if found else None


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return ""
