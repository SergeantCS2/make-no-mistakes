"""What `mnm disk` reports and what the disk policy keeps: typed, so the
CLI's table, `--json`, the web's System → Disk and the doctor's line read the same facts.

Sizes are bytes; the words say them in decimal GB, as the Hub and `df -H` do."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1  # `<state>/disk/disk.json`
KEEP_EVICTIONS = 50  # the evictions `mnm disk` lists, newest last
DEFAULT_PERCENT = 85.0  # on by default at 85% of the models folder's disk
FIRST_WARNING = 15.0  # points under the limit: the doctor WARNs, a notice once
SECOND_WARNING = 5.0  # points under the limit: a second notice; eviction stops under it
_UNITS = {"k": 10**3, "m": 10**6, "g": 10**9, "t": 10**12}
_SIZE = re.compile(r"^(\d+(?:\.\d+)?)(?:(bytes?|b)|([kmgt])(i?)b?)$")


def gb(size: int | float) -> str:
    """Bytes in words: `5.0 GB`, `512 MB` under one GB, `12 kB`, `400 bytes`."""
    if size >= 10**9:
        return f"{size / 10**9:.1f} GB"
    if size >= 10**6:
        return f"{size / 10**6:.0f} MB"
    return f"{size / 10**3:.0f} kB" if size >= 10**3 else f"{size:.0f} bytes"


class KeepMode(StrEnum):
    PERCENT = "percent"  # of the models folder's filesystem, as `df` counts it
    SIZE = "size"  # what the models folder itself holds
    OFF = "off"


class Keep(BaseModel):
    """`mnm disk keep --models 85%|N G|off`."""

    mode: KeepMode = KeepMode.PERCENT
    percent: float = Field(default=DEFAULT_PERCENT, gt=0, le=100)
    size_bytes: int | None = Field(default=None, gt=0)

    @property
    def on(self) -> bool:
        return self.mode is not KeepMode.OFF

    @property
    def words(self) -> str:
        if self.mode is KeepMode.OFF:
            return "off"
        if self.mode is KeepMode.SIZE and self.size_bytes is not None:
            return f"{gb(self.size_bytes)} in the models folder"
        return f"{self.percent:g}% of the models folder's disk"


def parse_keep(text: str) -> Keep:
    """`85%` (or `85`), a size (`400G`, `1.5T`, `500M`, `1GiB`, `700 bytes`; G is 10^9 bytes,
    GiB 2^30), or `off`; ValueError in words for anything else."""
    word = text.strip().lower()
    if word == "off":
        return Keep(mode=KeepMode.OFF)
    if word.endswith("%") or re.fullmatch(r"\d+(?:\.\d+)?", word):
        number = float(word.rstrip("%").strip())
        if not 0 < number <= 100:
            raise ValueError(f"{text!r}: a share of the disk is more than 0% and at most 100%")
        return Keep(mode=KeepMode.PERCENT, percent=number)
    found = _SIZE.match(word.replace(" ", ""))
    if found is None:
        raise ValueError(
            f"{text!r} is not a limit: give a share of the disk (`85%`), a size (`400G`, "
            "`1.5T`) or `off`"
        )
    amount, plain, unit, binary = found.groups()
    binary_scale = 1024 ** ("kmgt".index(unit) + 1) if unit and binary else None
    scale = 1 if plain else binary_scale or _UNITS[unit]
    size = int(float(amount) * scale)
    if size <= 0:
        raise ValueError(f"{text!r}: a size limit is more than nothing")
    return Keep(mode=KeepMode.SIZE, size_bytes=size)


class EvictionRecord(BaseModel):
    """One instance whose files the policy deleted."""

    name: str
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    freed_bytes: int = Field(ge=0)
    files: int = Field(ge=0)
    last_used: datetime | None = None
    usage_before: float  # the gauge's percent when it went
    limit: float
    mode: KeepMode = KeepMode.PERCENT
    limit_words: str = ""  # `85%`, `500 MB`

    @property
    def where(self) -> str:
        """`86% of the disk, limit 85%` / `149% of the 500 MB limit`."""
        if self.mode is KeepMode.SIZE:
            return f"{self.usage_before:g}% of the {self.limit_words} limit"
        return f"{self.usage_before:g}% of the disk, limit {self.limit:g}%"


class DiskFile(BaseModel):
    """`<state>/disk/disk.json`: the user's limit, the warnings already sent (one notice per
    crossing, again after a drop below) and the last `KEEP_EVICTIONS` evictions."""

    schema_version: int = SCHEMA_VERSION
    keep: Keep = Field(default_factory=Keep)
    notified: list[float] = Field(default_factory=list)  # the levels told (70, 80, 85)
    evictions: list[EvictionRecord] = Field(default_factory=list)


class Level(StrEnum):
    OK = "ok"
    WARN = "warn"  # past the first warning
    OVER = "over"  # at or past the limit


class Gauge(BaseModel):
    """How full the policy's measure is: the disk (`used` of `base` = the filesystem) or the
    models folder against a size limit (`base` = the limit, `limit` 100)."""

    mode: KeepMode
    used_bytes: int
    base_bytes: int
    limit: float  # percent of `base_bytes` at which files go

    @property
    def percent(self) -> float:
        return 100.0 * self.used_bytes / self.base_bytes if self.base_bytes else 0.0

    @property
    def warnings(self) -> tuple[float, float]:
        return (max(self.limit - FIRST_WARNING, 0.0), max(self.limit - SECOND_WARNING, 0.0))

    @property
    def target(self) -> float:
        """Eviction goes on until the gauge is under this."""
        return self.warnings[1]

    def bytes_over(self, percent: float) -> int:
        """What must go for the gauge to fall under `percent`."""
        return max(self.used_bytes - int(self.base_bytes * percent / 100.0) + 1, 0)

    def after(self, freed: int) -> Gauge:
        return self.model_copy(update={"used_bytes": max(self.used_bytes - freed, 0)})

    @property
    def level(self) -> Level:
        if self.percent >= self.limit:
            return Level.OVER
        return Level.WARN if self.percent >= self.warnings[0] else Level.OK

    @property
    def subject(self) -> str:
        """`the models folder's disk is at 71%` / `the models folder holds 0.9 GB of its
        1.0 GB limit (88%)`."""
        if self.mode is KeepMode.SIZE:
            return (
                f"the models folder holds {gb(self.used_bytes)} of its {gb(self.base_bytes)} "
                f"limit ({self.percent:.0f}%)"
            )
        return f"the models folder's disk is at {self.percent:.0f}%"


class SharedFile(BaseModel):
    path: str  # relative to the models folder
    size_bytes: int
    users: list[str]  # the other instances and workflows that list it


class InstanceDisk(BaseModel):
    name: str
    own_bytes: int  # files only this instance lists
    shared: list[SharedFile] = Field(default_factory=list)  # counted once, under each user
    files: int
    state: str  # on disk, partly on disk, not downloaded, evicted <date>
    last_used: datetime | None
    last_used_from: str  # what says so: running now, a start, a request, saved
    evicted: datetime | None = None
    prompt_bytes: int = 0  # Its saved prompts (`cache=on`), the policy's first to go

    @property
    def shared_bytes(self) -> int:
        return sum(f.size_bytes for f in self.shared)


class FolderSize(BaseModel):
    label: str
    path: str
    exists: bool
    size_bytes: int = 0
    files: int = 0


class UnlistedFile(BaseModel):
    path: str
    size_bytes: int
    why: str  # never finished, or no recipe, workflow or LoRA lists it


class PolicyState(BaseModel):
    keep: Keep
    gauge: Gauge | None  # None: off, or the disk could not be read
    level: Level
    next: list[str] = Field(default_factory=list)  # the instances that would go, in order
    held: list[str] = Field(default_factory=list)  # why the others would stay, in words
    evictions: list[EvictionRecord] = Field(default_factory=list)


class DiskReport(BaseModel):
    """`mnm disk`: one report, the table and `--json` both."""

    models_dir: str
    disk_total_bytes: int  # used + available, as `df` counts its Use%
    disk_free_bytes: int
    held_bytes: int  # what the models folder holds, links not followed
    instances: list[InstanceDisk]
    sources: list[FolderSize]  # hf/, civitai/, … under the models folder
    workflows: list[FolderSize]  # ComfyUI workflows' files, by workflow (counted in sources)
    unlisted: list[UnlistedFile]
    other: list[FolderSize]  # entries under the models folder that are not mnm's
    folders: list[FolderSize]  # outputs, state, ComfyUI's, apps, backups
    policy: PolicyState
    notes: list[str] = Field(default_factory=list)  # what could not be read

    @property
    def unlisted_bytes(self) -> int:
        return sum(f.size_bytes for f in self.unlisted)
