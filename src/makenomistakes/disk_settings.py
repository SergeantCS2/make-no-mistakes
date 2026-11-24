"""The disk policy's settings and gauge: `<state>/disk/disk.json` (the
user's limit, the warnings told, the last evictions) and how full the measure the limit names
is. Kept apart from `disk_policy`, which decides and deletes."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from makenomistakes.disk_tree import tree_size
from makenomistakes.disk_types import SCHEMA_VERSION, DiskFile, Gauge, Keep, KeepMode, Level, gb
from makenomistakes.json_folder import JsonFolder
from makenomistakes.local_time import local
from makenomistakes.recipes import Recipe

FOLDER = "disk"  # <state>/disk/disk.json
TURN_OFF = "`mnm disk keep --models off` turns it off"
Usage = Callable[[Path], tuple[int, int]]  # a folder's filesystem: (total, free) bytes


class DiskSettingsError(RuntimeError):
    """`disk.json` cannot be read: the policy deletes nothing until it can."""


class DiskSettings:
    """The policy's own small file, written whole behind its folder's lock."""

    def __init__(self, state_dir: Path) -> None:
        self._files = JsonFolder(state_dir / FOLDER)

    @property
    def path(self) -> Path:
        return self._files.path(FOLDER)

    def read(self) -> DiskFile:
        with self._files.locked():
            return self._read()

    def update(self, change: Callable[[DiskFile], DiskFile]) -> DiskFile:
        with self._files.locked():
            changed = change(self._read())
            self._files.write(FOLDER, changed.model_dump_json(indent=2))
            return changed

    def _read(self) -> DiskFile:
        try:
            text = self._files.read(FOLDER)
        except FileNotFoundError:
            return DiskFile()
        except OSError as exc:
            raise DiskSettingsError(f"{self.path} cannot be read ({exc})") from exc
        try:
            found = DiskFile.model_validate_json(text)
        except ValidationError as exc:
            raise DiskSettingsError(
                f"{self.path} cannot be read by this version ({exc.errors()[0]['msg']}): delete "
                "it and `mnm disk keep --models …` again"
            ) from exc
        if found.schema_version > SCHEMA_VERSION:
            raise DiskSettingsError(
                f"{self.path} was written by a newer makenomistakes: run `mnm update`"
            )
        return found


def filesystem(path: Path) -> tuple[int, int]:
    """The filesystem holding `path`: total and free bytes, as `df` counts its Use% (used
    over used + available: the root's reserve is not the user's to fill)."""
    usage = shutil.disk_usage(path)
    return usage.used + usage.free, usage.free


def measure() -> Usage:
    """The filesystem reader in use now: `filesystem`, unless it was replaced."""
    return filesystem


def gauge_for(keep: Keep, models_dir: Path, usage: Usage | None = None) -> Gauge | None:
    """None when the policy is off or the folder cannot be measured. `usage` left out is
    `filesystem`, looked up at the call (tests point it at a fake disk)."""
    if not keep.on:
        return None
    try:
        if keep.mode is KeepMode.SIZE and keep.size_bytes is not None:
            held, _files = tree_size(models_dir)
            return Gauge(mode=keep.mode, used_bytes=held, base_bytes=keep.size_bytes, limit=100)
        total, free = (usage or measure())(models_dir)
    except OSError:
        return None
    return Gauge(mode=keep.mode, used_bytes=total - free, base_bytes=total, limit=keep.percent)


def warning_words(gauge: Gauge) -> str:
    """The line the doctor, `mnm add` and `mnm ls` print from the first warning on."""
    limit = limit_words(gauge)
    if gauge.level is Level.OVER:
        return (
            f"{gauge.subject}, past the disk policy's {limit}, where the least recently "
            f"used instances' files go (`mnm disk` lists them); {TURN_OFF}"
        )
    return (
        f"{gauge.subject}: at {limit} the least recently used instances' files go; `mnm disk` "
        f"lists them; {TURN_OFF}"
    )


def limit_words(gauge: Gauge) -> str:
    return f"{gauge.limit:g}%" if gauge.mode is KeepMode.PERCENT else gb(gauge.base_bytes)


def when_words(moment: datetime | None) -> str:
    return f"{local(moment):%Y-%m-%d %H:%M}" if moment is not None else "never"


def redownloaded_words(recipe: Recipe) -> str | None:
    """What a start says once an evicted instance's files are back (`recipe` as it was)."""
    if recipe.evicted is None:
        return None
    return (
        f"{recipe.name}'s files ({gb(recipe.evicted.freed_bytes)}), evicted on "
        f"{when_words(recipe.evicted.at)[:10]} by the disk policy, were downloaded again."
    )


def redownload_words(recipe: Recipe) -> str | None:
    """What a start of an evicted instance says first (and the router's loading notes)."""
    if recipe.evicted is None:
        return None
    return (
        f"re-downloading {recipe.name} ({gb(recipe.evicted.freed_bytes)}): its files were "
        f"evicted on {when_words(recipe.evicted.at)[:10]} by the disk policy"
    )


def warning_now(state_dir: Path, models_dir: Path) -> str | None:
    """The policy's warning line as things stand (`mnm add`, `mnm ls`), or None: under the
    first warning, off, or unreadable (`mnm disk` says why). Measures only, never evicts."""
    try:
        keep = DiskSettings(state_dir).read().keep
    except DiskSettingsError:
        return None
    gauge = gauge_for(keep, models_dir)
    if gauge is None or gauge.level is Level.OK:
        return None
    return warning_words(gauge)
