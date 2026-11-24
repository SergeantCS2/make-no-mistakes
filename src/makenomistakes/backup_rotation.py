"""The backups folder (`MNM_BACKUPS_DIR`, /data/backups by default): what `mnm backup
--auto` writes there, and the rotation that keeps it small.

An automatic backup is `mnm-<local date>-<local time>.tar.gz` (`mnm-2026-10-26-033000.tar.gz`,
the box's clock: `local_time`). The rotation keeps the newest backup of each of the last
seven days that have one (*daily*) and the first backup of each of the last four ISO weeks
that have one (*weekly*), and deletes the other automatic ones. A file of any other name (a
`mnm backup --to /data/backups/…` by hand) is listed and never rotated: only the names this
module writes are its to delete, and never a link or a folder.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path

from makenomistakes.backup import BackupError
from makenomistakes.folder_hint import folder_hint

DAILY = 7
WEEKLY = 4
AUTO_NAME = re.compile(r"^mnm-(\d{4}-\d\d-\d\d-\d{6})\.tar\.gz$")
LISTED = "*.tar.gz"


class Kept(StrEnum):
    DAILY = "daily"
    WEEKLY = "weekly"
    BOTH = "daily + weekly"
    BY_HAND = "by hand (never rotated)"


@dataclass(frozen=True)
class BackupFile:
    path: Path
    size: int
    taken: datetime  # a naive local time: the name's for an automatic one, else the file's mtime
    automatic: bool

    @property
    def name(self) -> str:
        return self.path.name


def auto_name(now: datetime) -> str:
    """`now` on the box's clock (a naive local time), as an automatic backup's file name."""
    return f"mnm-{now:%Y-%m-%d-%H%M%S}.tar.gz"


def taken_at(name: str) -> datetime | None:
    """When an automatic backup was taken, from its name; None for any other name."""
    found = AUTO_NAME.match(name)
    if found is None:
        return None
    try:
        return datetime.strptime(found.group(1), "%Y-%m-%d-%H%M%S")
    except ValueError:  # 2026-13-40: not a name this module wrote
        return None


def listing(folder: Path) -> list[BackupFile]:
    """Every backup archive in the folder (regular files only), newest first."""
    if not folder.is_dir():
        return []
    found = []
    for path in folder.glob(LISTED):
        if path.is_symlink() or not path.is_file():
            continue
        info = path.stat()
        taken = taken_at(path.name)
        when = taken or datetime.fromtimestamp(info.st_mtime)
        found.append(BackupFile(path, info.st_size, when, taken is not None))
    return sorted(found, key=lambda b: (b.taken, b.name), reverse=True)


def keep(backups: list[BackupFile], daily: int = DAILY, weekly: int = WEEKLY) -> dict[Path, Kept]:
    """What the rotation keeps of the automatic backups, and why; every other file is kept
    `BY_HAND`. An automatic backup missing here is one the rotation deletes."""
    auto = sorted((b for b in backups if b.automatic), key=lambda b: (b.taken, b.name))
    newest_of_day: dict[date, BackupFile] = {}
    first_of_week: dict[tuple[int, int], BackupFile] = {}
    for backup in auto:  # oldest first: a day's last one wins, a week's first one stays
        newest_of_day[backup.taken.date()] = backup
        iso = backup.taken.isocalendar()
        first_of_week.setdefault((iso.year, iso.week), backup)
    days = sorted(newest_of_day, reverse=True)[:daily]
    weeks = sorted(first_of_week, reverse=True)[:weekly]
    on_day = {newest_of_day[d].path for d in days}
    on_week = {first_of_week[w].path for w in weeks}
    kept: dict[Path, Kept] = {b.path: Kept.BY_HAND for b in backups if not b.automatic}
    for path in on_day | on_week:
        if path in on_day and path in on_week:
            kept[path] = Kept.BOTH
        else:
            kept[path] = Kept.DAILY if path in on_day else Kept.WEEKLY
    return kept


def rotate(folder: Path) -> list[BackupFile]:
    """Delete the automatic backups `keep` does not keep; the deleted ones, oldest first."""
    found = listing(folder)
    kept = keep(found)
    gone = []
    for backup in sorted(found, key=lambda b: b.taken):
        if backup.path in kept:
            continue
        try:
            backup.path.unlink()
        except FileNotFoundError:  # another rotation took it first
            continue
        except OSError as exc:
            raise BackupError(f"Could not delete the old backup {backup.path}: {exc}") from exc
        gone.append(backup)
    return gone


def ready(folder: Path) -> None:
    """The backups folder exists and is writable: made when its parent is writable, else a
    BackupError with the two commands that make it (`folder_hint`'s words, as setup's)."""
    if not folder.exists():
        with contextlib.suppress(OSError):  # the hint below says how to make it
            folder.mkdir(parents=True)
    hint = folder_hint(folder, "MNM_BACKUPS_DIR")
    if hint is not None:
        raise BackupError(f"The backups folder: {hint}")


def age(taken: datetime, now: datetime) -> str:
    """How long ago, in the largest whole unit: `40 min`, `5 h`, `3 d`."""
    seconds = max(0, int((now - taken).total_seconds()))
    if seconds < 3600:
        return f"{seconds // 60} min"
    if seconds < 86400:
        return f"{seconds // 3600} h"
    return f"{seconds // 86400} d"


def size_words(size: int) -> str:
    if size < 1024**2:
        return f"{size / 1024:.0f} KiB"
    return f"{size / 1024**2:.1f} MiB"
