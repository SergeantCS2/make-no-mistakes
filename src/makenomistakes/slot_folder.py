"""An instance's saved-slots folder, `<models>/prompt-cache/<name>/` (`SlotFolder`): the
slot files llama-server writes there, a sidecar beside each saying whose settings and build it
holds, and the stamp of the container start that may save into it (`StartStamp`).

The engine writes the `.bin` files; mnm writes everything else and decides what stays: any file
of other settings or another build is deleted unread (a slot restores into other weights
without a word: llama.cpp checks shapes and token ids only), a `.bin` without a sidecar is
deleted, and the oldest go first when the folder passes its cap or the disk policy needs the
room. The engine side (save, restore) is `slot_files`."""

from __future__ import annotations

import contextlib
import shutil
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from makenomistakes.prompt_cache import GIB
from makenomistakes.recipes.llamacpp_cache import CACHE_FOLDER

SIDECAR = ".json"
STAMP = "started.stamp"  # not `*.json`: never read as a sidecar
HELD = ".held"  # `mnm try --agent` sets the user's files aside here while it measures


class SavedSlot(BaseModel):
    """The sidecar beside a slot file: whose it is and what it holds."""

    schema_version: int = 1
    file: str
    key: str  # `measurements.settings_key` of the container that saved it
    build: str  # the engine's image (`ContainerStatus.build`)
    tokens: int
    bytes: int
    digest: str
    saved_at: datetime


class StartStamp(BaseModel):
    """The container start that may save into the folder, written by that start once its
    engine answered (`slot_files.after_start`): the settings it was started with, never the
    recipe as saved since (`mnm set`, `mnm replan`, `mnm machine adopt` apply at the next
    start; a slot saved under the new key would be put back into the old weights' files)."""

    schema_version: int = 1
    key: str
    build: str
    started_at: str  # Docker's StartedAt: the start this stamp is of


class SlotFolder:
    """One instance's folder; every method reads it afresh (the engine writes into it)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def of(cls, models_dir: Path, name: str) -> SlotFolder:
        return cls(models_dir / CACHE_FOLDER / name)

    def saved(self) -> list[SavedSlot]:
        """The sidecars, newest first; one that cannot be read is skipped."""
        found = []
        for side in self.path.glob(f"*{SIDECAR}") if self.path.is_dir() else ():
            with contextlib.suppress(OSError, ValueError, ValidationError):
                found.append(SavedSlot.model_validate_json(side.read_text(encoding="utf-8")))
        return sorted(found, key=lambda s: s.saved_at, reverse=True)

    def held_bytes(self) -> int:
        """What the folder holds on disk (`mnm disk`), the files set aside included."""
        total = 0
        for entry in self.path.rglob("*") if self.path.is_dir() else ():
            with contextlib.suppress(OSError):
                if entry.is_file() and not entry.is_symlink():
                    total += entry.stat().st_size
        return total

    def size(self, file: str) -> int:
        """A slot file and its sidecar, on disk."""
        return _size(self.path / file) + _size(self.path / (file + SIDECAR))

    def drop(self, file: str) -> None:
        for name in (file, file + SIDECAR):
            with contextlib.suppress(OSError):
                (self.path / name).unlink()

    def write(self, row: SavedSlot) -> None:
        (self.path / (row.file + SIDECAR)).write_text(row.model_dump_json(), encoding="utf-8")

    def stale(self, key: str, build: str) -> None:
        """Every file of other settings or another build goes, and any file without one."""
        rows = self.saved()
        for row in rows:
            if row.key != key or row.build != build:
                self.drop(row.file)
        known = {row.file for row in rows}
        for entry in self.path.glob("*.bin") if self.path.is_dir() else ():
            if entry.name not in known:
                self.drop(entry.name)

    def trimmed(self, cap: int, keep: str) -> list[str]:
        """The oldest files but `keep` go until the folder is under `cap`; their names."""
        gone = []
        for row in reversed(self.saved()):  # oldest first
            if self.held_bytes() <= cap:
                break
            if row.file != keep:
                self.drop(row.file)
                gone.append(row.file)
        return gone

    def describe(self) -> str:
        """One line: "2 saved prompts, 1.4 GiB (newest 21,344 tokens)"."""
        rows = self.saved()
        if not rows:
            return "no saved prompts"
        count = f"{len(rows)} saved prompt{'s' if len(rows) != 1 else ''}"
        return f"{count}, {gib(self.held_bytes())} GiB (newest {rows[0].tokens:,} tokens)"

    # --- the start that may save --------------------------------------------------------

    def stamp(self, stamp: StartStamp) -> None:
        (self.path / STAMP).write_text(stamp.model_dump_json(), encoding="utf-8")

    def stamped(self, started_at: str | None) -> StartStamp | None:
        """The stamp of the container start `started_at`; None for any other start (one made
        before this mnm, by hand, or with the cache off: it has no folder mounted)."""
        try:
            stamp = StartStamp.model_validate_json((self.path / STAMP).read_text("utf-8"))
        except (OSError, ValueError, ValidationError):
            return None
        return stamp if started_at is not None and stamp.started_at == started_at else None

    # --- `mnm try --agent` --------------------------------------------------------------

    @contextlib.contextmanager
    def set_aside(self) -> Iterator[None]:
        """The saved prompts out of the way while a check measures with its own (it would
        otherwise put the user's back at its load and push them over the cap with its
        synthetic one): the check's files go after it, the user's come back."""
        held = self.path / HELD
        if not self.path.is_dir():
            yield
            return
        held.mkdir(exist_ok=True)
        for row in self.saved():
            for name in (row.file, row.file + SIDECAR):
                with contextlib.suppress(OSError):
                    (self.path / name).rename(held / name)
        try:
            yield
        finally:
            for row in self.saved():  # the check's own
                self.drop(row.file)
            for entry in held.iterdir():
                with contextlib.suppress(OSError):
                    entry.rename(self.path / entry.name)
            with contextlib.suppress(OSError):
                shutil.rmtree(held)


def gib(size: int) -> str:
    """Bytes as GiB for a line: "1.25", "0.23", "12"."""
    return f"{size / GIB:.2f}".rstrip("0").rstrip(".")


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def drop_oldest(models_dir: Path, need: int) -> tuple[int, int]:
    """The disk policy's first step: every instance's saved prompts, oldest first,
    until `need` bytes are free; (files dropped, bytes freed). A cache only: the next stop
    saves again. A file the engine wrote without a sidecar (a stop cut short) counts too,
    oldest by its time."""
    rows: list[tuple[float, SlotFolder, str]] = []
    for path in sorted((models_dir / CACHE_FOLDER).glob("*")):
        folder = SlotFolder(path)
        rows += [(row.saved_at.timestamp(), folder, row.file) for row in folder.saved()]
        known = {row.file for row in folder.saved()}
        for entry in path.glob("*.bin") if path.is_dir() else ():
            if entry.name not in known:
                rows.append((_mtime(entry), folder, entry.name))
    dropped = freed = 0
    for _at, folder, file in sorted(rows, key=lambda row: row[0]):
        if freed >= need:
            break
        size = folder.size(file)
        folder.drop(file)
        dropped, freed = dropped + 1, freed + size
    return dropped, freed


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
