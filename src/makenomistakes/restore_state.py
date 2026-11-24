"""`mnm restore`: a backup's files back into this machine's state folder.

The whole archive is checked before anything is written (`backup.check`: a newer schema is
refused), and so is every folder a file would land in (a link is refused: a restore writes
only into real folders). A recipe that would replace a running instance's differing one is
refused unless `force` (its engine keeps the settings it started with). Each file is written
under the lock its store takes, through a `.restoring` file moved into place (removed when
the write fails). Readings are merged, never replaced: the ones taken here since stay, the
archive's are added, every line in time order (readers take the file's order as time's);
every box's readings are kept, each naming its box, and the boxes in `machine.json` merged
(`restore_machines`: another box's readings wait for it to return, never used here).
This machine's own engine ledger is kept.
Nothing is planned again: each instance downloads its files at its first `mnm start`."""

from __future__ import annotations

import contextlib
import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from makenomistakes.backup import Archive, BackupError, check
from makenomistakes.comfy.store import WorkflowMeta, WorkflowStore, WorkflowStoreError
from makenomistakes.engines import ledger_locked
from makenomistakes.locking import shared_lock
from makenomistakes.machine_facts import MachineFactsError, MachineFile, load_file
from makenomistakes.readings_filter import here_in
from makenomistakes.restore_machines import (
    MACHINE_FILE,
    MachinesError,
    Stamper,
    box_words,
    foreign,
    merge_into,
    theirs,
)
from makenomistakes.studio.style_plan import PROBES, merged_probes

READINGS = "measurements/"
_EARLIEST = datetime.min.replace(tzinfo=UTC)
# The folder whose lock each kind of file is written under: the one its store takes.
_LOCKS = (
    ("instances/", "instances"),
    ("workflows/", "workflows"),
    ("desk/", "desk"),
    ("measurements/", "measurements"),
    ("tuning/", "tuning"),  # `TuneStore`'s lock
    ("comfy/workflows/", "comfy/workflows"),
    ("comfy/packs.json", "comfy/packs"),
    ("comfy/models.json", "comfy"),
    ("studio/voices/", "studio/voices"),  # the voice library's `JsonFolder`
    ("studio/styles/", "studio/styles"),  # the style library's `JsonFolder`
)


@dataclass
class Restored:
    written: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)  # files that were there and differed
    kept: list[str] = field(default_factory=list)  # this machine's own, left as they were
    merged: list[str] = field(default_factory=list)  # readings added to the ones here
    # The archive's readings of other boxes, kept for when they return, and the boxes
    other_machines: int = 0
    machines: set[str] = field(default_factory=set)
    rehashed: list[str] = field(default_factory=list)  # workflows scrubbed on backup

    def count(self, folder: str) -> int:
        return sum(1 for rel in self.written if rel.startswith(folder))


class _Line(BaseModel):
    """What a restore reads of a reading: when, and what makes it the same reading whatever
    fields a later mnm wrote beside them (its box among them)."""

    at: datetime
    kind: str = ""
    settings: str = ""
    build: str | None = None

    @property
    def key(self) -> tuple[str, str, str, str | None]:
        return (self.at.isoformat(), self.kind, self.settings, self.build)


def _parsed(line: str) -> _Line | None:
    try:
        return _Line.model_validate_json(line)
    except ValidationError:
        return None


def merged_readings(here: bytes, given: bytes) -> bytes:
    """The readings here and the archive's not among them, sorted by when each was taken (a
    line that does not parse keeps its place at the start)."""
    lines = here.decode(errors="replace").splitlines()
    seen = {_identity(line) for line in lines}
    added = [
        line for line in given.decode(errors="replace").splitlines() if _identity(line) not in seen
    ]
    if not added:
        return here

    def when(line: str) -> datetime:
        found = _parsed(line)
        return found.at if found is not None else _EARLIEST

    return ("\n".join(sorted([*lines, *added], key=when)) + "\n").encode()


def _identity(line: str) -> object:
    """A reading's identity: when, what kind, which settings and build. The same reading
    written by a newer mnm (a `machine` field more) is the same; a line that does not parse
    is itself."""
    found = _parsed(line)
    return found.key if found is not None else line


def _lock_for(state_dir: Path, rel: str) -> contextlib.AbstractContextManager[None]:
    if rel == "engines.json":
        return ledger_locked(state_dir)
    folder = next((f for prefix, f in _LOCKS if rel.startswith(prefix)), None)
    if folder is None:
        return contextlib.nullcontext()
    (state_dir / folder).mkdir(parents=True, exist_ok=True)
    return shared_lock(state_dir / folder)()


def _refuse_links(state_dir: Path, rels: list[str]) -> None:
    """Every folder between the state folder and a file is a real folder (or not yet there)."""
    for rel in rels:
        folder = state_dir
        for part in rel.split("/")[:-1]:
            folder = folder / part
            if folder.is_symlink():
                raise BackupError(
                    f"{folder} is a link; restore writes only into real folders. Nothing was "
                    "restored."
                )


def restore(
    archive: Archive,
    state_dir: Path,
    running: Callable[[str], bool],
    force: bool = False,
    card: str | None = None,
) -> Restored:
    """Write the archive's files into `state_dir`. `running(name)` says whether an instance
    runs now (it raises BackupError when Docker cannot say); `card` is this machine's GPU."""
    check(archive)
    _refuse_links(state_dir, list(archive.files))
    (state_dir / "instances").mkdir(parents=True, exist_ok=True)
    # The recipe store's own lock, from the running check to the last write: a start in
    # another process (which saves its recipe under it) cannot land in between.
    with shared_lock(state_dir / "instances")():
        done = _restore_all(archive, state_dir, running, force, card)
    _refresh_hashes(state_dir, archive, done)
    return done


def _restore_all(
    archive: Archive,
    state_dir: Path,
    running: Callable[[str], bool],
    force: bool,
    card: str | None,
) -> Restored:
    busy = [
        rel.removeprefix("instances/").removesuffix(".json")
        for rel in archive.files
        if rel.startswith("instances/")
        and (state_dir / rel).exists()
        and (state_dir / rel).read_bytes() != archive.files[rel]
    ]
    busy = [name for name in busy if running(name)]
    if busy and not force:
        raise BackupError(
            f"Would replace the recipe of {', '.join(busy)}, which "
            f"{'runs' if len(busy) == 1 else 'run'} now: `mnm stop {' '.join(busy)}` first, or "
            "`--force` (a running engine keeps the settings it started with until its next "
            "start). Nothing was restored."
        )
    done = Restored()
    try:
        boxes = theirs(archive.files.get(MACHINE_FILE))
    except MachinesError as exc:
        raise BackupError(f"{exc}. Nothing was restored.") from exc
    if boxes is not None:  # before the readings: their count is against the merged boxes
        _merge_boxes(state_dir, boxes, done)
    try:
        here_file = load_file(state_dir)
    except MachineFactsError:  # unreadable here: every box's readings are still kept
        here_file = None
    here = here_in(here_file)
    stamper = Stamper(
        machine=boxes.unstamped_on if boxes is not None else None,
        card=card if boxes is None else None,
        their_card=archive.manifest.gpu,
    )
    for rel, data in sorted(archive.files.items()):
        given = data
        if rel == MACHINE_FILE:
            continue
        if rel.startswith(READINGS):
            given = stamper.lines(data)
            away = foreign(given, here)
            done.other_machines += len(away)
            done.machines.update(box_words(here_file, box) for box in away)
            if not given:
                continue
        with _lock_for(state_dir, rel):
            _restore_one(state_dir, rel, given, done)
    return done


def _merge_boxes(state_dir: Path, boxes: MachineFile, done: Restored) -> None:
    path = state_dir / MACHINE_FILE
    before = path.read_bytes() if path.exists() else None
    try:
        merge_into(state_dir, boxes)
    except (MachineFactsError, OSError) as exc:
        raise BackupError(f"Restoring stopped at {MACHINE_FILE}: {exc}") from exc
    if path.read_bytes() != before:
        done.written.append(MACHINE_FILE)


def _restore_one(state_dir: Path, rel: str, given: bytes, done: Restored) -> None:
    target = state_dir / rel
    if rel == "engines.json" and target.exists():
        done.kept.append(rel)  # this machine's builds, not the old one's
        return
    data = given
    if target.exists():
        here = target.read_bytes()
        if (merge := _merger(rel)) is not None:  # what this box measured since is kept
            data = merge(here, given)
            if data == here:
                return
            done.merged.append(rel)
        elif here == given:
            return
        else:
            done.replaced.append(rel)
    partial = target.with_name(target.name + ".restoring")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(data)
        os.replace(partial, target)
    except OSError as exc:
        with contextlib.suppress(OSError):
            partial.unlink(missing_ok=True)
        raise BackupError(
            f"Restoring stopped at {rel} ({exc}); {len(done.written)} files were written "
            "before it. Fix that, then restore again (unchanged files are skipped)."
        ) from exc
    done.written.append(rel)


def _merger(rel: str) -> Callable[[bytes, bytes], bytes] | None:
    """How a state file this box measured into is merged with the backup's, never replaced:
    the readings, the style trainer's probes."""
    if rel.startswith(READINGS):
        return merged_readings
    return merged_probes if rel == PROBES else None


def _refresh_hashes(state_dir: Path, archive: Archive, done: Restored) -> None:
    """A workflow file scrubbed on its way into the backup (`report.redact`) no longer hashes
    to its meta's sha256: the meta takes the file's hash, and the user is told."""
    names = sorted(
        {rel.split("/")[2] for rel in done.written if rel.startswith("comfy/workflows/")}
    )
    store = WorkflowStore(state_dir)
    for name in names:
        folder = store.folder(name)
        try:
            actual = hashlib.sha256((folder / "workflow.json").read_bytes()).hexdigest()
            saved = store.meta(name).sha256
        except (OSError, WorkflowStoreError):
            continue  # restored as it came; the store names what it cannot read when used
        if saved == actual:
            continue
        try:
            store.update(name, _hashed(actual))
        except (OSError, WorkflowStoreError) as exc:
            raise BackupError(f"Restoring stopped at workflow {name}'s hash: {exc}") from exc
        done.rehashed.append(name)


def _hashed(actual: str) -> Callable[[WorkflowMeta], None]:
    def change(meta: WorkflowMeta) -> None:
        meta.sha256 = actual

    return change
