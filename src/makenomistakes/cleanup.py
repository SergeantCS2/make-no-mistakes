"""`mnm clean` and the web's Clean up panel: empty what mnm made while it ran,
short of uninstalling, one target at a time.

Each target says what it removes and how much, before and after. Models, recipes, sets and
the engine images are never touched here (`mnm rm`, `uninstall.sh`). Every deletion goes
through `RunStore` or stays inside its own folder under the state directory.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from makenomistakes import run_lineage
from makenomistakes.comfy.store import WorkflowStore
from makenomistakes.last_logs import LOGS_DIR
from makenomistakes.pipeline.workflow_users import in_use, runs_using
from makenomistakes.runs import RunRecord, RunsRead, RunStore


class Target(StrEnum):
    OUTPUTS = "outputs"  # the files runs made; the records stay, marked
    RUNS = "runs"  # the run history: records, and their files unless kept
    LOGS = "logs"  # engine logs saved when `mnm stop` removed a container
    WORKFLOWS = "workflows"  # earlier versions of imported ComfyUI workflows
    IMPORTS = "imports"  # the imported ComfyUI workflows themselves
    COMFY = "comfy"  # ComfyUI's input, output, temp and home folders, and its history


DESCRIPTIONS = {
    Target.OUTPUTS: "pictures, clips, sound and text that runs made (the run history stays)",
    Target.RUNS: "the run history, with each run's files",
    Target.LOGS: "engine logs kept after `mnm stop` removed their containers",
    Target.WORKFLOWS: "earlier versions of imported workflows (the workflows stay)",
    Target.IMPORTS: "every imported workflow, with its earlier versions (its runs stay)",
    Target.COMFY: (
        "ComfyUI's uploads, temporary and leftover output files, the caches in its home "
        "folder, and its history"
    ),
}
# `mnm clean --auto` (the nightly's option): a run's record with its files, and saved
# engine logs; never models, recipes or a workflow's files
AUTO = (Target.RUNS, Target.LOGS)
# `home` holds only caches (matplotlib's, compiled kernels): ComfyUI's settings are in `user`
COMFY_FOLDERS = ("input", "output", "temp", "home")
ClearHistory = Callable[[], bool]  # ComfyUI's `POST /history {"clear": true}`; False: not asked
GOING = "a workflow is in use (a run going, or a film or a song using one): nothing touched"


@dataclass(frozen=True)
class Item:
    """One target's share: how many things, how many bytes."""

    target: Target
    count: int
    size: int
    note: str = ""  # what else happened (ComfyUI's history cleared, a run going)

    @property
    def description(self) -> str:
        return DESCRIPTIONS[self.target]


def parse_age(text: str) -> timedelta:
    """`30d`, `12h`, `2w`: an age for `--older-than`; ValueError otherwise."""
    units = {"h": timedelta(hours=1), "d": timedelta(days=1), "w": timedelta(weeks=1)}
    number, unit = text[:-1], text[-1:].lower()
    if unit not in units or not number.isdigit():
        raise ValueError(f"'{text}' is not an age such as 12h, 30d or 2w")
    return units[unit] * int(number)


def folder_size(path: Path) -> int:
    """Bytes of the regular files under `path`, links not followed."""
    if path.is_symlink() or not path.is_dir():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            file = Path(root) / name
            if not file.is_symlink():
                total += file.stat().st_size
    return total


class Cleaner:
    def __init__(
        self, store: RunStore, state_dir: Path, history: ClearHistory | None = None
    ) -> None:
        self.store = store
        self.state_dir = state_dir
        self.logs_dir = state_dir / LOGS_DIR
        self.workflows = WorkflowStore(state_dir)
        self.comfy_dir = state_dir / "comfyui"
        self.history = history
        self._read: RunsRead | None = None  # a survey's one read of the runs

    def survey(
        self, older_than: timedelta | None = None, runs: RunsRead | None = None
    ) -> list[Item]:
        """What each target would remove now; `runs`: the runs as the caller read them (the
        Runs page), else read once here for every target."""
        self._read = runs if runs is not None else self.store.read_all()
        try:
            return [self._item(target, older_than) for target in Target]
        finally:
            self._read = None

    def _runs_read(self) -> RunsRead:
        return self._read if self._read is not None else self.store.read_all()

    def _all(self) -> list[RunRecord]:
        return self._runs_read().records

    def comfy_kept(self) -> str | None:
        """Why ComfyUI's folders stay now, None when they may go: a run uses a workflow
        (`workflow_users.runs_using`: a workflow run going, or a film or a song going, waiting
        or interrupted whose clips or song come from one; a run's uploads and outputs are in
        there), or the runs folder could not be listed, so one may (read as no run in
        use, `mnm clean comfy` would clear them under a going run)."""
        runs = self._runs_read()
        if runs.unsure:
            return f"{runs.unsure}, so a run may be using them: nothing touched"
        return GOING if runs_using(runs.records) else None

    def clean(
        self,
        targets: list[Target],
        older_than: timedelta | None = None,
        keep_files: bool = False,
    ) -> list[Item]:
        """Remove each target's share; returns what went. Each target reads the runs once,
        so its count and its keep-checks decide from one snapshot, and the next target sees
        what this one removed."""
        removed: list[Item] = []
        try:
            for target in dict.fromkeys(targets):  # once each, in the order given
                self._read = self.store.read_all()
                removed.append(self._clean_one(target, older_than, keep_files))
        finally:
            self._read = None
        return removed

    def _clean_one(self, target: Target, older_than: timedelta | None, keep_files: bool) -> Item:
        item = self._item(target, older_than)
        match target:
            case Target.OUTPUTS:
                for run in self._runs(older_than, with_files=True):
                    self.store.delete_files(run.id)
            case Target.RUNS:
                for run in self._runs(older_than, with_files=False):
                    self.store.delete(run.id, keep_files=keep_files)
                if keep_files:
                    item = Item(target, item.count, 0)
            case Target.IMPORTS:
                for name in self._imports(older_than)[0]:
                    self.workflows.remove(name)
            case Target.COMFY:
                item = self._clean_comfy(item, older_than)
            case Target.LOGS | Target.WORKFLOWS:
                for path in self._files(target, older_than):
                    path.unlink()
        return item

    def _clean_comfy(self, item: Item, older_than: timedelta | None) -> Item:
        """ComfyUI's folders, then its history, which only goes whole: with an age it stays."""
        if (kept := self.comfy_kept()) is not None:
            return replace(item, note=kept)
        for path in self._files(Target.COMFY, older_than):
            path.unlink()
        if older_than is not None:
            return replace(item, note="its history stays: it is cleared only without an age")
        if self.history is not None and self.history():
            return replace(item, note="ComfyUI's history cleared")
        return item

    def _imports(self, older_than: timedelta | None) -> tuple[list[str], str]:
        """Imported workflows nothing uses (`workflow_users.in_use`: a run, a film or a song,
        a studio stage's pick), imported before the age if given; and why some stay."""
        cut = datetime.now(UTC) - older_than if older_than is not None else None
        used = in_use(self._runs_read(), self.state_dir)
        names = [
            meta.name
            for meta in self.workflows.all()
            if used.why(meta.name) is None and (cut is None or meta.imported_at < cut)
        ]
        return names, used.unread

    def _item(self, target: Target, older_than: timedelta | None) -> Item:
        match target:
            case Target.OUTPUTS:
                runs = self._runs(older_than, with_files=True)
                return Item(target, len(runs), sum(self._size(r) for r in runs))
            case Target.RUNS:
                runs = self._runs(older_than, with_files=False)
                return Item(target, len(runs), sum(self._size(r) for r in runs))
            case Target.IMPORTS:
                names, unread = self._imports(older_than)
                size = sum(folder_size(self.workflows.folder(n)) for n in names)
                return Item(target, len(names), size, note=unread)
            case Target.LOGS | Target.WORKFLOWS | Target.COMFY:
                files = self._files(target, older_than)
                return Item(target, len(files), sum(f.stat().st_size for f in files))

    def _size(self, run: RunRecord) -> int:
        return 0 if run.files_removed else folder_size(self.store.folder(run))

    def _runs(self, older_than: timedelta | None, with_files: bool) -> list[RunRecord]:
        """Finished runs (one going, or a film the user continues, is never cleaned), older
        than the age if given."""
        cut = datetime.now(UTC) - older_than if older_than is not None else None
        read = self._runs_read()
        held = run_lineage.kept(read)  # a waiting song's candidates, an unread record's runs
        return [
            run
            for run in read.records
            if run.id not in held
            and (cut is None or run.created_at < cut)
            and not (with_files and run.files_removed)
        ]

    def _files(self, target: Target, older_than: timedelta | None) -> list[Path]:
        """A file target's regular files (never a link), older than the age if given."""
        match target:
            case Target.LOGS:
                found = list(self.logs_dir.glob("*.log")) if self.logs_dir.is_dir() else []
            case Target.WORKFLOWS:
                found = self.workflows.versions()
            case _ if self.comfy_kept() is not None:
                found = []  # a run's upload and outputs are in there, or may be
            case _:
                found = [
                    path
                    for name in COMFY_FOLDERS
                    if (self.comfy_dir / name).is_dir()
                    for path in (self.comfy_dir / name).rglob("*")
                ]
        cut = (datetime.now(UTC) - older_than).timestamp() if older_than is not None else None
        return sorted(
            path
            for path in found
            if path.is_file()
            and not path.is_symlink()
            and (cut is None or path.stat().st_mtime < cut)
        )
