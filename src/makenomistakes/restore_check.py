"""`mnm restore --test`: a backup restored into a scratch state folder, read back by
the stores, and thrown away, so a backup is known to restore before the day it is needed.

The scratch folder is a fresh temporary one; this machine's readings are copied into it
first, with `machine.json`, so the restore merges readings as a real restore onto this box
would (and counts another box's, kept). Then `restore_state.restore` runs as `mnm restore`
does (`backup.check`: every file parses, no newer schema), and each store reads its files
back: the recipes, the sets, the ComfyUI workflows, the readings. A recipe name saved here
and missing from the backup, or the other way round, is a warning, never a failure: the
backup is older than this box. Nothing of this machine's state folder is written; the apps'
files are never restored
(they live outside the state folder)."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from makenomistakes.backup import BackupError, many, read_backup
from makenomistakes.comfy.store import WorkflowStore as ComfyWorkflows
from makenomistakes.comfy.store import WorkflowStoreError
from makenomistakes.instance_types import InstanceError
from makenomistakes.json_folder import JsonFolder, unlisted
from makenomistakes.measurements import MeasurementStore
from makenomistakes.restore_state import restore
from makenomistakes.store import InstanceStore, StoreError
from makenomistakes.workflows import WorkflowStore as SetStore

SCRATCH_PREFIX = "mnm-restore-test-"


@dataclass
class RestoreCheck:
    file: Path
    ok: bool = False
    recipes: int = 0
    sets: int = 0
    workflows: int = 0
    readings: int = 0
    other_machines: int = 0  # Readings of other boxes, kept as a restore here would
    failure: str = ""  # why it failed, worded whole
    warnings: list[str] = field(default_factory=list)

    @property
    def line(self) -> str:
        """The one line the CLI, the web panel and the nightly print."""
        if not self.ok:
            return f"restored in a scratch state: FAILED: {self.failure}"
        counts = ", ".join(
            (
                many(self.recipes, "recipe", "recipes"),
                many(self.sets, "set", "sets"),
                many(self.workflows, "workflow", "workflows"),
            )
        )
        return f"restored in a scratch state: OK ({counts})"


def check_restore(
    file: Path, live_state: Path, card: str | None, scratch_root: Path | None = None
) -> RestoreCheck:
    """Restore `file` into a scratch state folder (under `scratch_root`, else the system's
    temporary folder), read it back, remove it. Never raises for what the backup holds: a
    failure is `RestoreCheck.failure`."""
    found = RestoreCheck(file)
    scratch = Path(tempfile.mkdtemp(prefix=SCRATCH_PREFIX, dir=scratch_root))
    try:
        _seed_readings(live_state, scratch)
        archive = read_backup(file)
        done = restore(archive, scratch, _nothing_runs, card=card)
        found.other_machines = done.other_machines
        _read_back(scratch, found)
        names = {
            rel.removeprefix("instances/").removesuffix(".json")
            for rel in archive.files
            if rel.startswith("instances/")
        }
        found.warnings = _compared(names, live_state / "instances")
        found.ok = True
    except (BackupError, StoreError, InstanceError, WorkflowStoreError, OSError) as exc:
        found.failure = str(exc) or type(exc).__name__
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return found


def _compared(names: set[str], live: Path) -> list[str]:
    """The archive's recipes against this box's: a live folder the system will not list is
    a warning that they were not compared, never the archive's failure ("does not restore
    ([Errno 13] …)" would blame the archive)."""
    try:
        here = JsonFolder(live).names()
    except OSError as exc:
        why = unlisted("recipes folder", live, exc)  # a warning's words, as `_differences`'
        return [f"{why}, so the backup's recipes were not compared with this box's"]
    return _differences(names, here)


def _nothing_runs(_name: str) -> bool:
    return False  # a scratch state's instances run nowhere


def _seed_readings(live_state: Path, scratch: Path) -> None:
    """This machine's readings and boxes (`machine.json`) in the scratch folder, so the
    archive's are merged into them and counted against this box."""
    boxes = live_state / "machine.json"
    if boxes.is_file() and not boxes.is_symlink():
        shutil.copyfile(boxes, scratch / "machine.json")
    here = live_state / "measurements"
    if not here.is_dir():
        return
    target = scratch / "measurements"
    target.mkdir(parents=True, exist_ok=True)
    for path in here.glob("*.jsonl"):
        if path.is_file() and not path.is_symlink():
            shutil.copyfile(path, target / path.name)


def _read_back(scratch: Path, found: RestoreCheck) -> None:
    """Every store reads what the restore wrote, as `mnm ls`, `sets ls` and the Workflows tab
    would; a file none of them reads is the failure."""
    found.recipes = len(InstanceStore(scratch).list())
    found.sets = len(SetStore(scratch).list())
    flows = ComfyWorkflows(scratch)
    bad = flows.problems()
    if bad:
        raise BackupError(f"ComfyUI workflows this mnm cannot read: {', '.join(bad)}")
    found.workflows = len(flows.all())
    readings = MeasurementStore(scratch)
    for name in readings.names():
        lines = [
            line
            for line in readings.path(name).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        parsed = len(readings.readings(name))
        if parsed < len(lines):
            found.warnings.append(
                f"{many(len(lines) - parsed, 'reading line', 'reading lines')} of {name} do not "
                "parse (calibration skips them)"
            )
        found.readings += parsed


def _differences(backed_up: set[str], here: list[str]) -> list[str]:
    """The recipe names that differ, as warnings: the backup is older than this box."""
    newer = sorted(set(here) - backed_up)
    gone = sorted(backed_up - set(here))
    said = []
    if newer:
        said.append(
            f"saved here and not in the backup (added since it was taken): {', '.join(newer)}"
        )
    if gone:
        said.append(f"in the backup and no longer saved here (removed since): {', '.join(gone)}")
    return said
