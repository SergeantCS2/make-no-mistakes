"""`mnm backup --auto`: a backup into the backups folder, its restore test, then the
rotation; and the record of the last of each, which `mnm backup ls` and the web's System →
Backup panel show.

One path for the timer (`backup_timer.py`), the CLI and the web's Back up now. `mnm nightly`
calls `auto_backup` too and owns the schedule once installed. The rotation runs only after the new
backup restored in a scratch state: a backup that does not restore deletes nothing older.
A failure (the backup, its restore test, or the rotation) sends one notification with the
reason (`Notice`); success is silent. The
record is `<state>/backup/last.json`, schema-versioned, never in a backup (`backup.PATTERNS`
does not name it)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ValidationError

from makenomistakes import config
from makenomistakes.apps.backup_files import to_back_up
from makenomistakes.backup import BackupError, Manifest, many, tally, write_backup
from makenomistakes.backup_rotation import BackupFile, auto_name, listing, ready, rotate
from makenomistakes.config import Settings
from makenomistakes.local_time import local
from makenomistakes.locking import shared_lock
from makenomistakes.notify import Notice
from makenomistakes.report import secrets
from makenomistakes.restore_check import RestoreCheck, check_restore

RECORD_SCHEMA = 1
RECORD = Path("backup") / "last.json"
BACKUP_TITLE = "mnm backup"
TEST_TITLE = "mnm restore --test"
Tell = Callable[[Notice], None]


class Outcome(BaseModel):
    at: datetime  # aware UTC; shown on the box's clock
    file: str  # the archive's name
    ok: bool
    line: str  # what was said, whole


class LastRuns(BaseModel):
    schema_version: int = RECORD_SCHEMA
    backup: Outcome | None = None
    test: Outcome | None = None


def read_last(state_dir: Path) -> LastRuns:
    """The last backup and restore test; empty when none ran or the record cannot be read
    (it is a convenience: the archives themselves are what counts)."""
    try:
        found = LastRuns.model_validate_json((state_dir / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError):
        return LastRuns()
    return found if found.schema_version <= RECORD_SCHEMA else LastRuns()


def _note(state_dir: Path, backup: Outcome | None = None, test: Outcome | None = None) -> None:
    path = state_dir / RECORD
    with shared_lock(path.parent)():
        last = read_last(state_dir)
        changed = last.model_copy(
            update={k: v for k, v in (("backup", backup), ("test", test)) if v is not None}
        )
        partial = path.with_name(path.name + ".part")
        try:
            partial.write_text(changed.model_dump_json(indent=2) + "\n", encoding="utf-8")
            partial.replace(path)
        except OSError:
            partial.unlink(missing_ok=True)  # the record is a convenience; the run stands


@dataclass
class AutoRun:
    file: Path | None = None
    manifest: Manifest | None = None
    check: RestoreCheck | None = None
    removed: list[BackupFile] = field(default_factory=list)
    failure: str = ""  # the backup or the rotation failed: why

    @property
    def ok(self) -> bool:
        return not self.failure and self.check is not None and self.check.ok


def auto_backup(
    settings: Settings,
    card: str | None,
    tell: Tell,
    now: datetime | None = None,
    scratch_root: Path | None = None,
) -> AutoRun:
    """Back up into the backups folder, restore-test it, rotate. Never raises for what goes
    wrong: `AutoRun` says it, and `tell` got one notice."""
    state = settings.expanded_state_dir()
    folder = settings.expanded_backups_dir()
    moment = now or datetime.now(UTC)
    run = AutoRun()
    with shared_lock(state / RECORD.parent)():  # one automatic backup at a time
        try:
            ready(folder)
            out = folder / auto_name(local(moment).replace(tzinfo=None))
            if out.exists():
                raise BackupError(f"{out} exists already: one backup a second at most.")
            known = secrets(settings, config.SECRETS_FILE).known  # read now: tests move it
            run.manifest = write_backup(state, out, known, gpu=card, app_files=to_back_up(settings))
        except (BackupError, OSError) as exc:
            run.failure = f"The backup failed: {exc}"
            _note(state, backup=Outcome(at=moment, file="", ok=False, line=run.failure))
            tell(Notice(BACKUP_TITLE, run.failure, failed=True))
            return run
        run.file = out
        _note(state, backup=Outcome(at=moment, file=out.name, ok=True, line=f"wrote {out.name}"))
        run.check = check_restore(out, state, card, scratch_root)
        _note(state, test=Outcome(at=moment, file=out.name, ok=run.check.ok, line=run.check.line))
        if not run.check.ok:
            tell(
                Notice(
                    BACKUP_TITLE,
                    f"{out.name} was written but does not restore ({run.check.failure}); no "
                    "older backup was deleted.",
                    failed=True,
                )
            )
            return run
        try:
            run.removed = rotate(folder)
        except BackupError as exc:
            run.failure = f"The rotation failed: {exc}"
            tell(Notice(BACKUP_TITLE, run.failure, failed=True))
    return run


def newest(folder: Path) -> Path | None:
    """The newest backup in the folder (an automatic one by its name, any other by its time)."""
    found = listing(folder)
    return found[0].path if found else None


def check_backup(
    settings: Settings,
    file: Path | None,
    card: str | None,
    tell: Tell,
    scratch_root: Path | None = None,
) -> RestoreCheck:
    """`mnm restore --test [FILE]`: the newest backup unless a file is named."""
    state = settings.expanded_state_dir()
    chosen = file or newest(settings.expanded_backups_dir())
    if chosen is None:
        check = RestoreCheck(
            Path(), failure=f"no backup in {settings.expanded_backups_dir()} to test"
        )
    else:
        check = check_restore(chosen, state, card, scratch_root)
    _note(
        state,
        test=Outcome(
            at=datetime.now(UTC),
            file=chosen.name if chosen else "",
            ok=check.ok,
            line=check.line,
        ),
    )
    if not check.ok:
        tell(Notice(TEST_TITLE, check.line, failed=True))
    return check


class Tone(StrEnum):
    """How a line is shown: the CLI's rich style, the web's class."""

    PLAIN = ""
    GOOD = "green"
    WARN = "yellow"
    BAD = "red"
    DIM = "dim"


Said = list[tuple[Tone, str]]
KEPT_WORDS = "the last 7 daily and 4 weekly are kept"


def check_words(check: RestoreCheck) -> Said:
    said: Said = [(Tone.GOOD if check.ok else Tone.BAD, check.line)]
    said += [(Tone.WARN, f"  WARN: {warning}") for warning in check.warnings]
    if check.other_machines:
        said.append(
            (
                Tone.DIM,
                f"  {many(check.other_machines, 'reading', 'readings')} of another machine kept "
                "for when it returns, as a restore here would",
            )
        )
    return said


def run_words(done: AutoRun) -> Said:
    """What `mnm backup --auto` says, line by line: the CLI prints it, the web's job keeps it."""
    if done.file is None:
        return [(Tone.BAD, done.failure)]
    files = done.manifest.files if done.manifest else []
    said: Said = [(Tone.PLAIN, f"Wrote {done.file}: {many(len(files), 'file', 'files')}.")]
    said += [(Tone.PLAIN, f"  {line}") for line in tally(files)]
    if done.check is not None:
        said += check_words(done.check)
        if not done.check.ok:
            return [*said, (Tone.WARN, "No older backup was deleted: this one does not restore.")]
    if done.failure:
        said.append((Tone.BAD, done.failure))
    elif done.removed:
        names = ", ".join(b.name for b in done.removed)
        gone = many(len(done.removed), "older backup", "older backups")
        said.append((Tone.PLAIN, f"Rotation: removed {gone} ({names}); {KEPT_WORDS}."))
    else:
        said.append((Tone.PLAIN, f"Rotation: nothing to remove ({KEPT_WORDS})."))
    return said
