"""Runs and their outputs: everything mnm produces, in one store.

A run is one record under `<state>/runs/<id>.json`: what was asked, of which instance,
workflow or pipeline, with which settings, and the files it made. The files live in
`<outputs>/<yyyy-mm-dd>/<id>/`. The gallery, `mnm outputs` and `mnm clean` read and delete
through this module only, and only inside those two folders.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from makenomistakes import run_lineage
from makenomistakes.folder_hint import folder_hint
from makenomistakes.json_folder import BadNameError, JsonFolder, cause, unlisted
from makenomistakes.local_time import local
from makenomistakes.run_errors import RunDamagedError, RunError
from makenomistakes.run_files import (
    MEDIA_TYPES,
    FileKind,
    file_kind,
    folder_of,
    new_id,
    remove_folder,
    safe_file_name,
)
from makenomistakes.run_parts import (
    ActsOn,
    InputRecord,
    ModelCredit,
    OutputFile,
    Removed,
    RunKind,
    RunRole,
    RunState,
    StepRecord,
    StepState,
)
from makenomistakes.run_process import alive, process_started
from makenomistakes.runs_read import RunsRead, size_limit

if TYPE_CHECKING:
    from makenomistakes.config import Settings

__all__ = [  # moved to `run_files`, `run_parts`, `run_process`, `run_errors`, `runs_read`
    "MEDIA_TYPES",
    "FileKind",
    "OutputFile",
    "Removed",
    "RunDamagedError",
    "RunError",
    "RunKind",
    "RunState",
    "RunsRead",
    "StepRecord",
    "StepState",
    "file_kind",
    "new_id",
    "process_started",
    "safe_file_name",
    "size_limit",
]
# 2: a run's lineage and credits (`run_parts`), every field defaulted: a schema-1 file
# reads as is, and an older runner refuses a run that has a parent instead of dropping it.
SCHEMA_VERSION = 2  # a file from a newer runner is refused, never read with fields dropped
_ID = re.compile(r"^\d{8}-\d{6}-[a-z0-9-]{1,40}-[0-9a-f]{4}$")
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class RunRecord(BaseModel):
    schema_version: int = SCHEMA_VERSION
    id: str
    kind: RunKind
    target: str  # the instance, workflow or pipeline it ran
    state: RunState = RunState.RUNNING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    prompt: str | None = None
    settings: dict[str, str] = Field(default_factory=dict)  # as used, the seed included
    files: list[OutputFile] = Field(default_factory=list)
    answer: str = ""  # one line: what came back, worded for the user
    error: str = ""
    seconds: float | None = None
    numbers: dict[str, float] = Field(default_factory=dict)
    folder: str  # `<yyyy-mm-dd>/<id>`, under the outputs folder
    files_removed: bool = False  # `mnm clean outputs` kept the record, not the files
    # Pipelines: each item's step, the review points passed, the one waited at,
    # and the process running it (a run whose process is gone was interrupted: Resume).
    steps: list[StepRecord] = Field(default_factory=list)
    reviewed: list[str] = Field(default_factory=list)
    waiting_at: str | None = None
    pid: int | None = None
    pid_started: int | None = None  # its start (ticks since boot): a reused pid differs
    prompt_id: str = ""  # a workflow run's id in ComfyUI's queue, once queued (its Stop)
    # a workflow run: the nodes ComfyUI's cache served (its VRAM reading is low then);
    # None: not recorded (a run an older mnm made), counted as unknown
    cached_nodes: list[str] | None = None
    # Where the run sits in a take's tree and what made it
    parent: str | None = None  # the run it is a child of (a candidate, an edit, a mix)
    role: RunRole | None = None
    acts_on: ActsOn | None = None  # an edit's stem, range or section
    models: list[ModelCredit] = Field(default_factory=list)
    voices: list[str] = Field(default_factory=list)  # voice rows
    inputs_recorded: list[InputRecord] = Field(default_factory=list)
    score: float | None = None  # a candidate's score, with what scored it
    scorer: str | None = None
    score_axes: dict[str, float] = Field(default_factory=dict)  # every axis it measured
    disclosure: str = ""  # rendered once at export and kept
    notes: list[str] = Field(default_factory=list)  # worded notes, never refusals

    @property
    def outputs(self) -> list[OutputFile]:
        return [f for f in self.files if f.role == "output"]

    @property
    def size(self) -> int:
        return 0 if self.files_removed else sum(f.size for f in self.files)

    def first(self, kind: FileKind) -> OutputFile | None:
        return next((f for f in self.outputs if f.kind is kind), None)

    def step(self, stage: str, index: int = 0) -> StepRecord | None:
        return next((s for s in self.steps if s.stage == stage and s.index == index), None)

    @property
    def going(self) -> bool:
        """Running in a live process: never deleted or cleaned under it."""
        return self.state is RunState.RUNNING and not self.interrupted

    @property
    def kept_for_owner(self) -> bool:
        """Never cleaned or pruned: going, or a film the user continues (waiting at a review,
        or interrupted and resumable), or a training the router resumes (once queued: its
        step is saved last, so one killed while its inputs were copied in is no training to
        resume). `outputs rm <id>` still deletes one by name."""
        resumes = self.kind is RunKind.PIPELINE or (self.kind is RunKind.TRAINING and self.steps)
        resumable = resumes and (self.state is RunState.WAITING or self.interrupted)
        return self.going or bool(resumable)

    @property
    def interrupted(self) -> bool:
        """Running, but its process is gone (a restart, a closed terminal): resume it."""
        return (
            self.state is RunState.RUNNING
            and self.pid is not None
            and not alive(self.pid, self.pid_started)
        )


class _Head(BaseModel):
    """What a record says of itself, read before the rest: its version and its id."""

    model_config = ConfigDict(extra="ignore")
    schema_version: int = 1
    id: str = ""


class RunStore:
    """Run records under `<state>/runs`, their files under the outputs folder."""

    def __init__(self, state_dir: Path, outputs_dir: Path) -> None:
        self._records = JsonFolder(state_dir / "runs")
        self.outputs_dir = outputs_dir
        self._once = threading.local()  # `reading_once`'s records, per thread

    @classmethod
    def for_settings(cls, settings: Settings) -> RunStore:
        """This box's store: records under the state folder, files under `MNM_OUTPUTS_DIR`."""
        return cls(settings.expanded_state_dir(), settings.expanded_outputs_dir())

    # -- records -------------------------------------------------------------------------

    def new(  # noqa: PLR0913 - what a record is made with, its place in its first save
        self,
        kind: RunKind,
        target: str,
        prompt: str | None = None,
        settings: dict[str, str] | None = None,
        now: datetime | None = None,
        *,
        parent: str | None = None,
        role: RunRole | None = None,
        steps: list[StepRecord] | None = None,
    ) -> RunRecord:
        """A running record and its empty folder; a child's place under its parent and a
        pipeline's steps are in its first save (a child saved before them would be left a
        top-level run when what followed failed)."""
        when = now or datetime.now(UTC)
        named = local(when)  # the id and its day folder on the box's clock; the record UTC
        run_id = new_id(target, named)
        record = RunRecord(
            id=run_id,
            kind=kind,
            target=target,
            created_at=when,
            prompt=prompt,
            settings=dict(settings or {}),
            folder=folder_of(run_id),
            pid=os.getpid(),  # a run whose process is gone reads interrupted: cleanable
            pid_started=process_started(os.getpid()),
            parent=parent,
            role=role,
            steps=list(steps or []),
        )
        try:
            self.folder(record).mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            hint = folder_hint(self.outputs_dir, "MNM_OUTPUTS_DIR")
            raise RunError(
                f"Cannot make the run's folder under {self.outputs_dir}: "
                f"{exc.strerror or exc}. {hint or 'Set MNM_OUTPUTS_DIR to a folder you can write.'}"
            ) from exc
        self.save(record)
        return record

    def locked(self) -> AbstractContextManager[None]:
        """The records' lock (a flock across processes): a read, a check and a save under it
        are one step, so the CLI and the web never both take one run."""
        return self._records.locked()

    def save(self, record: RunRecord) -> None:
        with self._records.locked():
            self._records.write(record.id, record.model_dump_json(indent=2))

    def get(self, run_id: str) -> RunRecord:
        if not _ID.match(run_id):
            raise RunError(f"'{run_id}' is not a run id (`mnm outputs ls` lists them).")
        try:
            data = self._records.path(run_id).read_bytes()
        except (BadNameError, FileNotFoundError):
            raise RunError(f"No run '{run_id}' (`mnm outputs ls` lists them).") from None
        except OSError as exc:  # a permission, an I/O error: never "damaged"
            path = self._records.folder / f"{run_id}.json"
            raise RunError(f"The run file {path} could not be read ({cause(exc)}).") from exc
        return self._parse(run_id, data)

    def all(self) -> list[RunRecord]:
        """Every readable run, newest first; a file this version cannot read is skipped
        (`problems` names them)."""
        return self.read_all().records

    def problems(self) -> list[str]:
        """Run files this version cannot read, by name."""
        return self.read_all().problems

    def read_all(self) -> RunsRead:
        """`all` and `problems` from one read of every file (the Runs page's one parse). A
        folder the system will not list is a problem with no records, and says so in
        `unlisted`: every page and command that shows runs shows it, the gallery and prune
        delete nothing they do not list, and what keeps the files a run uses keeps everything
        (`RunsRead.unlisted`)."""
        found, bad, unread = [], [], []
        try:
            names = self._records.names()
        except OSError as exc:
            why = unlisted("runs folder", self._records.folder, exc)
            return RunsRead([], [why], unlisted=why)
        for name in names:
            path = self._records.folder / f"{name}.json"
            try:
                found.append(self._parse(name, path.read_bytes()))
            except RunError as exc:  # its words name the file and what clears it
                bad.append(str(exc))
                unread.append(name)
            except OSError as exc:
                bad.append(f"The run file {path} could not be read ({cause(exc)}).")
                unread.append(name)
        newest = sorted(found, key=lambda r: r.created_at, reverse=True)
        return RunsRead(newest, bad, unread=tuple(unread))

    def children(self, run_id: str) -> list[RunRecord]:
        """The runs made under this one (a song's candidates), oldest first."""
        held: list[RunRecord] | None = getattr(self._once, "records", None)
        return run_lineage.children(held if held is not None else self.all(), run_id)

    @contextmanager
    def reading_once(self) -> Iterator[None]:
        """Every `children` asked inside, on this thread, reads the records once (the Song
        page asks for a song's mixes, edits and masters: each would be a parse of every
        record). For a page's reads only: nothing saved inside shows."""
        if getattr(self._once, "records", None) is not None:
            yield
            return
        self._once.records = self.all()
        try:
            yield
        finally:
            self._once.records = None

    def lineage(self, run_id: str) -> list[RunRecord]:
        """This run and its parents up to the first, this one first."""
        return run_lineage.lineage(self.all(), run_id)

    def _parse(self, name: str, data: bytes) -> RunRecord:
        """The record, its version and id read first (`_Head`): a newer mnm's record that
        carries a kind or role this one lacks is newer, never damaged (read whole first,
        it would be called damaged, and `mnm outputs rm` would delete it).
        Damaged (`RunDamagedError`, removable by `mnm outputs rm`) is only what no mnm
        reads: not JSON, not a record, naming another run."""
        path = self._records.folder / f"{name}.json"
        named = _ID.match(name) is not None  # else `mnm outputs rm` refuses the name
        clears = (
            f"`mnm outputs rm {name}` removes it and its outputs folder (`--keep-files` keeps "
            "the folder)"
        )
        stray = (  # a copy, a note: mnm names nothing it would remove
            f"The file {path} is not named as a run (a run's file is its id): mnm reads and "
            "removes nothing of it; remove it by hand."
        )
        try:
            head = _Head.model_validate_json(data)
        except ValidationError as exc:
            why = "not a run record" if exc.errors()[0]["type"] != "json_invalid" else "not JSON"
            if not named:
                raise RunError(stray) from None
            raise RunDamagedError(f"The run file {path} is {why}: {clears}.") from None
        if head.schema_version > SCHEMA_VERSION:  # before the name: never damaged
            raise RunError(f"The run file {path} is from a newer mnm: run `mnm update`.")
        if not named:
            raise RunError(stray)
        if head.id != name:
            says = f"names another run ({head.id})" if head.id else "names no run"
            raise RunDamagedError(f"The run file {path} {says}: {clears}.")
        try:
            record = RunRecord.model_validate_json(data)
        except ValidationError as exc:  # its own, at this version: a newer mnm's value?
            err = exc.errors()[0]
            where = ".".join(str(part) for part in err["loc"])
            raise RunError(
                f"The run file {path} cannot be read ({where}: {err['msg']}): a newer mnm "
                "that added a value may have written it (`mnm update` reads it then); else "
                f"remove the file and its outputs folder "
                f"{self.outputs_dir / folder_of(name)} by hand."
            ) from None
        # 1 -> 2 is a no-op: the new fields default; saved again, the file says 2
        record.schema_version = SCHEMA_VERSION
        return record

    # -- files ---------------------------------------------------------------------------

    def folder(self, record: RunRecord) -> Path:
        """The run's folder; RunError for a record whose folder would leave the outputs."""
        day, _, rest = record.folder.partition("/")
        if not _DAY.match(day) or rest != record.id:
            raise RunError(f"The run {record.id} names a folder outside the outputs folder.")
        return self.outputs_dir / day / record.id

    def file(self, record: RunRecord, name: str) -> Path:
        """One of the run's files, by the name its record lists (never a path)."""
        if record.files_removed or name not in {f.name for f in record.files}:
            raise RunError(f"The run {record.id} has no file '{name}'.")
        path = self.folder(record) / name
        if path.is_symlink() or not path.is_file():
            raise RunError(f"The file {name} of run {record.id} is gone.")
        return path

    def inputs(self, record: RunRecord, roles: list[str]) -> dict[str, tuple[str, bytes]]:
        """What the user gave a run, by the role it was kept under (`input-<role>-<name>`),
        read back for "Run again". An input that is gone, or kept under a role not asked
        for (an older run's name), is a RunError: a repeat never runs without it."""
        found: dict[str, tuple[str, bytes]] = {}
        by_length = sorted(roles, key=len, reverse=True)  # `30.image` before `3`
        for item in (f for f in record.files if f.role == "input"):
            role = next((r for r in by_length if item.name.startswith(f"input-{r}-")), None)
            if role is None:
                raise RunError(
                    f"Run {record.id} kept {item.name} under a name this version cannot give "
                    "back to a setting: run it again from the form."
                )
            if record.files_removed:
                raise RunError(f"The files of run {record.id} were deleted, {item.name} with them.")
            try:
                data = self.file(record, item.name).read_bytes()
            except OSError as exc:
                raise RunError(f"The file {item.name} of run {record.id} cannot be read.") from exc
            found[role] = (item.name.removeprefix(f"input-{role}-"), data)
        return found

    def keep(self, record: RunRecord, path: Path, role: str = "output") -> None:
        """A file a step made in the run's folder, listed (again, if it was remade)."""
        if path.parent != self.folder(record) or not path.is_file():
            raise RunError(f"{path} is not a file in the run's folder.")
        record.files = [f for f in record.files if f.name != path.name]
        size = path.stat().st_size
        record.files.append(
            OutputFile(name=path.name, kind=file_kind(path.name), size=size, role=role)
        )

    def add_input(self, record: RunRecord, name: str, data: bytes, role: str = "file") -> Path:
        """What the user gave the run (a picture, a clip), kept beside its outputs under its
        role (`input-mask-<name>`): two inputs of one name never overwrite each other."""
        safe = safe_file_name(name, "input.bin")
        path = self.folder(record) / f"input-{role}-{safe}"
        path.unlink(missing_ok=True)
        path.write_bytes(data)
        record.files.append(
            OutputFile(name=path.name, kind=file_kind(path.name), size=len(data), role="input")
        )
        self.save(record)
        return path

    def finish(
        self,
        record: RunRecord,
        made: list[Path],
        answer: str,
        seconds: float | None,
        numbers: dict[str, float] | None = None,
    ) -> RunRecord:
        """Done: the files it made (each inside its folder), the one-line answer."""
        folder = self.folder(record)
        for path in made:
            if path.parent != folder or not path.is_file():
                raise RunError(f"{path} is not a file in the run's folder {folder}.")
            if path.name not in {f.name for f in record.files}:
                kind = file_kind(path.name)
                size = path.stat().st_size
                record.files.append(OutputFile(name=path.name, kind=kind, size=size))
        record.state = RunState.DONE
        record.answer = answer
        record.seconds = seconds
        record.numbers = dict(numbers or {})
        record.finished_at = datetime.now(UTC)
        self.save(record)
        return record

    def fail(self, record: RunRecord, error: str) -> RunRecord:
        record.state = RunState.FAILED
        record.error = error
        record.finished_at = datetime.now(UTC)
        self.save(record)
        return record

    # -- deleting ------------------------------------------------------------------------

    def delete(self, run_id: str, keep_files: bool = False) -> RunRecord:
        """The record, and its folder unless `keep_files`; a run still going is refused (its
        work would write the record back)."""
        record = self.get(run_id)
        if record.going:
            raise RunError(f"{record.id} is still running: stop it, or wait, then delete it.")
        if not keep_files:
            self._remove_folder(record)
        with self._records.locked():
            self._records.unlink(record.id)
        return record

    def remove(self, run_id: str, keep_files: bool = False) -> Removed:
        """`mnm outputs rm`: a run (as `delete`), or a damaged record and, unless `keep_files`,
        the folder its id names, `<day>/<id>` (the id and the day folder come from one time,
        `new`); one read under the records' lock, so a record rewritten meanwhile is never
        taken for the damaged one it replaced."""
        with self._records.held(self._refused):
            try:
                record = self.get(run_id)
            except RunDamagedError:
                folder = self.outputs_dir / folder_of(run_id)
                gone = not keep_files and (folder.is_dir() or folder.is_symlink())
                try:
                    if gone:
                        remove_folder(folder)
                    self._records.unlink(run_id)
                except OSError as exc:
                    raise RunError(f"{run_id} could not be removed ({cause(exc)}).") from exc
                return Removed(run_id, 0, damaged=True, folder=folder if gone else None)
            size = 0 if keep_files else record.size
            return Removed(self.delete(run_id, keep_files).id, size)

    def _refused(self, exc: OSError) -> RunError:
        return RunError(
            f"The runs folder {self._records.folder} could not be locked ({cause(exc)})."
        )

    def delete_files(self, run_id: str) -> RunRecord:
        """The run's files go; its record stays, saying so."""
        record = self.get(run_id)
        self._remove_folder(record)
        record.files_removed = True
        self.save(record)
        return record

    def delete_file(self, run_id: str, name: str) -> RunRecord:
        record = self.get(run_id)
        self.file(record, name).unlink()
        record.files = [f for f in record.files if f.name != name]
        self.save(record)
        return record

    def _remove_folder(self, record: RunRecord) -> None:
        remove_folder(self.folder(record))
