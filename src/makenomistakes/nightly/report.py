"""The nightly's record: one `<state>/nightly/<local date>.json` per run (a second run
the same day is `<date>-2.json`), typed and schema-versioned, written after every step so a run
that dies leaves what it did. The regression rule reads these reports, never the measurements
store: a reading the user takes by hand never moves a nightly baseline.

A report from a newer mnm (a higher `schema_version`) is left out of the rule and listed as
such; one that does not parse is listed by its name and skipped."""

from __future__ import annotations

import contextlib
import re
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from makenomistakes.apps.names import App
from makenomistakes.disk_types import EvictionRecord
from makenomistakes.doctor_types import Check, Status
from makenomistakes.healing_words import HealEvent
from makenomistakes.hub.facts import Engine, Modality
from makenomistakes.json_folder import JsonFolder
from makenomistakes.measurements import SpeedUnit
from makenomistakes.nightly.freshness import Finding
from makenomistakes.try_all import Refusal

# 2: the router step; 3: freshness findings, `newer`; 4: a row's box;
# 5: the `hardware` verdict
SCHEMA_VERSION = 5
FOLDER = "nightly"
_ID = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:-(\d+))?$")


class StepName(StrEnum):
    DOCTOR = "doctor"  # `mnm doctor --deep`
    APPS = "apps"  # every installed or connected app's `mnm doctor --app`
    TRY = "try"  # `mnm try --all --speed`
    BACKUP = "backup"  # `mnm backup --auto`
    FRESHNESS = "freshness"  # `mnm engines check` and the family table
    DISK = "disk"  # `mnm disk` and `mnm disk check`
    CLEAN = "clean"  # `mnm clean --auto`, when the user turned it on
    ROUTER = "router"  # The engines that died since the last night, from the journal


STEPS = tuple(StepName)  # the order a night runs them in


class StepState(StrEnum):
    OK = "ok"
    WARN = "warn"  # done, with something worth reading (a doctor WARN, a row not tried)
    FAILED = "failed"  # it raised, or what it ran said it failed
    SKIPPED = "skipped"  # `--skip`, or not turned on
    NOT_BUILT = "not built"  # from before the freshness step: kept so older reports still read


class TryRecord(BaseModel):
    """One `try --all` row, with the build and settings it ran on (the rule's key)."""

    name: str
    modality: Modality
    engine: Engine | None = None
    build: str | None = None  # the container's image, short id
    settings: str | None = None  # `measurements.settings_key`
    machine: str | None = None  # The box's fingerprint id (None: an older report's)
    load_seconds: float | None = None
    answer_seconds: float | None = None
    speed: float | None = None  # the figure the rule compares, in `unit`
    unit: SpeedUnit | None = None
    shown: str | None = None  # the row's own speed text: `61.2 tok/s`, `4.1 s/image`
    vram_added: int | None = None
    tools: str | None = None
    figures: str = ""
    why: str | None = None
    refusal: Refusal | None = None


class AppRecord(BaseModel):
    app: App
    where: str = ""
    checks: list[Check] = Field(default_factory=list)
    error: str | None = None  # the doctor could not run

    @property
    def ok(self) -> bool:
        return self.error is None and all(c.status is not Status.FAIL for c in self.checks)


class BackupRecord(BaseModel):
    file: str | None = None
    ok: bool
    restored: bool | None = None  # the restore test; None: it never got that far
    failure: str | None = None
    lines: list[str] = Field(default_factory=list)


class DiskRecord(BaseModel):
    held_bytes: int | None = None  # the models folder
    disk_total_bytes: int | None = None
    disk_free_bytes: int | None = None
    unlisted_bytes: int | None = None
    policy: str = ""  # the limit in words
    gauge: str | None = None  # `the models folder's disk is at 71%`
    warning: str | None = None
    evicted: list[EvictionRecord] = Field(default_factory=list)
    refused: str | None = None
    notes: list[str] = Field(default_factory=list)


class CleanRecord(BaseModel):
    target: str
    count: int
    size_bytes: int
    note: str = ""


class StepResult(BaseModel):
    name: StepName
    state: StepState
    seconds: float = 0.0  # monotonic
    summary: str = ""
    error: str | None = None
    checks: list[Check] = Field(default_factory=list)  # doctor
    apps: list[AppRecord] = Field(default_factory=list)
    rows: list[TryRecord] = Field(default_factory=list)  # try
    stopped: str | None = None  # try: "stopped at tiny: the router was in use"
    backup: BackupRecord | None = None
    disk: DiskRecord | None = None
    cleaned: list[CleanRecord] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)  # freshness, each in a line
    fresh: list[Finding] = Field(default_factory=list)  # the same, typed
    healing: list[HealEvent] = Field(default_factory=list)  # router, oldest first


class ActionKind(StrEnum):
    RETUNE = "retune"  # a tune queued in the router's tuner
    BACKUP_AGAIN = "backup again"
    APP_RESTART = "app restart"
    ENGINE_RESTART = "engine restart"  # the router restarted a dead engine
    PARKED = "parked"  # the router parked a crash loop
    OFFERED = "offered"  # a command the user may run; never run by the nightly


class Action(BaseModel):
    kind: ActionKind
    line: str  # what was done, or the command offered
    ok: bool | None = None  # an automatic action's outcome; None for an offer

    @property
    def automatic(self) -> bool:
        return self.kind is not ActionKind.OFFERED


class VerdictKind(StrEnum):
    SLOWER = "slower"  # more than the rule's margin under its baseline
    STEADY = "steady"
    BASELINE = "baseline"  # too few earlier readings on this build and settings
    ROW_FAILED = "row failed"  # a load or answer that failed, a row refused for a fault
    # Not tried for room, not downloaded, stopped by the user, held back: never a regression;
    # the first night for room is told once
    CANNOT_TRY = "cannot try"
    DOCTOR = "doctor"  # a doctor line now FAIL
    APP = "app"  # an app's doctor failing
    BACKUP = "backup"  # the backup failed
    RESTORE = "restore"  # the restore test failed
    EVICTED = "evicted"  # the disk policy deleted an instance's files (said, never a fault)
    DIED = "died"  # An engine died and the router restarted it
    RESTARTING = "restarting"  # died, its restart not ended when the nightly read the journal
    NOT_RESTARTED = "not restarted"  # died, and the router's restart failed
    CRASH_LOOP = "crash loop"  # died twice in ten minutes: the router parked it
    NEWER = "newer"  # A newer engine build or model, an upgrade's `-old` kept (offered)
    HARDWARE = "hardware"  # The box changed since plans were made (`mnm machine adopt`)
    STEP = "step"  # a step that raised or failed on its own


class Verdict(BaseModel):
    kind: VerdictKind
    subject: str  # the instance, the doctor line, the app, the step
    figures: str  # the words with the figures
    regression: bool = False
    told: bool = False  # in the night's notice (a regression, or an app restarted)
    # what the rule's key was, for "never twice for the same regression"
    build: str | None = None
    settings: str | None = None
    action: Action | None = None

    @property
    def line(self) -> str:
        return f"{self.subject}: {self.figures}"


class Outcome(StrEnum):
    """The one word `mnm nightly ls` shows."""

    QUIET = "quiet"
    REGRESSION = "regression"
    TO_READ = "to read"  # told, nothing regressed: a first night not tried for room, an app restart
    FAILED = "failed"
    WAITED = "waited"
    RUNNING = "running"  # not finished (running now, or the process died)


class NightlyReport(BaseModel):
    schema_version: int = SCHEMA_VERSION
    id: str  # `2026-10-26`, `2026-10-26-2`
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    waited: str | None = None  # it never ran: why
    waited_seconds: float = 0.0
    steps: list[StepResult] = Field(default_factory=list)
    verdicts: list[Verdict] = Field(default_factory=list)
    lines: list[str] = Field(default_factory=list)  # every journal line the run wrote
    notice: list[str] = Field(default_factory=list)  # each channel's outcome, or why none

    def step(self, name: StepName) -> StepResult | None:
        return next((s for s in self.steps if s.name is name), None)

    @property
    def outcome(self) -> Outcome:
        if self.waited is not None:
            return Outcome.WAITED
        if self.finished_at is None:
            return Outcome.RUNNING
        if any(s.state is StepState.FAILED for s in self.steps):
            return Outcome.FAILED
        if any(v.regression for v in self.verdicts):
            return Outcome.REGRESSION
        if any(v.told for v in self.verdicts):
            return Outcome.TO_READ
        return Outcome.QUIET

    @property
    def regressions(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.regression]

    @property
    def told(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.regression or v.told]


def order_key(report_id: str) -> tuple[str, int]:
    found = _ID.match(report_id)
    if found is None:
        return ("", 0)
    return (found.group(1), int(found.group(2) or 1))


class Listed(BaseModel):
    """A report's name and what could be read of it."""

    id: str
    report: NightlyReport | None = None
    problem: str | None = None  # unreadable, or from a newer mnm


# Reports kept: three months of nights, well past the rule's window (`nightly.rule`) and
# the speed graph's reach.
KEEP_REPORTS = 90


class NightlyStore:
    def __init__(self, state_dir: Path) -> None:
        self.files = JsonFolder(state_dir / FOLDER)

    @property
    def folder(self) -> Path:
        return self.files.folder

    def ids(self) -> list[str]:
        """Every report's name, oldest first (`-2` after its day's first). A folder the system
        will not list shows none: the reports only show nights, the health answer must never
        fail on them, and the rotation deletes only what it listed."""
        try:
            names = self.files.names()
        except OSError:
            return []
        return sorted((n for n in names if _ID.match(n)), key=order_key)

    def new_id(self, day: date) -> str:
        """`<day>`, or `<day>-N` after the day's earlier runs."""
        taken = {order_key(n)[1] for n in self.ids() if order_key(n)[0] == day.isoformat()}
        if not taken:
            return day.isoformat()
        return f"{day.isoformat()}-{max(taken) + 1}"

    def save(self, report: NightlyReport) -> Path:
        """Written whole; the oldest beyond `KEEP_REPORTS` go (kept forever and parsed for
        every look, a year's worth would be read at each health poll)."""
        with self.files.locked():
            path = self.files.write(report.id, report.model_dump_json(indent=2) + "\n")
            for old in self.ids()[:-KEEP_REPORTS]:
                with contextlib.suppress(OSError, LookupError):
                    self.files.unlink(old)
            return path

    def listed(self, report_id: str) -> Listed:
        try:
            text = self.files.read(report_id)
        except (OSError, LookupError) as exc:
            return Listed(id=report_id, problem=f"cannot be read: {exc}")
        try:
            found = NightlyReport.model_validate_json(text)
        except ValidationError as exc:
            newer = _newer(text)
            if newer:
                return Listed(id=report_id, problem=newer)
            return Listed(id=report_id, problem=f"does not parse: {exc.errors()[0]['msg']}")
        if found.schema_version > SCHEMA_VERSION:
            return Listed(id=report_id, problem=_NEWER)
        return Listed(id=report_id, report=found)

    def every(self) -> list[Listed]:
        """Oldest first."""
        return [self.listed(i) for i in self.ids()]

    def reports(self) -> list[NightlyReport]:
        """The readable reports, oldest first."""
        return [found.report for found in self.every() if found.report is not None]

    def latest(self) -> NightlyReport | None:
        """The newest readable report, parsing only as far back as it must."""
        for report_id in reversed(self.ids()):
            if (found := self.listed(report_id).report) is not None:
                return found
        return None


_NEWER = "written by a newer mnm (`mnm update`)"


def _newer(text: str) -> str | None:
    found = re.search(r'"schema_version"\s*:\s*(\d+)', text)
    return _NEWER if found and int(found.group(1)) > SCHEMA_VERSION else None
