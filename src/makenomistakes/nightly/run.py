"""One nightly run, as the timer runs it and `mnm nightly run` and the web's Run now:
wait for the card (`gate.wait`), then each step in order, each guarded (a step that raises
is that step's result, never the night's end), the report saved after every step; then the
rule's verdicts against the earlier nights, the actions, the one notice. One run at a time
(`<state>/nightly/run/.lock`); the report's name is the local date (`-2` for a second)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from makenomistakes.doctor_types import Status
from makenomistakes.healing_words import HealKind
from makenomistakes.journal import Note
from makenomistakes.local_time import local
from makenomistakes.locking import FolderLock, shared_lock
from makenomistakes.nightly.actions import Doers, act
from makenomistakes.nightly.freshness import Finding, FindingKind
from makenomistakes.nightly.gate import HOLD_SECONDS, IDLE_SECONDS, LIMIT_SECONDS, Gate, wait
from makenomistakes.nightly.notice import send
from makenomistakes.nightly.report import (
    FOLDER,
    STEPS,
    NightlyReport,
    NightlyStore,
    StepName,
    StepResult,
    StepState,
    VerdictKind,
    order_key,
)
from makenomistakes.nightly.rule import judge
from makenomistakes.nightly.steps import Steps
from makenomistakes.notify import Notice
from makenomistakes.readings_filter import here_of

RUN_LOCK = "run"  # <state>/nightly/run/.lock: one run at a time


class Skip(StrEnum):
    """What `--skip` leaves out: a step, or `deep` (the doctor without `--deep`: no GPU
    container test, no RAM bandwidth)."""

    DOCTOR = "doctor"
    DEEP = "deep"
    APPS = "apps"
    TRY = "try"
    BACKUP = "backup"
    FRESHNESS = "freshness"
    DISK = "disk"
    CLEAN = "clean"
    ROUTER = "router"


@dataclass(frozen=True)
class Options:
    now: bool = False  # skip the idle minute (never the busy check)
    skip: frozenset[Skip] = frozenset()
    clean: timedelta | None = None  # the user's `clean=`; None: the step is off
    limit_seconds: float = LIMIT_SECONDS  # how long to wait for the card at most
    wait: bool = True  # the web's Run now: refused at once instead of waiting


@dataclass
class NightlyRun:
    steps: Steps
    gate: Gate
    store: NightlyStore
    doers: Doers
    channels: Callable[[Notice], list[str]]  # the notifier's `send`
    say: Note
    clock: Callable[[], float] = time.monotonic
    today: Callable[[], date] = field(default=lambda: local().date())

    def run(self, options: Options) -> NightlyReport:
        """The night, holding the router's tuner off the card from the wait to the end
        (`gate`), let go however it ends."""
        try:
            return self._night(options)
        finally:
            self.gate.hold(0)

    def _night(self, options: Options) -> NightlyReport:
        report = NightlyReport(id=self.store.new_id(self.today()))
        self.store.save(report)
        idle = 0.0 if options.now else IDLE_SECONDS
        limit = options.limit_seconds if options.wait else 0.0
        waited = wait(self.gate, idle, limit, say=self.say)
        report.waited_seconds = waited.seconds
        if waited.gave_up is not None:
            minutes = waited.seconds / 60
            report.waited = (
                f"waited {minutes:.0f} min for the card and gave up: {waited.gave_up}"
                if options.wait
                else f"not run: {waited.gave_up}"
            )
            return self._finish(report)
        for name in STEPS:
            self.gate.hold(HOLD_SECONDS)  # the lease renewed before each step
            report.steps.append(self._step(name, options, report))
            self.store.save(report)
        earlier = [r for r in self.store.reports() if order_key(r.id) < order_key(report.id)]
        report.verdicts = judge(report, earlier, here=here_of(self.store.folder.parent))
        for verdict in report.verdicts:
            if verdict.kind is VerdictKind.BASELINE:
                self._said(report, f"{verdict.subject}: {verdict.figures}")
        act(report, earlier, self.doers, lambda line: self._said(report, line))
        report.notice = send(report, self.channels)
        for line in report.notice:
            self._said(report, f"notice: {line}")
        return self._finish(report)

    def _finish(self, report: NightlyReport) -> NightlyReport:
        report.finished_at = datetime.now(UTC)
        self._said(report, f"The nightly {report.id}: {report.outcome.value}.")
        if report.waited:
            self._said(report, report.waited)
        self.store.save(report)
        return report

    def _said(self, report: NightlyReport, line: str) -> None:
        report.lines.append(line)
        self.say(line)

    def _step(self, name: StepName, options: Options, report: NightlyReport) -> StepResult:
        if Skip(name.value) in options.skip:
            return StepResult(name=name, state=StepState.SKIPPED, summary="skipped (--skip)")
        began = self.clock()
        try:
            result = self._run(name, options, report)
        except Exception as exc:
            error = " ".join(str(exc).split()) or type(exc).__name__
            result = StepResult(name=name, state=StepState.FAILED, summary="failed", error=error)
        result.seconds = round(self.clock() - began, 2)
        said = f"{name.value}: {result.state.value} in {result.seconds:.0f} s"
        self._said(report, f"{said}: {result.error or result.summary}")
        return result

    def _run(self, name: StepName, options: Options, report: NightlyReport) -> StepResult:
        return _RUNS[name](self, options, report)

    def _doctor(self, options: Options, _report: NightlyReport) -> StepResult:
        checks = self.steps.doctor(Skip.DEEP not in options.skip)
        counts = {s: sum(c.status is s for c in checks) for s in Status}
        summary = ", ".join(f"{n} {s.value.upper()}" for s, n in counts.items())
        bad = counts[Status.FAIL] or counts[Status.WARN]
        return StepResult(name=StepName.DOCTOR, state=_warn(bad), summary=summary, checks=checks)

    def _apps(self, _options: Options, _report: NightlyReport) -> StepResult:
        apps = self.steps.apps()
        failing = [a.app.value for a in apps if not a.ok]
        summary = (
            f"{len(apps) - len(failing)} of {len(apps)} pass"
            + (f"; failing: {', '.join(failing)}" if failing else "")
            if apps
            else "no app installed or connected"
        )
        return StepResult(name=StepName.APPS, state=_warn(failing), summary=summary, apps=apps)

    def _try(self, _options: Options, _report: NightlyReport) -> StepResult:
        tried = self.steps.tried(self.gate.in_use)
        faults = [r for r in tried.rows if r.refusal is not None and not r.refusal.chosen]
        done = sum(r.refusal is None for r in tried.rows)
        stopped = f"; {tried.stopped}" if tried.stopped else ""
        return StepResult(
            name=StepName.TRY,
            state=_warn(faults or tried.stopped),
            summary=f"{done} of {len(tried.rows)} tried{stopped}",
            rows=tried.rows,
            stopped=tried.stopped,
        )

    def _backup(self, _options: Options, _report: NightlyReport) -> StepResult:
        record = self.steps.backup()
        state = StepState.OK if record.ok else StepState.FAILED
        summary = record.lines[0] if record.ok and record.lines else record.failure or ""
        return StepResult(name=StepName.BACKUP, state=state, summary=summary, backup=record)

    def _freshness(self, _options: Options, _report: NightlyReport) -> StepResult:
        found = self.steps.freshness.check()
        unchecked = [f for f in found if f.kind is FindingKind.UNCHECKED]
        return StepResult(
            name=StepName.FRESHNESS,
            state=_warn(unchecked),  # an offer is no warning; a check that could not run is
            summary=fresh_summary(found),
            findings=[f.line for f in found],
            fresh=found,
        )

    def _disk(self, _options: Options, report: NightlyReport) -> StepResult:
        disk = self.steps.disk(report.started_at)
        return StepResult(
            name=StepName.DISK,
            state=_warn(disk.warning or disk.refused),
            summary=disk.warning or disk.gauge or disk.policy,
            disk=disk,
        )

    def _clean(self, options: Options, _report: NightlyReport) -> StepResult:
        if options.clean is None:
            off = "off (`mnm nightly set clean=30d` turns it on)"
            return StepResult(name=StepName.CLEAN, state=StepState.SKIPPED, summary=off)
        cleaned = self.steps.clean(options.clean)
        summary = "; ".join(f"{c.target}: {c.count}" for c in cleaned)
        return StepResult(name=StepName.CLEAN, state=StepState.OK, summary=summary, cleaned=cleaned)

    def _router(self, _options: Options, report: NightlyReport) -> StepResult:
        """What the router's journal says died since the last night (else the last day),
        and what the router did about each."""
        if self.steps.router is None:  # a run built without the router's journal
            return StepResult(name=StepName.ROUTER, state=StepState.SKIPPED, summary="not read")
        earlier = [r for r in self.store.reports() if order_key(r.id) < order_key(report.id)]
        since = earlier[-1].started_at if earlier else report.started_at - timedelta(days=1)
        events = self.steps.router(since)
        if events is None:
            words = "the router did not answer: its journal could not be read"
            return StepResult(name=StepName.ROUTER, state=StepState.WARN, summary=words)
        deaths = sum(e.kind in (HealKind.DIED, HealKind.LOOPED) for e in events)
        trouble = [e for e in events if e.kind in (HealKind.NOT_RESTARTED, HealKind.LOOPED)]
        died = "1 engine death" if deaths == 1 else f"{deaths} engine deaths"
        summary = f"{died} since {local(since):%Y-%m-%d %H:%M}" if deaths else "no engine died"
        return StepResult(
            name=StepName.ROUTER, state=_warn(trouble), summary=summary, healing=events
        )


_RUNS: dict[StepName, Callable[[NightlyRun, Options, NightlyReport], StepResult]] = {
    StepName.DOCTOR: NightlyRun._doctor,
    StepName.APPS: NightlyRun._apps,
    StepName.TRY: NightlyRun._try,
    StepName.BACKUP: NightlyRun._backup,
    StepName.FRESHNESS: NightlyRun._freshness,
    StepName.DISK: NightlyRun._disk,
    StepName.CLEAN: NightlyRun._clean,
    StepName.ROUTER: NightlyRun._router,
}


_FRESH_WORDS = {
    FindingKind.ENGINE: ("engine update offered", "engine updates offered"),
    FindingKind.MODEL: ("newer model", "newer models"),
    FindingKind.KEPT: ("-old kept since an upgrade", "-old instances kept since upgrades"),
    FindingKind.UNCHECKED: ("could not be checked", "could not be checked"),
}


def fresh_summary(found: list[Finding]) -> str:
    """`1 engine update offered, 2 newer models`, or that everything is current."""
    parts = []
    for kind, (one, many) in _FRESH_WORDS.items():
        count = sum(f.kind is kind for f in found)
        if count:
            parts.append(f"{count} {one if count == 1 else many}")
    return ", ".join(parts) or "everything is current"


def _warn(something: object) -> StepState:
    return StepState.WARN if something else StepState.OK


def running_lock(state_dir: Path) -> FolderLock:
    """The lock one run holds (`attempt()`: False when another run holds it)."""
    return shared_lock(state_dir / FOLDER / RUN_LOCK)
