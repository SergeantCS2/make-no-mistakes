"""What each of the nightly's steps runs on this box: the same functions the commands
run (`mnm doctor --deep`, `mnm doctor --app`, `mnm try --all --speed`, `mnm backup --auto`,
`mnm disk check`, `mnm clean`), each returning the typed record the report keeps. The runner
(`run.py`) takes them as `Steps`, so each can be swapped."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

from makenomistakes import try_run
from makenomistakes.apps.cards import local_manifest
from makenomistakes.apps.context import AppContext
from makenomistakes.apps.doctor_run import app_doctor
from makenomistakes.apps.files import Ledger
from makenomistakes.apps.hub import AppsService
from makenomistakes.apps.names import App
from makenomistakes.backup_auto import auto_backup, run_words
from makenomistakes.cleanup import AUTO, Cleaner
from makenomistakes.comfy.engine import ComfyEngine
from makenomistakes.config import Settings
from makenomistakes.disk import disk_report
from makenomistakes.disk_policy import check as disk_check
from makenomistakes.disk_settings import DiskSettings, DiskSettingsError
from makenomistakes.doctor_deep import deep_tail
from makenomistakes.doctor_types import Check
from makenomistakes.hardware.probe import SubprocessRunner, probe
from makenomistakes.healing_words import HealEvent, parse
from makenomistakes.hub.facts import CHAT_LIKE, Modality
from makenomistakes.instances import InstanceService
from makenomistakes.journal import Note
from makenomistakes.machine_checks import machine_checks
from makenomistakes.measurements import SpeedUnit, gpu_name, settings_key
from makenomistakes.nightly.freshness import Freshness
from makenomistakes.nightly.report import (
    AppRecord,
    BackupRecord,
    CleanRecord,
    DiskRecord,
    TryRecord,
)
from makenomistakes.notify import Notice
from makenomistakes.readings_filter import here_of
from makenomistakes.recipes import Recipe
from makenomistakes.router_client import router_events
from makenomistakes.runs import RunStore
from makenomistakes.try_all import Row, Stopped, steps_for, try_all
from makenomistakes.try_figures import Figure

PER_PICTURE = {
    Modality.IMAGE_GEN: SpeedUnit.IMAGE,
    Modality.IMAGE_EDIT: SpeedUnit.IMAGE,
    Modality.VIDEO: SpeedUnit.CLIP,
}


InUse = Callable[[frozenset[str]], str | None]  # the gate's `in_use`, given the row's own


@dataclass(frozen=True)
class Tried:
    rows: list[TryRecord]
    stopped: str | None = None  # "stopped at tiny: the router was in use (…)"


@dataclass(frozen=True)
class Steps:
    """The nightly's steps as callables; anything they raise is the step's error."""

    doctor: Callable[[bool], list[Check]]  # deep or not
    apps: Callable[[], list[AppRecord]]
    tried: Callable[[InUse], Tried]  # given the between-rows check
    backup: Callable[[], BackupRecord]
    disk: Callable[[datetime], DiskRecord]  # since the night began: its evictions
    clean: Callable[[timedelta], list[CleanRecord]]
    freshness: Freshness
    # The router's healing lines since then, oldest first; None: no router answered
    router: Callable[[datetime], list[HealEvent] | None] | None = None


def doctor_checks(service: InstanceService, deep: bool) -> list[Check]:
    """`mnm doctor --deep`'s lines: the machine, the GPU container test, the RAM's speed
    (`--skip deep`: `mnm doctor`'s alone)."""
    settings = service.settings
    report = probe(settings.expanded_models_dir(), docker_gpu_smoke=deep, host_ram=True)
    checks = machine_checks(settings, report, deep=deep)
    return [*checks, *deep_tail(settings, report)] if deep else checks


def _apps(service: InstanceService) -> AppsService:
    ctx = AppContext(
        service.settings, SubprocessRunner(), lambda: local_manifest(service), say=_quiet
    )
    return AppsService(ctx)


def _quiet(_line: str) -> None:
    return None


def managed_apps(service: InstanceService) -> frozenset[App]:
    """The apps mnm installed (`mnm apps install`): those it may restart."""
    return frozenset(s.app for s in _apps(service).statuses() if s.installed is not None)


def doctored_apps(service: InstanceService) -> list[App]:
    """Every app mnm installed or wrote a file for (`--write`, `connect`), in `App`'s order."""
    ours = managed_apps(service)
    written = {w.app for w in Ledger(service.settings.expanded_state_dir()).read().files.values()}
    return [app for app in App if app in ours or app.value in written]


def app_record(service: InstanceService, app: App) -> AppRecord:
    try:
        found = app_doctor(app, service)
    except Exception as exc:  # one app's doctor never stops the others'
        return AppRecord(app=app, error=" ".join(str(exc).split()) or type(exc).__name__)
    return AppRecord(app=app, where=found.where, checks=found.checks)


def restart_app(service: InstanceService, app: App) -> AppRecord:
    apps = _apps(service)
    apps.stop(app)
    apps.start(app)
    return app_record(service, app)


def speed_of(row: Row) -> tuple[float | None, SpeedUnit | None]:
    """The figure the rule compares: a chat's decode tok/s, a picture's s/image, a clip's
    s/clip; nothing for the other types (their one request's time says little)."""
    if row.why is not None:
        return None, None
    if (unit := PER_PICTURE.get(row.modality)) is not None:
        return row.answer_seconds, unit
    if row.modality in CHAT_LIKE:
        for figure in (Figure.PREDICTED_PER_SECOND, Figure.REQUEST_TOKENS_PER_SECOND):
            if figure in row.numbers:
                return row.numbers[figure], SpeedUnit.TOKENS
    return None, None


def record(
    row: Row,
    recipe: Recipe | None,
    ran_on: tuple[str | None, str] | None,
    machine: str | None = None,
) -> TryRecord:
    speed, unit = speed_of(row)
    build, key = ran_on if ran_on is not None else (None, None)
    if key is None and recipe is not None:  # never loaded: the settings it would have run
        key = settings_key(recipe)
    return TryRecord(
        name=row.name,
        modality=row.modality,
        engine=recipe.engine if recipe is not None else None,
        build=build,
        settings=key,
        machine=machine,
        load_seconds=row.load_seconds,
        answer_seconds=row.answer_seconds,
        speed=speed,
        unit=unit,
        shown=row.speed,
        vram_added=row.vram_added,
        tools=row.tools,
        figures=row.figures,
        why=row.why,
        refusal=row.refusal,
    )


def try_every(service: InstanceService, in_use: InUse, say: Note) -> Tried:
    """`mnm try --all --speed` (its `steps_for`), each row with the build and settings key it
    ran on; `in_use` between rows stops the run when the router got busy with anything but
    the row just tried."""
    recipes = {r.name: r for r in service.all()}
    ran_on: dict[str, tuple[str | None, str]] = {}
    last: list[str] = []
    with try_run.http_client(service.settings) as http, tempfile.TemporaryDirectory() as out:
        steps = steps_for(service, http, Path(out), True, try_run.used_bytes, say)
        load = steps.load

        def loaded(recipe: Recipe) -> Recipe:
            running = load(recipe)
            try:
                build = service.status(running).build
            except Exception:  # Docker unreadable now: the row stands, unkeyed
                build = None
            ran_on[running.name] = (build, settings_key(running))
            last[:] = [running.name]
            return running

        steps = replace(steps, load=loaded)
        stopped = None
        try:
            rows = try_all(service, steps, before=lambda: in_use(frozenset(last)))
        except Stopped as cut:
            rows, stopped = cut.rows, f"stopped at {cut.at}: the router was in use ({cut.why})"
    box = here_of(service.settings.expanded_state_dir()).id  # The baseline's box
    found = [record(r, recipes.get(r.name), ran_on.get(r.name), box) for r in rows]
    return Tried(found, stopped)


def backup_now(service: InstanceService) -> BackupRecord:
    """`mnm backup --auto`, its own notice never sent: a failure is the record's, and the
    night's one notice says it."""
    done = auto_backup(service.settings, gpu_name(service.memory), _unsent)
    check = done.check
    failure = done.failure or (check.failure if check is not None and not check.ok else None)
    return BackupRecord(
        file=done.file.name if done.file is not None else None,
        ok=done.ok,
        restored=check.ok if check is not None else None,
        failure=failure or None,
        lines=[text.strip() for _tone, text in run_words(done)],
    )


def disk_now(service: InstanceService, since: datetime) -> DiskRecord:
    """The policy as `mnm disk check` runs it (its own notices), the report's sizes, and
    every eviction since the night began (a start during the try step runs the policy too)."""
    outcome = disk_check(service)
    report = disk_report(service)
    try:
        recorded = DiskSettings(service.settings.expanded_state_dir()).read().evictions
    except DiskSettingsError:
        recorded = outcome.evicted
    tonight = [e for e in recorded if e.at >= since]
    return DiskRecord(
        held_bytes=report.held_bytes,
        disk_total_bytes=report.disk_total_bytes,
        disk_free_bytes=report.disk_free_bytes,
        unlisted_bytes=report.unlisted_bytes,
        policy=outcome.keep.words,
        gauge=outcome.gauge.subject if outcome.gauge is not None else None,
        warning=outcome.warning,
        evicted=tonight,
        refused=outcome.refused,
        notes=[*outcome.lines, *report.notes],
    )


def clean_old(service: InstanceService, age: timedelta) -> list[CleanRecord]:
    """`mnm clean --auto`: run records with their files, and saved engine logs, older than
    `age`; never a model, a recipe or a workflow's files."""
    settings = service.settings
    cleaner = Cleaner(
        RunStore.for_settings(settings),
        settings.expanded_state_dir(),
        ComfyEngine(settings).clear_history,
    )
    return [
        CleanRecord(target=i.target.value, count=i.count, size_bytes=i.size, note=i.note)
        for i in cleaner.clean(list(AUTO), age)
    ]


def router_healing(settings: Settings, since: datetime) -> list[HealEvent] | None:
    """The healer's lines in the router's journal ring (`/__mnm/events`, its last 200
    lines) since `since`; None when no router answers."""
    events = router_events(settings)
    if events is None:
        return None
    found = (parse(e.line, e.at) for e in events if e.at >= since)
    return [event for event in found if event is not None]


def _unsent(_notice: Notice) -> None:
    return None


def real_steps(service: InstanceService, say: Note, freshness: Freshness) -> Steps:
    return Steps(
        doctor=lambda deep: doctor_checks(service, deep),
        apps=lambda: [app_record(service, app) for app in doctored_apps(service)],
        tried=lambda before: try_every(service, before, say),
        backup=lambda: backup_now(service),
        disk=lambda since: disk_now(service, since),
        clean=lambda age: clean_old(service, age),
        freshness=freshness,
        router=lambda since: router_healing(service.settings, since),
    )
