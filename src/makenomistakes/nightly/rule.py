"""The regression rule: tonight's report against the earlier nights'. Pure:
reports in, verdicts out; the actions are `actions.py`'s.

- A row's speed against the median of its last `WINDOW` (5) nightly readings on the same
  engine build and settings key: decode tok/s more than `MARGIN` (15%) under it, or a
  picture's s/image or a clip's s/clip more than 15% over it, is a regression. Fewer than
  `NEEDED` (3) earlier readings on that build and settings: "baseline recorded (N of 5)".
- A load or answer that failed (or a row a step broke), only when the same instance, build
  and settings loaded and answered on an earlier night on this box (else a model that never
  fit the card reads "regression" every night): one that never worked is
  `row failed`, told the first night it fails that way and quiet after while it holds.
- A row that could not be tried (`CANNOT_TRY`: no room with everything else parked, not
  downloaded, stopped by the user, held back): `cannot try` with the reason, never a
  regression; the first night it cannot be tried for room is told once, none after while
  the reason holds.
- A doctor line that is FAIL now and was not in the last report that ran the doctor.
- An app whose doctor passed last time and fails now.
- A backup that failed, a restore test that failed.
- The disk policy's evictions: said, never a regression.
- The router's healing since the last night (its journal): a death it restarted is said;
  a restart that failed and a crash loop it parked are regressions.
- Freshness: a newer engine build or model, an upgrade's `-old` still kept: said with
  the command offered, never a regression.

`MNM_NIGHTLY_WINDOW` overrides the window (lowered, a regression shows after two nights);
a user never needs it.
"""

from __future__ import annotations

import os
import statistics
from collections.abc import Callable, Iterable

from makenomistakes.disk_types import gb
from makenomistakes.doctor_machine import NAME as MACHINE
from makenomistakes.doctor_types import Status
from makenomistakes.healing_words import HealEvent, HealKind
from makenomistakes.local_time import local
from makenomistakes.machine import CHANGED
from makenomistakes.measurements import SpeedUnit
from makenomistakes.nightly.report import (
    Action,
    ActionKind,
    AppRecord,
    NightlyReport,
    StepName,
    StepResult,
    StepState,
    TryRecord,
    Verdict,
    VerdictKind,
)
from makenomistakes.readings_filter import Here
from makenomistakes.try_all import Refusal

MARGIN = 0.15
WINDOW = 5
NEEDED = 3
_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
SLOWER_IS_MORE = frozenset({SpeedUnit.IMAGE, SpeedUnit.CLIP})
# Rows not tried, never a fault: no room with everything else parked, or the user's choices
CANNOT_TRY = frozenset({Refusal.NO_FIT, Refusal.STOPPED, Refusal.NOT_DOWNLOADED, Refusal.HELD})
NEVER_WORKED = "it never loaded and answered on this build and settings on an earlier night"


def window() -> int:
    """The readings a baseline is the median of: 5, or `MNM_NIGHTLY_WINDOW`."""
    try:
        found = int(os.environ.get("MNM_NIGHTLY_WINDOW", ""))
    except ValueError:
        return WINDOW
    return found if found >= 1 else WINDOW


def judge(
    report: NightlyReport,
    earlier: list[NightlyReport],
    size: int | None = None,
    here: Here | None = None,
) -> list[Verdict]:
    """Every verdict for tonight, in the page's order: rows, doctor, apps, backup, disk,
    then any step that failed on its own. `earlier` is oldest first; a row's baseline is of
    nights on `here` only (without it, of nights on the row's own box)."""
    size = size or window()
    found: list[Verdict] = []
    if (tried := report.step(StepName.TRY)) is not None:
        for row in tried.rows:
            if (verdict := row_verdict(row, earlier, size, here)) is not None:
                found.append(verdict)
    found += doctor_verdicts(report, earlier)
    found += app_verdicts(report, earlier)
    found += backup_verdicts(report)
    found += disk_verdicts(report)
    found += healing_verdicts(report)
    found += freshness_verdicts(report)
    found += hardware_verdicts(report)
    found += [
        Verdict(
            kind=VerdictKind.STEP,
            subject=f"the {step.name.value} step",
            figures=f"failed: {step.error}",
            regression=True,
        )
        for step in report.steps
        if step.state is StepState.FAILED and step.error is not None
    ]
    return found


def readings_before(
    row: TryRecord, earlier: Iterable[NightlyReport], size: int, here: Here | None = None
) -> list[float]:
    """The row's speeds in earlier nights on its build and settings, the newest `size`, on
    this box (`readings_filter.Here`: a row of an older report names none and is the first
    box's); without `here`, on the row's own box."""
    owns = _owns(row, here)
    speeds = [
        then.speed
        for report in earlier
        if (step := report.step(StepName.TRY)) is not None
        for then in step.rows
        if then.name == row.name
        and then.build == row.build
        and then.settings == row.settings
        and then.unit == row.unit
        and then.speed is not None
        and then.refusal is None
        and owns(then.machine)
    ]
    return speeds[-size:]


def _owns(row: TryRecord, here: Here | None) -> Callable[[str | None], bool]:
    return here.owns if here is not None else (lambda machine: machine == row.machine)


def _rows_before(
    row: TryRecord, earlier: Iterable[NightlyReport], here: Here | None
) -> list[TryRecord]:
    """The instance's rows in earlier nights on this box, oldest first."""
    owns = _owns(row, here)
    return [
        then
        for report in earlier
        if (step := report.step(StepName.TRY)) is not None
        for then in step.rows
        if then.name == row.name and owns(then.machine)
    ]


def worked_before(row: TryRecord, earlier: Iterable[NightlyReport], here: Here | None) -> bool:
    """Whether the instance loaded and answered on an earlier night on its build (when
    tonight's is known: a load that failed never learned it) and settings."""
    return any(
        then.refusal is None
        and then.speed is not None
        and then.settings == row.settings
        and (row.build is None or then.build == row.build)
        for then in _rows_before(row, earlier, here)
    )


def same_as_last(row: TryRecord, earlier: Iterable[NightlyReport], here: Here | None) -> bool:
    """Whether the instance's last earlier row on this box was refused the same way."""
    before = _rows_before(row, earlier, here)
    return bool(before) and before[-1].refusal is row.refusal


def refused_verdict(row: TryRecord, earlier: list[NightlyReport], here: Here | None) -> Verdict:
    """A row not tried, or tried and failed."""
    refusal = row.refusal
    assert refusal is not None
    why = row.why or refusal.value
    common = {"subject": row.name, "build": row.build, "settings": row.settings}
    if refusal in CANNOT_TRY:
        told = refusal is Refusal.NO_FIT and not same_as_last(row, earlier, here)
        return Verdict(
            kind=VerdictKind.CANNOT_TRY, figures=f"cannot try: {why}", told=told, **common
        )
    if worked_before(row, earlier, here):
        return Verdict(kind=VerdictKind.ROW_FAILED, figures=why, regression=True, **common)
    figures = f"{why.rstrip('.')} ({NEVER_WORKED}: not a regression)"
    told = not same_as_last(row, earlier, here)
    return Verdict(kind=VerdictKind.ROW_FAILED, figures=figures, told=told, **common)


def row_verdict(
    row: TryRecord, earlier: list[NightlyReport], size: int, here: Here | None = None
) -> Verdict | None:
    if row.refusal is not None:
        return refused_verdict(row, earlier, here)
    if row.speed is None or row.unit is None or row.build is None or row.settings is None:
        return None
    before = readings_before(row, earlier, size, here)
    now = f"{number(row.speed)} {row.unit.value}"
    build = f"build {row.build}"
    common = {"subject": row.name, "build": row.build, "settings": row.settings}
    if len(before) < min(NEEDED, size):
        seen = len(before) + 1
        return Verdict(
            kind=VerdictKind.BASELINE,
            figures=f"{now} on {build}: baseline recorded ({seen} of {size})",
            **common,
        )
    median = statistics.median(before)
    count = _WORDS.get(len(before), str(len(before)))
    last = "its last nightly reading" if len(before) == 1 else f"its last {count} nightly readings'"
    base = f"{now}, {last}{'' if len(before) == 1 else ' median'} {number(median)} ({build})"
    if _slower(row.speed, median, row.unit):
        change = abs(row.speed / median - 1) * 100 if median else 0.0
        return Verdict(
            kind=VerdictKind.SLOWER,
            figures=f"{base}: {change:.0f}% slower",
            regression=True,
            **common,
        )
    return Verdict(kind=VerdictKind.STEADY, figures=f"{base}: steady", **common)


def _slower(speed: float, median: float, unit: SpeedUnit) -> bool:
    if unit in SLOWER_IS_MORE:  # seconds a picture or a clip: more is slower
        return speed > median * (1 + MARGIN)
    return speed < median * (1 - MARGIN)


def number(value: float) -> str:
    """`72`, `8.4`: the figures a person reads (a tenth below ten)."""
    return f"{value:.0f}" if abs(value) >= 10 else f"{value:.1f}"


def _last_with(earlier: list[NightlyReport], name: StepName) -> StepResult | None:
    """The newest earlier step of that name that ran (not skipped, not a wait)."""
    for report in reversed(earlier):
        step = report.step(name)
        if step is not None and step.state not in (StepState.SKIPPED, StepState.NOT_BUILT):
            return step
    return None


def doctor_verdicts(report: NightlyReport, earlier: list[NightlyReport]) -> list[Verdict]:
    step = report.step(StepName.DOCTOR)
    if step is None:
        return []
    last = _last_with(earlier, StepName.DOCTOR)
    before = {c.name: c.status for c in last.checks} if last is not None else None
    found = []
    for check in step.checks:
        if check.status is not Status.FAIL:
            continue
        was = before.get(check.name) if before is not None else None
        if before is None:
            words, turned = "FAIL (no earlier night to compare)", False
        elif was is Status.FAIL:
            words, turned = "still FAIL", False
        else:
            words = f"FAIL, {was.value.upper()} last night" if was else "FAIL, new tonight"
            turned = True
        found.append(
            Verdict(
                kind=VerdictKind.DOCTOR,
                subject=check.name,
                figures=f"{words}: {check.detail}",
                regression=turned,
            )
        )
    return found


def app_verdicts(report: NightlyReport, earlier: list[NightlyReport]) -> list[Verdict]:
    step = report.step(StepName.APPS)
    if step is None:
        return []
    last = _last_with(earlier, StepName.APPS)
    before = {a.app: a.ok for a in last.apps} if last is not None else {}
    found = []
    for record in step.apps:
        if record.ok:
            continue
        passed = before.get(record.app) is True
        when = "passed last night" if passed else "no pass last night to compare"
        found.append(
            Verdict(
                kind=VerdictKind.APP,
                subject=record.app.value,
                figures=f"{failing(record)} ({when})",
                regression=passed,
            )
        )
    return found


def failing(record: AppRecord) -> str:
    """What fails in an app's doctor, in one line."""
    if record.error is not None:
        return f"its doctor could not run: {record.error}"
    bad = [c for c in record.checks if c.status is Status.FAIL]
    first = bad[0]
    more = f" (and {len(bad) - 1} more)" if len(bad) > 1 else ""
    return f"{first.name} FAIL: {first.detail}{more}"


def backup_verdicts(report: NightlyReport) -> list[Verdict]:
    step = report.step(StepName.BACKUP)
    if step is None or step.backup is None or step.backup.ok:
        return []
    record = step.backup
    if record.restored is False:
        return [
            Verdict(
                kind=VerdictKind.RESTORE,
                subject="the restore test",
                figures=f"{record.file} does not restore: {record.failure}",
                regression=True,
            )
        ]
    return [
        Verdict(
            kind=VerdictKind.BACKUP,
            subject="the backup",
            figures=record.failure or "failed",
            regression=True,
        )
    ]


def disk_verdicts(report: NightlyReport) -> list[Verdict]:
    step = report.step(StepName.DISK)
    if step is None or step.disk is None:
        return []
    return [
        Verdict(
            kind=VerdictKind.EVICTED,
            subject=gone.name,
            figures=(
                f"its files deleted by the disk policy ({gb(gone.freed_bytes)} freed at "
                f"{gone.where}); it downloads them again at its next start"
            ),
        )
        for gone in step.disk.evicted
    ]


def healing_verdicts(report: NightlyReport) -> list[Verdict]:
    """One verdict per death the router handled, by its outcome (the journal's lines in
    order: a death, then its restart's outcome; a crash loop is one line)."""
    step = report.step(StepName.ROUTER)
    if step is None:
        return []
    pending: dict[str, HealEvent] = {}
    found = []
    for event in step.healing:
        when = f"{local(event.at):%H:%M}"
        if event.kind is HealKind.DIED:
            pending[event.name] = event
            continue
        death = pending.pop(event.name, None)
        cause = death.said if death is not None else "its cause is not in the journal's lines"
        died = f"{local(death.at):%H:%M}" if death is not None else when
        if event.kind is HealKind.RESTARTED:
            found.append(_healed(VerdictKind.DIED, event.name, f"died at {died}: {event.said}"))
        elif event.kind is HealKind.NOT_RESTARTED:
            figures = f"died at {died}: {cause}; the restart failed: {event.said}"
            found.append(_healed(VerdictKind.NOT_RESTARTED, event.name, figures, True))
        else:
            figures = f"died twice in ten minutes (the second at {when}): {event.said}"
            found.append(_healed(VerdictKind.CRASH_LOOP, event.name, figures, True))
    for event in pending.values():
        figures = (
            f"died at {local(event.at):%H:%M}: {event.said}; its restart had not ended when "
            "the nightly read the router's journal"
        )
        found.append(_healed(VerdictKind.RESTARTING, event.name, figures))
    return found


def _healed(kind: VerdictKind, name: str, figures: str, regression: bool = False) -> Verdict:
    return Verdict(kind=kind, subject=name, figures=figures, regression=regression)


def freshness_verdicts(report: NightlyReport) -> list[Verdict]:
    """One `newer` verdict per offer (`actions` gives each its command); a check that could
    not run stays in the step's lines."""
    step = report.step(StepName.FRESHNESS)
    if step is None:
        return []
    return [
        Verdict(kind=VerdictKind.NEWER, subject=f.subject, figures=f.figures)
        for f in step.fresh
        if f.offered is not None
    ]


HARDWARE_WORDS = "hardware changed: `mnm machine adopt`"
ADOPT_OFFER = "`mnm machine adopt --dry-run`, then `mnm machine adopt`"


def hardware_verdicts(report: NightlyReport) -> list[Verdict]:
    """One verdict when the doctor step's *Machine* line (which took and recorded the box)
    says the hardware changed since plans were made; never a regression, and never
    run: the user adopts."""
    step = report.step(StepName.DOCTOR)
    if step is None:
        return []
    for check in step.checks:
        if check.name == MACHINE and check.status is Status.WARN and CHANGED in check.detail:
            offer = Action(kind=ActionKind.OFFERED, line=ADOPT_OFFER)
            return [
                Verdict(
                    kind=VerdictKind.HARDWARE, subject=MACHINE, figures=HARDWARE_WORDS, action=offer
                )
            ]
    return []
