"""How a nightly report reads: `mnm nightly show`'s page and `mnm nightly ls`'s lines,
verdicts first, then each step. The web's System → Nightly renders the same report from its
template; the words for a state and a verdict are here once."""

from __future__ import annotations

from makenomistakes.backup_auto import Said, Tone
from makenomistakes.disk_types import gb
from makenomistakes.doctor_types import Status
from makenomistakes.local_time import local
from makenomistakes.nightly.freshness import FindingKind
from makenomistakes.nightly.report import (
    AppRecord,
    Listed,
    NightlyReport,
    Outcome,
    StepResult,
    StepState,
    TryRecord,
    Verdict,
    VerdictKind,
)
from makenomistakes.nightly.rule import failing

OUTCOME_TONES = {
    Outcome.QUIET: Tone.GOOD,
    Outcome.REGRESSION: Tone.BAD,
    Outcome.TO_READ: Tone.WARN,
    Outcome.FAILED: Tone.BAD,
    Outcome.WAITED: Tone.WARN,
    Outcome.RUNNING: Tone.DIM,
}
STATE_TONES = {
    StepState.OK: Tone.GOOD,
    StepState.WARN: Tone.WARN,
    StepState.FAILED: Tone.BAD,
    StepState.SKIPPED: Tone.DIM,
    StepState.NOT_BUILT: Tone.DIM,
}


def verdict_tone(verdict: Verdict) -> Tone:
    if verdict.regression or verdict.told:
        return Tone.BAD
    if verdict.kind is VerdictKind.STEADY:
        return Tone.GOOD
    if verdict.kind in (VerdictKind.BASELINE, VerdictKind.EVICTED, VerdictKind.NEWER):
        return Tone.PLAIN
    return Tone.WARN  # a doctor line still failing, an app with no pass to compare


def heading(report: NightlyReport) -> str:
    began = local(report.started_at)
    span = f"{began:%Y-%m-%d %H:%M}"
    if report.finished_at is not None:
        took = (report.finished_at - report.started_at).total_seconds() / 60
        span += f" to {local(report.finished_at):%H:%M}, {took:.0f} min"
    return f"The nightly {report.id} ({span}): {report.outcome.value}"


def page(report: NightlyReport) -> Said:
    """`mnm nightly show`: the verdicts with their actions, then every step."""
    said: Said = [(OUTCOME_TONES[report.outcome], heading(report))]
    if report.waited:
        said.append((Tone.WARN, report.waited))
    if report.verdicts:
        said.append((Tone.PLAIN, ""))
        said.append((Tone.PLAIN, "Verdicts:"))
        for verdict in report.verdicts:
            said.append((verdict_tone(verdict), f"  {verdict.line}"))
            if (action := verdict.action) is not None:
                how = "done" if action.automatic else "offered"
                tone = Tone.BAD if action.ok is False else Tone.DIM
                said.append((tone, f"    {how}: {action.line}"))
    if report.steps:
        said.append((Tone.PLAIN, ""))
        said.append((Tone.PLAIN, "Steps:"))
        for step in report.steps:
            said += step_lines(step)
    if report.notice:
        said.append((Tone.PLAIN, ""))
        said += [(Tone.DIM, f"Notice: {line}") for line in report.notice]
    return said


def step_lines(step: StepResult) -> Said:
    head = f"  {step.name.value:<9} {step.state.value:<9} {step.seconds:>6.0f} s  "
    said: Said = [(STATE_TONES[step.state], head + (step.error or step.summary))]
    for check in step.checks:
        if check.status is not Status.OK:
            tone = Tone.BAD if check.status is Status.FAIL else Tone.WARN
            said.append((tone, f"      {check.status.value.upper()} {check.name}: {check.detail}"))
    for app in step.apps:
        said.append((Tone.GOOD if app.ok else Tone.BAD, f"      {app.app.value}: {app_word(app)}"))
    for row in step.rows:
        said.append((Tone.PLAIN if row.why is None else Tone.WARN, f"      {row_line(row)}"))
    if step.backup is not None:
        said += [(Tone.DIM, f"      {line}") for line in step.backup.lines[1:]]
    if step.disk is not None:
        disk = step.disk
        if disk.held_bytes is not None:
            free = gb(disk.disk_free_bytes or 0)
            said.append((Tone.DIM, f"      models folder {gb(disk.held_bytes)}, {free} free"))
        said += [(Tone.WARN, f"      {line}") for line in (disk.refused, *disk.notes) if line]
    said += [(Tone.DIM, f"      {c.target}: {c.count}, {gb(c.size_bytes)}") for c in step.cleaned]
    if step.fresh:  # An offer is plain, a check that could not run a warning
        said += [
            (Tone.WARN if f.kind is FindingKind.UNCHECKED else Tone.PLAIN, f"      {f.line}")
            for f in step.fresh
        ]
    else:
        said += [(Tone.WARN, f"      {line}") for line in step.findings]
    said += [
        (Tone.WARN, f"      {local(e.at):%H:%M} {e.name} {e.kind.value}: {e.said}")
        for e in step.healing
    ]
    return said


def app_word(app: AppRecord) -> str:
    return "passes" if app.ok else failing(app)


def row_line(row: TryRecord) -> str:
    if row.why is not None:
        return f"{row.name}: not tried: {row.why}"
    load = f"load {row.load_seconds:.1f} s" if row.load_seconds is not None else "load -"
    build = f"build {row.build}" if row.build else "build unknown"
    return f"{row.name}: {row.shown or '-'}, {load}, {build}"


def listing_line(found: Listed) -> tuple[Tone, str]:
    """`mnm nightly ls`: the date and one word."""
    if found.report is None:
        return Tone.DIM, f"{found.id}  ({found.problem})"
    report = found.report
    told = len(report.told)
    more = f"  {told} to read" if told else ""
    return OUTCOME_TONES[report.outcome], f"{found.id}  {report.outcome.value}{more}"
