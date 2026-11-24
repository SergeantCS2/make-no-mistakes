"""What the nightly does about its verdicts: each verdict gets the action mnm took or the
command it offers, one line on the page, and each action taken one journal line.

Automatic: a tune queued in the router's tuner when a chat-like model's decode regressed on
the same build (`POST /__mnm/tune/<name>`, the instance type's default mode; the router's
tuner decides whether it tunes it; never twice for the same regression: same instance,
build and settings key), an installed app restarted once when its doctor fails (then its
doctor again; in the notice either way), a failed backup run once more at the end. A dead
engine restarted and a crash loop parked are the router's own actions, listed here as
the journal tells them, never done again by the nightly. Never
automatic, offered as a command: an engine or model update (from the freshness step), a
look at the log. An action that fails is a line, never an exception out of the run."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from makenomistakes.apps.names import App
from makenomistakes.doctor_types import Check
from makenomistakes.journal import Note
from makenomistakes.measurements import SpeedUnit
from makenomistakes.nightly.freshness import FindingKind
from makenomistakes.nightly.report import (
    Action,
    ActionKind,
    AppRecord,
    BackupRecord,
    NightlyReport,
    StepName,
    TryRecord,
    Verdict,
    VerdictKind,
)
from makenomistakes.nightly.rule import failing


@dataclass(frozen=True)
class Doers:
    retune: Callable[[str], tuple[int, str] | None]  # the router's answer; None: no router
    backup: Callable[[], BackupRecord]
    restart: Callable[[App], AppRecord]
    managed: Callable[[], frozenset[App]]


def act(report: NightlyReport, earlier: list[NightlyReport], doers: Doers, say: Note) -> None:
    """Give each verdict its action; the backup again last, once."""
    backup_again = None
    for verdict in report.verdicts:
        try:
            match verdict.kind:
                case VerdictKind.SLOWER:
                    verdict.action = _slower(verdict, _row(report, verdict.subject), earlier, doers)
                case VerdictKind.ROW_FAILED | VerdictKind.CANNOT_TRY:
                    verdict.action = row_offer(verdict)
                case VerdictKind.DOCTOR if verdict.regression:
                    check = _check(report, verdict.subject)
                    fix = check.remedy if check is not None and check.remedy else "`mnm doctor`"
                    verdict.action = _offer(fix)
                case VerdictKind.APP:
                    verdict.action = _app(verdict, doers)
                case VerdictKind.BACKUP | VerdictKind.RESTORE:
                    backup_again = backup_again or verdict
                case VerdictKind.NEWER:
                    verdict.action = fresh_offer(report, verdict)
                case (
                    VerdictKind.DIED
                    | VerdictKind.RESTARTING
                    | VerdictKind.NOT_RESTARTED
                    | VerdictKind.CRASH_LOOP
                ):
                    verdict.action = healed(verdict)
                case _:
                    continue
        except Exception as exc:  # an action that fails is a line, never the night's end
            verdict.action = Action(kind=ActionKind.OFFERED, line=f"the action failed: {exc}")
        if verdict.action is not None and verdict.action.automatic:
            say(f"{verdict.subject}: {verdict.action.line}")
    if backup_again is not None:
        backup_again.action = _backup_again(doers)
        say(f"{backup_again.subject}: {backup_again.action.line}")


def row_offer(verdict: Verdict) -> Action | None:
    """A row that failed: its log; one that could not be tried for room, the first night
    a re-plan for the card as it is now; else nothing."""
    name = verdict.subject
    if verdict.kind is VerdictKind.ROW_FAILED:
        return _offer(f"`mnm logs {name}` says why; `mnm start {name}` tries it again")
    if not verdict.told:
        return None
    return _offer(
        f"`mnm replan {name}` plans it for the card as it is now; the nightly tries it again "
        "each night and says nothing more while it cannot"
    )


def healed(verdict: Verdict) -> Action:
    """What the router did about a death: its restart, or its park of a crash loop."""
    name = verdict.subject
    match verdict.kind:
        case VerdictKind.DIED:
            return Action(kind=ActionKind.ENGINE_RESTART, line="restarted by the router", ok=True)
        case VerdictKind.RESTARTING:
            return Action(kind=ActionKind.ENGINE_RESTART, line="the router was restarting it")
        case VerdictKind.NOT_RESTARTED:
            return Action(
                kind=ActionKind.ENGINE_RESTART,
                line=f"the router's restart failed: `mnm logs {name} --review`, then "
                f"`mnm start {name}`",
                ok=False,
            )
        case _:
            return Action(
                kind=ActionKind.PARKED,
                line=f"parked after two deaths; `mnm logs {name} --review`",
                ok=True,
            )


def fresh_offer(report: NightlyReport, verdict: Verdict) -> Action | None:
    """The command a freshness finding offers, with what it does (never run here)."""
    step = report.step(StepName.FRESHNESS)
    found = next(
        (f for f in (step.fresh if step else []) if f.subject == verdict.subject and f.offered),
        None,
    )
    if found is None or found.offered is None:
        return None
    return _offer(f"`{found.offered}` {FRESH_WORDS[found.kind]}")


FRESH_WORDS = {
    FindingKind.ENGINE: (
        "pulls it, tests it on a copy of the smallest instance and rolls back by itself when "
        "the test fails"
    ),
    FindingKind.MODEL: "shows the plan; `--yes` adds it beside, the old one kept as <name>-old",
    FindingKind.KEPT: "removes it and the files no other instance uses",
    FindingKind.UNCHECKED: "",
}


def _offer(line: str) -> Action:
    return Action(kind=ActionKind.OFFERED, line=line)


def _row(report: NightlyReport, name: str) -> TryRecord | None:
    step = report.step(StepName.TRY)
    return next((r for r in step.rows if r.name == name), None) if step is not None else None


def _check(report: NightlyReport, name: str) -> Check | None:
    step = report.step(StepName.DOCTOR)
    return next((c for c in step.checks if c.name == name), None) if step is not None else None


def queued_before(verdict: Verdict, earlier: list[NightlyReport]) -> str | None:
    """The night a tune was queued for this same regression (instance, build, settings)."""
    for report in reversed(earlier):
        for then in report.verdicts:
            if (
                then.kind is VerdictKind.SLOWER
                and then.subject == verdict.subject
                and then.build == verdict.build
                and then.settings == verdict.settings
                and then.action is not None
                and then.action.kind is ActionKind.RETUNE
                and then.action.ok
            ):
                return report.id
    return None


def _slower(
    verdict: Verdict, row: TryRecord | None, earlier: list[NightlyReport], doers: Doers
) -> Action:
    name = verdict.subject
    engine = row.engine.value if row is not None and row.engine is not None else "<engine>"
    update = f"`mnm engines update {engine}` if a newer build is out (never run by the nightly)"
    if row is None or row.unit is not SpeedUnit.TOKENS:
        return _offer(f"`mnm try {name} --speed` again by hand; {update}")
    if (night := queued_before(verdict, earlier)) is not None:
        return _offer(
            f"a tune was queued for this on {night} (never twice): `mnm tune --status`; {update}"
        )
    answer = doers.retune(name)
    if answer is None:
        return _offer(f"`mnm tune {name}` (no router answered to queue it)")
    status, words = answer
    if status == 202:
        return Action(kind=ActionKind.RETUNE, line=words, ok=True)  # "Queued tiny: …"
    return _offer(f"the router's tuner did not take it ({status}: {words}); {update}")


def _app(verdict: Verdict, doers: Doers) -> Action:
    app = App(verdict.subject)
    if app not in doers.managed():
        return _offer(f"`mnm doctor --app {app.value}` says what fails; mnm did not install it")
    verdict.told = True  # restarted (or tried to): the user hears of it either way
    after = doers.restart(app)
    if after.ok:
        return Action(kind=ActionKind.APP_RESTART, line="restarted: its doctor passes now", ok=True)
    return Action(
        kind=ActionKind.APP_RESTART,
        line=f"restarted, and its doctor still fails: {failing(after)}",
        ok=False,
    )


def _backup_again(doers: Doers) -> Action:
    try:
        again = doers.backup()
    except Exception as exc:
        return Action(kind=ActionKind.BACKUP_AGAIN, line=f"tried again and failed: {exc}", ok=False)
    if again.ok:
        line = f"tried again at the end of the night: wrote {again.file}, it restores"
        return Action(kind=ActionKind.BACKUP_AGAIN, line=line, ok=True)
    return Action(
        kind=ActionKind.BACKUP_AGAIN,
        line=f"tried again at the end of the night and failed: {again.failure}",
        ok=False,
    )
