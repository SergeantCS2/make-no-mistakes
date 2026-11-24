"""`mnm status`: the box on one screen, from what other commands already read. The
health verdict is `/__mnm/health`'s (the router's own, else read here as the web UI reads it
when the router does not answer); the doctor's worst lines are `mnm doctor`'s checks without
`--deep`; the models by state are `mnm ls`'s rows in a line each; the last nightly is
`mnm nightly ls`'s word. `cli/status_cmd.py` prints these between `mnm ls`'s own lines (the
card, ComfyUI, the queue, the tuner, the box)."""

from __future__ import annotations

from makenomistakes.backup_auto import Said, Tone
from makenomistakes.capacity import Verdict
from makenomistakes.config import Settings
from makenomistakes.consumers import web_origin
from makenomistakes.doctor import Check, Status
from makenomistakes.hardware import SubprocessRunner, probe
from makenomistakes.hardware.box_facts import read_box
from makenomistakes.hardware.probe import CommandRunner
from makenomistakes.health import HealthReport, HealthStatus, RouterPart, WebPart
from makenomistakes.health_now import HealthReader, web_listens
from makenomistakes.instance_types import InstanceStatus
from makenomistakes.machine_checks import machine_checks
from makenomistakes.nightly.report import NightlyReport
from makenomistakes.nightly.words import OUTCOME_TONES
from makenomistakes.router_client import not_answering_note, router_answers, router_health
from makenomistakes.supervisor import ContainerState, Health

SHOWN_CHECKS = 3  # the doctor's worst lines; `mnm doctor` has the rest and the remedies
HEALTH_TONES = {
    HealthStatus.OK: Tone.GOOD,
    HealthStatus.DEGRADED: Tone.WARN,
    HealthStatus.DOWN: Tone.BAD,
}
CHECK_TONES = {Status.OK: Tone.GOOD, Status.WARN: Tone.WARN, Status.FAIL: Tone.BAD}
NO_INSTANCES = "No instances yet. Try: mnm add unsloth/Qwen3-8B-GGUF"


def health_here(settings: Settings, runner: CommandRunner | None = None) -> HealthReport:
    """The router's `/__mnm/health` with the web UI as this machine finds it; when no router
    answers it, the rest read here, `down` (`health_now.for_web`'s rule, from the CLI)."""
    web = WebPart(up=web_listens(settings))
    found = router_health(settings)
    if found is not None:
        return found.model_copy(update={"web": web})
    reader = HealthReader(settings, runner or SubprocessRunner())
    return reader.report(RouterPart(up=router_answers(settings)), web, [])


def doctor_checks(settings: Settings) -> list[Check]:
    """`mnm doctor`'s checks without `--deep` (no smoke test, no bandwidth probe)."""
    report = probe(settings.expanded_models_dir())
    return machine_checks(settings, report, read_box(SubprocessRunner(), report.is_wsl), False)


def health_words(report: HealthReport, settings: Settings, said_last: str | None = None) -> Said:
    """The verdict with its reasons; `said_last` (the box's drift, which the screen's last line
    says) is left out of them, never said twice."""
    reasons = "; ".join(r for r in report.reasons if r != said_last)
    said: Said = [
        (
            HEALTH_TONES[report.status],
            f"Health: {report.status.value}" + (f": {reasons}." if reasons else ""),
        )
    ]
    if not report.router.up:
        said.append((Tone.DIM, f"  {not_answering_note(settings)}"))
    return said


def doctor_words(checks: list[Check]) -> Said:
    """The count, then the worst lines (every FAIL before any WARN), at most `SHOWN_CHECKS`."""
    fails = [c for c in checks if c.status is Status.FAIL]
    warns = [c for c in checks if c.status is Status.WARN]
    if not fails and not warns:
        return [(Tone.GOOD, f"Doctor: all {len(checks)} checks pass.")]
    counts = ", ".join(
        f"{len(found)} {word}" for found, word in ((fails, "fail"), (warns, "warn")) if found
    )
    said: Said = [
        (
            Tone.BAD if fails else Tone.WARN,
            f"Doctor: {counts} of {len(checks)} checks (`mnm doctor` says what to do):",
        )
    ]
    worst = [*fails, *warns]
    said += [
        (CHECK_TONES[c.status], f"  {c.status.value.upper()} {c.name}: {c.detail}")
        for c in worst[:SHOWN_CHECKS]
    ]
    if len(worst) > SHOWN_CHECKS:
        said.append((Tone.DIM, f"  and {len(worst) - SHOWN_CHECKS} more"))
    return said


def instance_words(
    statuses: list[InstanceStatus] | None, verdicts: dict[str, Verdict], problem: str | None = None
) -> Said:
    """Each instance in the group of its state, one line a group, as `mnm ls`'s State column
    reads it: loaded (what it holds), parked (the router loads it on request), stopped by
    itself (the exit code: its log says why), off (the user's `mnm stop`), not on disk. With
    `problem` (a recipe that cannot be read: `statuses` None) that problem is the line."""
    if statuses is None:
        return [(Tone.BAD, f"Models: {problem}")]
    if not statuses:
        return [(Tone.PLAIN, NO_INSTANCES)]
    error = next((s.error for s in statuses if s.error), None)
    if error is not None:  # nothing below is known: every state needs Docker
        return [(Tone.BAD, f"Models: Docker could not be asked: {error}")]
    groups: dict[str, list[str]] = {}
    for status in statuses:
        group, words = _grouped(status, verdicts.get(status.recipe.name))
        groups.setdefault(group, []).append(words)
    return [
        (tone, f"{label}: {', '.join(groups[label])}") for label, tone in GROUPS if label in groups
    ]


GROUPS = (
    ("Loaded", Tone.GOOD),
    ("Parked (loads on request)", Tone.PLAIN),
    ("Stopped by itself", Tone.BAD),
    ("Off (`mnm start` turns it on)", Tone.DIM),
    ("Not on disk (a start downloads it)", Tone.WARN),
)


def _grouped(status: InstanceStatus, verdict: Verdict | None) -> tuple[str, str]:
    name = status.recipe.name
    if status.state is ContainerState.RUNNING:
        if status.health is not Health.READY:
            return GROUPS[0][0], f"{name} ({status.health.value if status.health else 'starting'})"
        holds = verdict.holds if verdict is not None else None
        return GROUPS[0][0], f"{name} ({holds / 2**30:.1f} GiB)" if holds else name
    if status.state is ContainerState.EXITED:
        return GROUPS[2][0], f"{name} (exit {status.exit_code}: `mnm logs {name}`)"
    if not status.files_present:
        return GROUPS[4][0], f"{name} (evicted)" if status.evicted else name
    return (GROUPS[3][0] if status.recipe.off else GROUPS[1][0]), name


def nightly_words(report: NightlyReport | None) -> Said:
    if report is None:
        return [
            (
                Tone.DIM,
                "Nightly: no run yet (`mnm nightly install` runs one every night, "
                "`mnm nightly run --now` one now).",
            )
        ]
    told = len(report.told)
    more = f", {told} to read (`mnm nightly show`)" if told else ""
    return [
        (OUTCOME_TONES[report.outcome], f"Last nightly: {report.id}, {report.outcome.value}{more}.")
    ]


def web_words(settings: Settings, listening: bool) -> Said:
    if listening:
        return [(Tone.PLAIN, f"Web UI: {web_origin(settings.port)} (`mnm open` opens it).")]
    return [
        (
            Tone.WARN,
            f"Web UI: nothing answers on port {settings.port} (`mnm service install` runs it "
            "as a service, or `mnm serve` in a terminal).",
        )
    ]
