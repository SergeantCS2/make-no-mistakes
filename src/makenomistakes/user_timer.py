"""A systemd user timer that runs one `mnm` command every night (the backup, the
nightly), installed the way `mnm service install` installs the web UI's and router's units
(`service.py`: the same folders and ports in its environment, `uv run` in the checkout).

`<name>.service` is a oneshot, `<name>.timer` fires it at the user's hour, `Persistent=true`
so a night the machine slept (WSL shut down) runs at the next start. Installing again rewrites
both files and restarts the timer: the step is idempotent, and a new `--at` takes effect at
once."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from makenomistakes import service as svc
from makenomistakes.config import Settings
from makenomistakes.hardware.probe import CommandRunner

_AT = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")
_CALENDAR = re.compile(r"^OnCalendar=\*-\*-\* (\d\d:\d\d):00$", re.MULTILINE)


class ScheduleError(RuntimeError):
    """Why the timer was not installed or removed, worded for the user."""


@dataclass(frozen=True)
class TimerUnit:
    """One nightly timer: its units' name, what it runs and how the user installs it."""

    name: str  # makenomistakes-backup
    command: str  # after `uv run`: `mnm backup --auto`
    describes: str  # the service's Description
    timer_describes: str  # the timer's
    default_at: str  # 03:30
    install: str  # the command that installs it, without `--at`: `mnm backup schedule`
    fallback: str  # what runs it by hand meanwhile: "`mnm backup --auto` backs up by hand"
    extra: str = ""  # more [Service] lines (a time limit)

    @property
    def service(self) -> str:
        return f"{self.name}.service"

    @property
    def timer(self) -> str:
        return f"{self.name}.timer"


def parse_at(text: str) -> str:
    """`3:30` or `03:30` as `03:30`; anything else is refused."""
    found = _AT.match(text.strip())
    if found is None:
        raise ScheduleError(f"--at takes a time of day like 03:30 (24-hour clock), not {text!r}.")
    return f"{int(found.group(1)):02d}:{found.group(2)}"


def service_text(unit: TimerUnit, root: Path, uv: str, settings: Settings) -> str:
    """The oneshot the timer starts, niced, with mnm's environment."""
    return (
        "[Unit]\n"
        f"Description={unit.describes}\n"
        "\n"
        "[Service]\n"
        "Type=oneshot\n"
        f"WorkingDirectory={svc.specifiers(str(root))}\n"
        f"ExecStart={svc.exec_word(uv)} run {unit.command}\n"
        "Nice=10\n" + unit.extra + svc.environment_lines(settings)
    )


def timer_text(unit: TimerUnit, at: str) -> str:
    """Every day at `at` on the box's clock; a missed night runs at the next start."""
    return (
        "[Unit]\n"
        f"Description={unit.timer_describes}\n"
        "\n"
        "[Timer]\n"
        f"OnCalendar=*-*-* {at}:00\n"
        "Persistent=true\n"
        f"Unit={unit.service}\n"
        "\n"
        "[Install]\n"
        "WantedBy=timers.target\n"
    )


@dataclass(frozen=True)
class TimerState:
    unit: TimerUnit
    available: bool  # `systemctl --user` answers
    installed: bool  # the timer's file exists
    enabled: bool = False
    active: bool = False  # the timer is waiting (not its service running)
    at: str | None = None  # the hour in the installed file
    next_run: str | None = None  # systemd's words for the next elapse
    last_run: str | None = None
    detail: str = ""

    @property
    def line(self) -> str:
        """One line for `mnm service status`, the timer's own command and the web panel."""
        timer, install = self.unit.timer, self.unit.install
        if not self.installed:
            return f"{timer}: not installed (`{install} --at {self.unit.default_at}` installs it)"
        if not self.available:
            return f"{timer}: installed for {self.at}, but systemd --user does not answer"
        if not (self.enabled and self.active):
            return (
                f"{timer}: installed for {self.at} but not running "
                f"(enabled: {self.enabled}, active: {self.active}); `{install}` again starts it"
            )
        last = f"; last run {self.last_run}" if self.last_run else "; not run yet"
        return f"{timer}: every day at {self.at}; next run {self.next_run or 'unknown'}{last}"


def unit_dir(folder: Path | None = None) -> Path:
    return folder or svc.UNIT_DIR  # read now: tests point it at a folder of their own


def installed(unit: TimerUnit, folder: Path | None = None) -> tuple[bool, str | None]:
    """Whether the timer's file is there, and the hour it names (None: unreadable). Reads
    the file only: the doctor and the backup's refusal ask it without systemd."""
    try:
        text = (unit_dir(folder) / unit.timer).read_text(encoding="utf-8")
    except OSError:
        return False, None
    hour = _CALENDAR.search(text)
    return True, hour.group(1) if hour else None


def timer_state(unit: TimerUnit, runner: CommandRunner, folder: Path | None = None) -> TimerState:
    code, out, err = runner.run(["systemctl", "--user", "is-system-running"])
    answer = (out or err).strip()
    available = code in (0, 1) and answer in svc.MANAGER_UP
    there, at = installed(unit, folder)
    if not there:
        return TimerState(unit, available=available, installed=False, detail=answer)
    if not available:
        return TimerState(unit, available=False, installed=True, at=at, detail=answer)
    enabled = runner.run(["systemctl", "--user", "is-enabled", unit.timer])[0] == 0
    active = runner.run(["systemctl", "--user", "is-active", unit.timer])[0] == 0
    shown = runner.run(
        [
            "systemctl",
            "--user",
            "show",
            unit.timer,
            "--property=NextElapseUSecRealtime",
            "--property=LastTriggerUSec",
        ]
    )[1]
    props = dict(line.partition("=")[::2] for line in shown.splitlines() if "=" in line)

    def said(key: str) -> str | None:
        value = props.get(key, "").strip()
        return value if value and value not in ("n/a", "0") else None

    return TimerState(
        unit,
        available=True,
        installed=True,
        enabled=enabled,
        active=active,
        at=at,
        next_run=said("NextElapseUSecRealtime"),
        last_run=said("LastTriggerUSec"),
        detail=answer,
    )


def schedule(
    unit: TimerUnit,
    runner: CommandRunner,
    settings: Settings,
    at: str,
    folder: Path | None = None,
) -> list[str]:
    """Write both units, reload, enable and (re)start the timer; linger on, as `mnm service
    install` does, so it fires with nobody logged in. Returns what ran."""
    where = unit_dir(folder)
    state = svc.systemd_state(runner, unit_dir=where)
    if not state.available:
        fix = svc.systemd_note(state).split(" Until then")[0]  # the service's own fallback
        raise ScheduleError(
            fix.replace("mnm service install", unit.install) + f" Until then {unit.fallback}."
        )
    try:
        root, uv = svc.repo_root(), svc.uv_binary()
    except svc.ServiceError as exc:
        raise ScheduleError(str(exc)) from exc
    where.mkdir(parents=True, exist_ok=True)
    (where / unit.service).write_text(service_text(unit, root, uv, settings), encoding="utf-8")
    (where / unit.timer).write_text(timer_text(unit, at), encoding="utf-8")
    lingered = svc.linger_on(runner)
    ran = _run_all(
        runner,
        [
            ["systemctl", "--user", "daemon-reload"],
            ["systemctl", "--user", "enable", unit.timer],
            # a timer already waiting keeps its old hour until it is restarted
            ["systemctl", "--user", "restart", unit.timer],
        ],
    )
    if lingered is False:
        runner.run(["loginctl", "enable-linger"])  # best effort, as `mnm service install`
        ran.append("loginctl enable-linger")
        if svc.linger_on(runner):
            marker = settings.expanded_state_dir() / svc.LINGER_MARKER
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(f"{unit.install} turned linger on\n", encoding="utf-8")
    return ran


def unschedule(unit: TimerUnit, runner: CommandRunner, folder: Path | None = None) -> list[str]:
    """Stop and remove the timer and its service; nothing when they are not installed."""
    where = unit_dir(folder)
    if not (where / unit.timer).exists() and not (where / unit.service).exists():
        return []
    argv = ["systemctl", "--user", "disable", "--now", unit.timer]
    runner.run(argv)  # a manager that does not answer still gets its files removed
    for name in (unit.timer, unit.service):
        (where / name).unlink(missing_ok=True)
    runner.run(["systemctl", "--user", "daemon-reload"])
    return [
        " ".join(argv),
        f"removed {unit.timer} and {unit.service}",
        "systemctl --user daemon-reload",
    ]


def _run_all(runner: CommandRunner, commands: list[list[str]]) -> list[str]:
    ran = []
    for argv in commands:
        code, out, err = runner.run(argv)
        ran.append(" ".join(argv))
        if code != 0:
            raise ScheduleError(f"{' '.join(argv)} failed: {(err or out).strip()}")
    return ran
