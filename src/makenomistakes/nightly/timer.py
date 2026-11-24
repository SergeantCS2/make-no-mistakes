"""`mnm nightly install|status|uninstall`: the systemd user timer that runs `mnm
nightly run` every night (`user_timer.py`: a oneshot service and its timer, 03:00 unless
`--at` says otherwise, `Persistent=true`). It owns the backup schedule: installing it
removes the backup's own `makenomistakes-backup.timer`, since the nightly backs up itself, and `mnm
backup schedule` refuses while it is installed."""

from __future__ import annotations

from pathlib import Path

from makenomistakes import backup_timer, user_timer
from makenomistakes.config import Settings
from makenomistakes.hardware.probe import CommandRunner
from makenomistakes.user_timer import TimerState, TimerUnit

UNIT = TimerUnit(
    name="makenomistakes-nightly",
    command="mnm nightly run",
    describes="makenomistakes nightly self-check (mnm nightly run)",
    timer_describes="makenomistakes nightly self-check",
    default_at="03:00",
    install="mnm nightly install",
    fallback="`mnm nightly run` runs it by hand",
    # the doctor, a try of every instance and a backup: hours on a big box, never forever
    extra="TimeoutStartSec=6h\n",
)


def install(
    runner: CommandRunner, settings: Settings, at: str, unit_dir: Path | None = None
) -> list[str]:
    """The nightly's timer, after the backup's own is removed (the nightly backs up)."""
    ran = user_timer.schedule(UNIT, runner, settings, at, unit_dir)
    gone = backup_timer.unschedule(runner, unit_dir)
    if gone:
        ran += [
            *gone,
            f"{backup_timer.TIMER} removed: the nightly runs the backup now "
            "(`mnm backup schedule` stays for backups without the nightly)",
        ]
    return ran


def uninstall(runner: CommandRunner, unit_dir: Path | None = None) -> list[str]:
    return user_timer.unschedule(UNIT, runner, unit_dir)


def state(runner: CommandRunner, unit_dir: Path | None = None) -> TimerState:
    return user_timer.timer_state(UNIT, runner, unit_dir)


def installed_at(unit_dir: Path | None = None) -> str | None:
    """The hour the installed nightly runs at (its file only; None: not installed)."""
    there, at = user_timer.installed(UNIT, unit_dir)
    return (at or "its hour") if there else None
