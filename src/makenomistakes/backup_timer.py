"""`mnm backup schedule`: a systemd user timer that runs `mnm backup --auto` every
night (`user_timer.py`: a oneshot service and its timer, 03:30 unless `--at` says otherwise).

`mnm nightly` calls the same `--auto` path (`backup_auto.auto_backup`) and owns the
schedule: `mnm nightly install` removes this timer, and `mnm backup schedule` refuses while
the nightly's is installed, so one night never backs up twice.
"""

from __future__ import annotations

from pathlib import Path

from makenomistakes import user_timer
from makenomistakes.config import Settings
from makenomistakes.hardware.probe import CommandRunner
from makenomistakes.user_timer import ScheduleError, TimerState, TimerUnit, parse_at

__all__ = ["ScheduleError", "TimerState", "parse_at"]

UNIT = TimerUnit(
    name="makenomistakes-backup",
    command="mnm backup --auto",
    describes="makenomistakes backup (mnm backup --auto)",
    timer_describes="makenomistakes nightly backup",
    default_at="03:30",
    install="mnm backup schedule",
    fallback="`mnm backup --auto` backs up by hand",
)
NAME = UNIT.name
SERVICE = UNIT.service
TIMER = UNIT.timer
DEFAULT_AT = UNIT.default_at


def service_text(root: Path, uv: str, settings: Settings) -> str:
    return user_timer.service_text(UNIT, root, uv, settings)


def timer_text(at: str) -> str:
    return user_timer.timer_text(UNIT, at)


def timer_state(runner: CommandRunner, unit_dir: Path | None = None) -> TimerState:
    return user_timer.timer_state(UNIT, runner, unit_dir)


def schedule(
    runner: CommandRunner, settings: Settings, at: str, unit_dir: Path | None = None
) -> list[str]:
    return user_timer.schedule(UNIT, runner, settings, at, unit_dir)


def unschedule(runner: CommandRunner, unit_dir: Path | None = None) -> list[str]:
    return user_timer.unschedule(UNIT, runner, unit_dir)
