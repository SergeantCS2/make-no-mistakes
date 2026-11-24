"""The services' watchdog: both units already restart by themselves
(`Restart=`); this reads how often they did. `systemctl --user show` gives a unit's state and
its restart count since it was last started by hand (`NRestarts`); the user journal's
"Scheduled restart job" lines give the times, so "in the last hour" is counted from them.

`mnm service status` prints the count when there is one, the doctor's *Services* line WARNs
above `RESTART_LIMIT` an hour (the nightly's doctor step carries it), the router notices it at
most once an hour per unit (a stamp file, so a router that keeps restarting does not notice
each time), and the router's own start says when systemd restarted it. With no user manager
(`mnm serve` in a terminal, a container) nothing is read and nothing is said."""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from makenomistakes import service
from makenomistakes.doctor_types import Check, Status
from makenomistakes.hardware.probe import CommandRunner
from makenomistakes.local_time import local
from makenomistakes.notify import Notice
from makenomistakes.service import ROUTER_UNIT, UNIT_NAME

RESTART_LIMIT = 3  # more restarts than this in an hour: the doctor WARNs, a notice goes
HOUR = timedelta(hours=1)
RESTARTED = "Scheduled restart job"  # systemd's line for each automatic restart
NOTICE_FOLDER = "watchdog"  # <state>/watchdog/<unit>.noticed: when the last notice went
LABELS = {ROUTER_UNIT: "router", UNIT_NAME: "web"}


@dataclass(frozen=True)
class UnitRestarts:
    """One unit as systemd says: installed and running, how often it restarted by itself."""

    unit: str
    managed: bool  # the user manager answered about it
    active: bool = False
    total: int = 0  # `NRestarts`: since it was last started by hand
    last_hour: int = 0
    last_at: datetime | None = None  # the newest restart in the last hour

    @property
    def label(self) -> str:
        return LABELS.get(self.unit, self.unit)

    @property
    def over(self) -> bool:
        return self.last_hour > RESTART_LIMIT

    def words(self) -> str | None:
        """ "restarted 4 times in the last hour (last at 03:14)", or None when it did not."""
        if self.last_hour == 0:
            return None
        times = "once" if self.last_hour == 1 else f"{self.last_hour} times"
        at = f" (last at {local(self.last_at):%H:%M})" if self.last_at is not None else ""
        return f"restarted {times} in the last hour{at}"


class _Entry(BaseModel):
    """One `journalctl -o json` line: its time (microseconds since the epoch) and message
    (a byte array when it is not UTF-8: such a line is not systemd's)."""

    stamp: str = Field(default="", alias="__REALTIME_TIMESTAMP")
    message: str | list[int] | None = Field(default=None, alias="MESSAGE")


def _show(runner: CommandRunner, unit: str) -> dict[str, str] | None:
    asked = ["NRestarts", "LoadState", "ActiveState"]
    argv = ["systemctl", "--user", "show", *(w for p in asked for w in ("-p", p)), unit]
    code, out, _err = runner.run(argv)
    if code != 0:
        return None
    found = dict(line.partition("=")[::2] for line in out.splitlines() if "=" in line)
    return found if found.get("LoadState") not in (None, "", "not-found") else None


def restart_times(runner: CommandRunner, unit: str, since: str = "-1h") -> list[datetime]:
    """When systemd restarted the unit by itself, from the user journal; [] when it cannot
    be read."""
    argv = ["journalctl", "--user", "-u", unit, "--since", since, "-o", "json", "--no-pager"]
    code, out, _err = runner.run(argv)
    if code != 0:
        return []
    found = []
    for line in out.splitlines():
        try:
            entry = _Entry.model_validate_json(line)
        except ValidationError:
            continue
        if isinstance(entry.message, str) and RESTARTED in entry.message and entry.stamp.isdigit():
            found.append(datetime.fromtimestamp(int(entry.stamp) / 1e6, UTC))
    return found


def unit_restarts(
    runner: CommandRunner, unit: str, now: Callable[[], datetime] = lambda: datetime.now(UTC)
) -> UnitRestarts:
    shown = _show(runner, unit)
    if shown is None:
        return UnitRestarts(unit=unit, managed=False)
    total = int(shown["NRestarts"]) if shown.get("NRestarts", "").isdigit() else 0
    recent = [t for t in restart_times(runner, unit) if now() - t <= HOUR] if total else []
    return UnitRestarts(
        unit=unit,
        managed=True,
        active=shown.get("ActiveState") == "active",
        total=total,
        last_hour=len(recent),
        last_at=max(recent) if recent else None,
    )


def every_unit(runner: CommandRunner) -> list[UnitRestarts]:
    """The router's unit, then the web UI's."""
    return [unit_restarts(runner, unit) for unit in (ROUTER_UNIT, UNIT_NAME)]


def logs_remedy(unit: str) -> str:
    return (
        f"`mnm service logs`, or `journalctl --user -u {unit} --since -1h` says how each run ended"
    )


def services_check(runner: CommandRunner, unit_dir: Path | None = None) -> Check:
    """The doctor's *Services* line: each installed unit running, and how often it restarted
    in the last hour; WARN above `RESTART_LIMIT`."""
    folder = unit_dir or service.UNIT_DIR  # read now: tests point it elsewhere
    installed = [u for u in (ROUTER_UNIT, UNIT_NAME) if (folder / f"{u}.service").exists()]
    units = [unit_restarts(runner, unit) for unit in installed]  # systemd asked only then
    if not units:
        return Check(
            name="Services",
            status=Status.OK,
            detail="not installed",
            remedy="`mnm service install` runs the web UI and the router as services",
        )
    if not any(u.managed for u in units):
        return Check(
            name="Services", status=Status.OK, detail="systemd's user manager does not answer"
        )
    parts = []
    for u in units:
        state = "running" if u.active else "not running"
        said = u.words()
        limit = f", more than {RESTART_LIMIT} an hour" if u.over else ""
        parts.append(f"{u.label} {state}" + (f", {said}{limit}" if said else ""))
    over = [u for u in units if u.over]
    if over:
        worst = over[0]
        return Check(
            name="Services",
            status=Status.WARN,
            detail="; ".join(parts),
            remedy=logs_remedy(worst.unit),
        )
    down = [u for u in units if u.managed and not u.active]
    if down:
        return Check(
            name="Services",
            status=Status.WARN,
            detail="; ".join(parts),
            remedy=f"`systemctl --user start {down[0].unit}`, or `mnm service install`",
        )
    return Check(name="Services", status=Status.OK, detail="; ".join(parts))


@dataclass
class Watchdog:
    """The router's look at both units, at most every `every` seconds (the sweep runs every
    30); a notice per unit at most once an hour while it stays above the limit."""

    runner: CommandRunner
    state_dir: Path
    notify: Callable[[Notice], None]
    every: float = 300.0
    clock: Callable[[], float] = time.monotonic
    wall: Callable[[], float] = time.time
    _last: float | None = field(default=None, init=False)  # when it last looked

    def tick(self) -> list[UnitRestarts]:
        """The units looked at, or [] when it was not yet time."""
        now = self.clock()
        if self._last is not None and now - self._last < self.every:
            return []
        self._last = now
        units = every_unit(self.runner)
        for unit in units:
            if unit.over and self._due(unit.unit):
                said = unit.words() or ""
                self.notify(
                    Notice(
                        f"mnm: the {unit.label} service keeps restarting",
                        f"{unit.unit} {said}: {logs_remedy(unit.unit)}.",
                        failed=True,
                    )
                )
        return units

    def _due(self, unit: str) -> bool:
        """Whether an hour passed since this unit's last notice; the stamp moved if so."""
        stamp = self.state_dir / NOTICE_FOLDER / f"{unit}.noticed"
        try:
            if self.wall() - stamp.stat().st_mtime < HOUR.total_seconds():
                return False
        except OSError:
            pass
        with contextlib.suppress(OSError):
            stamp.parent.mkdir(parents=True, exist_ok=True)
            stamp.touch()
            os.utime(stamp, (self.wall(), self.wall()))
        return True


def started_by_systemd(runner: CommandRunner, environ: Mapping[str, str]) -> str | None:
    """The router's first journal line when systemd restarted it (its unit's `NRestarts`
    above 0, read once at start); None in a terminal (no `INVOCATION_ID`) or with no
    manager to ask."""
    if not environ.get("INVOCATION_ID"):
        return None
    shown = _show(runner, ROUTER_UNIT)
    count = shown.get("NRestarts", "") if shown is not None else ""
    if not count.isdigit() or int(count) == 0:
        return None
    times = "once" if count == "1" else f"{count} times"
    return (
        f"Restarted by systemd ({times} since the unit was last started by hand): the journal "
        f"before this line says how the last run ended (`journalctl --user -u {ROUTER_UNIT} "
        "--since -1h`)."
    )
