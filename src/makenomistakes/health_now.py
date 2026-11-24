"""What `/__mnm/health` reads on this box: the services from systemd, the models disk,
the latest nightly, each read at most every `CACHE_SECONDS` (a monitor polls every minute or
so; nothing here may cost a request more than a look). The router adds its own part from
memory (`for_router`); the web UI asks the router and answers for it when it is down
(`for_web`)."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from makenomistakes import disk_settings
from makenomistakes.config import Settings
from makenomistakes.disk_settings import DiskSettings, DiskSettingsError, gauge_for
from makenomistakes.disk_types import FIRST_WARNING, KeepMode
from makenomistakes.hardware.probe import CommandRunner
from makenomistakes.health import (
    DISK_DEGRADED,
    DiskPart,
    HealthReport,
    NightlyPart,
    RouterPart,
    ServicePart,
    WebPart,
    judged,
)
from makenomistakes.local_time import local
from makenomistakes.machine import drift_now, saved_recipes
from makenomistakes.nightly.report import NightlyStore
from makenomistakes.service_watch import RESTART_LIMIT, every_unit

if TYPE_CHECKING:
    from makenomistakes.router import Router

CACHE_SECONDS = 60.0  # the parts behind a poll read once a minute
WEB_PROBE_SECONDS = 0.5


@dataclass
class _Box:
    """The slow parts, as last read."""

    services: dict[str, ServicePart]
    disk: DiskPart
    disk_warn: float
    nightly: NightlyPart | None
    machine: str | None = None  # The hardware changed since plans were made


@dataclass
class HealthReader:
    settings: Settings
    runner: CommandRunner
    clock: Callable[[], float] = time.monotonic
    now: Callable[[], datetime] = local
    _cached: tuple[float, _Box] | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def box(self) -> _Box:
        with self._lock:
            if self._cached is not None and self.clock() - self._cached[0] < CACHE_SECONDS:
                return self._cached[1]
        found = _Box(self._services(), *self._disk(), self._nightly(), self._machine())
        with self._lock:
            self._cached = (self.clock(), found)
        return found

    def _services(self) -> dict[str, ServicePart]:
        return {
            unit.label: ServicePart(active=unit.active, restarts_last_hour=unit.last_hour)
            for unit in every_unit(self.runner)
            if unit.managed
        }

    def _disk(self) -> tuple[DiskPart, float]:
        """The models disk's use, and the share at which it reads degraded: the disk
        policy's first warning when it measures the disk, else `DISK_DEGRADED`."""
        models = self.settings.expanded_models_dir()
        try:
            total, free = disk_settings.measure()(models)  # read now: tests fake the disk
        except OSError:
            return DiskPart(), DISK_DEGRADED
        pct = round(100 * (total - free) / total) if total else None
        try:
            keep = DiskSettings(self.settings.expanded_state_dir()).read().keep
        except DiskSettingsError:
            return DiskPart(models_pct=pct), DISK_DEGRADED
        if keep.on and keep.mode is KeepMode.PERCENT and gauge_for(keep, models) is not None:
            return DiskPart(models_pct=pct), max(keep.percent - FIRST_WARNING, 0.0)
        return DiskPart(models_pct=pct), DISK_DEGRADED

    def _nightly(self) -> NightlyPart | None:
        latest = NightlyStore(self.settings.expanded_state_dir()).latest()
        if latest is None:
            return None
        return NightlyPart(last=latest.id, regressions=len(latest.told))

    def _machine(self) -> str | None:
        """From `machine.json` as the router's start recorded it and the saved recipes."""
        found = drift_now(self.settings, saved_recipes(self.settings))
        return found.words() if found is not None else None

    def report(self, router: RouterPart, web: WebPart, looped: list[str]) -> HealthReport:
        box = self.box()
        return judged(
            router,
            web,
            box.services,
            box.disk,
            box.nightly,
            self.now(),
            looped=looped,
            disk_warn=box.disk_warn,
            restart_limit=RESTART_LIMIT,
            machine=box.machine,
        )


def web_listens(settings: Settings) -> bool:
    """Something listens on the web UI's port on this machine (its unit binds 127.0.0.1)."""
    try:
        with socket.create_connection(("127.0.0.1", settings.port), WEB_PROBE_SECONDS):
            return True
    except OSError:
        return False


def for_router(
    router: Router, reader: HealthReader, web_up: Callable[[], bool] | None = None
) -> HealthReport:
    """The router's answer, from its own memory: what it counts as loaded (the healer's), the
    dead ones, the requests in flight; never a Docker call."""
    healer = router.healer
    part = RouterPart(
        up=True,
        loaded=healer.loaded() if healer is not None else [],
        dead=router.dead.names(),
        busy=sum(router.in_flight().values()),
    )
    web = WebPart(up=(web_up or (lambda: web_listens(reader.settings)))())
    return reader.report(part, web, router.dead.looped())


def for_web(
    reader: HealthReader,
    asked: Callable[[], HealthReport | None],
    answers: Callable[[], bool],
) -> HealthReport:
    """The web UI's answer: the router's own, with the web UI up (it is answering); else the
    rest read here, `down` when the router does not answer at all (`answers`: an older router
    without the route still answers its `/health`)."""
    found = asked()
    if found is not None:
        return found.model_copy(update={"web": WebPart(up=True)})
    return reader.report(RouterPart(up=answers()), WebPart(up=True), [])
