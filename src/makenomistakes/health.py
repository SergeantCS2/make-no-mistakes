"""`GET /__mnm/health`: one shape on the router and the web UI, for outside
monitors (Uptime Kuma, Home Assistant). Keyless: it names instances, never a model file, an
address or a key; 200 always, the verdict inside (`status`).

- `ok`: the router answers, nothing below.
- `degraded`: an engine dead (restarting, or parked by a crash loop), a service that restarted
  more than `RESTART_LIMIT` times in the last hour, or the models disk at the disk policy's
  first warning (70% at its default 85%; 70% when the policy is off).
- `down`: only the web UI says it, when the router does not answer (the router cannot say it
  of itself).

`reasons` says why in words (one line each). One reason never degrades the status: that the
hardware changed since saved plans were made, which mnm says until the user plans them
again; the models still answer."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

HEALTH_PATH = "/__mnm/health"
DISK_DEGRADED = 70.0  # percent of the models disk, when the disk policy does not say


class HealthStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"


class RouterPart(BaseModel):
    up: bool
    loaded: list[str] = Field(default_factory=list)
    dead: list[str] = Field(default_factory=list)  # restarting, or parked by a crash loop
    busy: int = 0  # requests in flight


class WebPart(BaseModel):
    up: bool


class ServicePart(BaseModel):
    active: bool
    restarts_last_hour: int = 0


class DiskPart(BaseModel):
    models_pct: int | None = None  # the models folder's filesystem, used of used + free


class NightlyPart(BaseModel):
    last: str  # the latest report's date (`2026-10-26`, `2026-10-26-2`)
    regressions: int  # its verdicts the user was told of


class HealthReport(BaseModel):
    status: HealthStatus
    router: RouterPart
    web: WebPart
    services: dict[str, ServicePart]  # "router", "web"; empty with no user manager
    disk: DiskPart
    nightly: NightlyPart | None
    checked_at: datetime
    reasons: list[str] = Field(default_factory=list)


def judged(  # noqa: PLR0913 - every part of the shape
    router: RouterPart,
    web: WebPart,
    services: dict[str, ServicePart],
    disk: DiskPart,
    nightly: NightlyPart | None,
    checked_at: datetime,
    *,
    looped: list[str] | None = None,
    disk_warn: float = DISK_DEGRADED,
    restart_limit: int,
    machine: str | None = None,
) -> HealthReport:
    """The report with its status and reasons, by the rules above."""
    reasons = []
    if not router.up:
        reasons.append("the router does not answer")
    parked = set(looped or [])
    for name in router.dead:
        if name in parked:
            reasons.append(f"{name} is parked after dying twice in ten minutes")
        else:
            reasons.append(f"{name} died and is being restarted")
    for label, service in services.items():
        if service.restarts_last_hour > restart_limit:
            reasons.append(
                f"the {label} service restarted {service.restarts_last_hour} times in the last hour"
            )
    if disk.models_pct is not None and disk.models_pct >= disk_warn:
        reasons.append(f"the models disk is at {disk.models_pct}%")
    status = HealthStatus.OK
    if not router.up:
        status = HealthStatus.DOWN
    elif reasons:
        status = HealthStatus.DEGRADED
    if machine is not None:  # Said, never a degradation
        reasons.append(machine)
    return HealthReport(
        status=status,
        router=router,
        web=web,
        services=services,
        disk=disk,
        nightly=nightly,
        checked_at=checked_at,
        reasons=reasons,
    )
