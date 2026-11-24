"""`mnm doctor`'s Apps section: each app `mnm apps` installed, whether it runs and its
version against this mnm's pin. Nothing is asked of an app that is not installed; the router's
side of each one is `mnm doctor --app <app>`."""

from __future__ import annotations

from makenomistakes.apps import exposure
from makenomistakes.apps.hub import AppStatus
from makenomistakes.doctor_types import Check, Status


def app_checks(statuses: list[AppStatus], key_set: bool = True) -> list[Check]:
    """`key_set`: the router's key is in the secrets file; an app open to the network
    without it (a key removed after `mnm apps expose`) is a FAIL."""
    installed = [s for s in statuses if s.installed is not None or s.error]
    if not installed:
        return [
            Check(
                name="Workspace apps",
                status=Status.OK,
                detail="none installed (`mnm apps ls` lists what mnm installs and connects)",
            )
        ]
    return [*(_check(s) for s in installed), *_exposed(installed, key_set)]


def _exposed(rows: list[AppStatus], key_set: bool) -> list[Check]:
    """One line per app `mnm apps expose` opened: what is reachable from the network."""
    found = []
    for row in rows:
        if row.installed is None or not row.installed.exposed:
            continue
        where = f"0.0.0.0:{row.definition.port}"
        close = f"`mnm apps expose {row.app.value} --off` closes it"
        if key_set:
            found.append(
                Check(
                    name=f"Exposed: {row.label}",
                    status=Status.WARN,
                    detail=f"open to your network on {where} over plain http, behind its own "
                    f"login, which crosses the network in clear text; {close}",
                    remedy=exposure.TAILSCALE,
                )
            )
        else:
            found.append(
                Check(
                    name=f"Exposed: {row.label}",
                    status=Status.FAIL,
                    detail=f"open to your network on {where} over plain http, and the router "
                    "has no key: "
                    "the app's users reach every model keyless through it",
                    remedy=f"{close}, or set MNM_API_KEY again (`mnm setup`, section 3)",
                )
            )
    return found


def _check(row: AppStatus) -> Check:
    name = f"App: {row.label}"
    if row.error or row.installed is None:
        return Check(name=name, status=Status.FAIL, detail=row.error or "unreadable", remedy=None)
    version = row.installed.pin.version
    if row.behind:
        return Check(
            name=name,
            status=Status.WARN,
            detail=f"{version} installed; this mnm pins {row.pinned}",
            remedy=f"mnm apps update {row.app.value} (the installed one is kept for rollback)",
        )
    if row.running is False:
        return Check(
            name=name,
            status=Status.WARN,
            detail=f"{version}, stopped",
            remedy=f"mnm apps start {row.app.value}",
        )
    if row.running is None:
        return Check(
            name=name,
            status=Status.WARN,
            detail=f"{version}; whether it runs could not be asked (Docker or systemd)",
            remedy=f"mnm apps status {row.app.value}",
        )
    where = f" at {row.definition.url}" if row.definition.url else ""
    return Check(name=name, status=Status.OK, detail=f"{version}, running{where}")
