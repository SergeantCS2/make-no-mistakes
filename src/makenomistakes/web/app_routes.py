"""System → Apps: each workspace app's state, version against this mnm's pin, URL and
where its password or key is, its install, start, stop, update and rollback as a job (one at a
time: they share the apps' lock, and a pull is minutes), and its card (`mnm url --for <app>`).
Uninstalling stays in a terminal (`mnm apps uninstall`): it asks before deleting an app's
data, and a click should not. The connect apps show their cards; Get started writes
their files."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from makenomistakes.apps.card import AppCard
from makenomistakes.apps.cards import card_for, local_manifest
from makenomistakes.apps.context import AppContext
from makenomistakes.apps.hub import AppsService, AppStatus
from makenomistakes.apps.names import LABELS, App
from makenomistakes.hardware import SubprocessRunner
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.jobs import JobKind, JobRegistry
from makenomistakes.web.common import Render

APPS_JOB = "system:apps"
VERBS: dict[str, Callable[[AppsService, App], object]] = {
    "install": lambda s, a: s.install(a),
    "start": lambda s, a: s.start(a),
    "stop": lambda s, a: s.stop(a),
    "update": lambda s, a: s.update(a),
    "rollback": lambda s, a: s.rollback(a),
}


AppsFactory = Callable[[Callable[[str], None]], AppsService]


def apps_service(service: InstanceService) -> AppsFactory:
    """The service a job runs with: Docker and systemd for real, the manifest built here."""

    def build(say: Callable[[str], None]) -> AppsService:
        ctx = AppContext(service.settings, SubprocessRunner(), lambda: local_manifest(service), say)
        return AppsService(ctx)

    return build


@dataclass(frozen=True)
class AppRow:
    status: AppStatus
    card: AppCard | None
    card_error: str | None


def register_app_routes(
    app: FastAPI,
    render: Render,
    service: InstanceService,
    jobs: JobRegistry,
    apps: AppsFactory,
) -> None:
    """`apps(say)` builds the service a job (or a page, with a silent `say`) runs with."""

    def rows() -> list[AppRow]:
        found = apps(lambda _line: None).statuses()
        try:
            manifest = local_manifest(service)
        except InstanceError as exc:
            return [AppRow(s, None, str(exc)) for s in found]
        settings = service.settings
        return [AppRow(s, card_for(s.app, manifest, settings), None) for s in found]

    @app.get("/system/apps", response_class=HTMLResponse)
    def apps_page(request: Request) -> HTMLResponse:
        return render(request, "apps.html", rows=rows(), job=jobs.latest_for(APPS_JOB))

    @app.post("/system/apps/{name}/{verb}")
    def act(name: App, verb: str) -> RedirectResponse:
        work = VERBS.get(verb)
        if work is None:
            raise InstanceError(
                f"Apps have no {verb!r} here: install, start, stop, update, rollback."
            )
        busy = jobs.active_for(APPS_JOB)
        if busy is not None:
            raise InstanceError(f"{busy.title} is still running; wait for it to finish.")

        def run(report: Callable[[str], None]) -> list[str]:
            lines: list[str] = []

            def say(line: str) -> None:
                lines.append(line)
                report(line)

            work(apps(say), name)  # an AppsError fails the job with its own words
            return lines[-40:]

        jobs.start_unless_active(
            JobKind.APPS, APPS_JOB, run, label=f"{verb} {LABELS[name]}", keep_lines=40
        )
        return RedirectResponse("/system/apps", status_code=303)
