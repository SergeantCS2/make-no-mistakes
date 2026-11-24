"""Get started's Workspace apps, what `mnm setup`'s section 8 offers: one paragraph per
app, its state, and the same steps as jobs. Installing posts to System → Apps' own install
(`/system/apps/<app>/install`, the job and its streamed lines are shown there); Goose's file,
Odysseus's connect and LibreChat's first account are jobs here, through `apps/setup_ops`, under
System → Apps' job key (one apps job at a time: they share the apps' lock). Odysseus's `.env`
diff is a page of its own, read again when the button is pressed (never taken from the form).
The projects folder is shown, not changed: it is `mnm setup`'s question (the units carry it)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from makenomistakes.apps.hub import AppStatus
from makenomistakes.apps.names import LABELS, App
from makenomistakes.apps.offer import Offered, offered
from makenomistakes.apps.projects import projects_problem
from makenomistakes.apps.setup_ops import AppsSetup
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.jobs import JobKind, JobRegistry
from makenomistakes.web.app_routes import APPS_JOB, AppsFactory
from makenomistakes.web.common import Render


@dataclass(frozen=True)
class AppOffer:
    offer: Offered
    paragraph: str
    status: AppStatus
    remote: list[str]


@dataclass(frozen=True)
class AppsPanel:
    offers: list[AppOffer]
    projects: str
    odysseus: str | None  # its folder when there
    error: str | None  # the router's models could not be read (the cards wait)
    projects_problem: str | None = None  # why the projects folder is refused


def apps_panel(service: InstanceService, ops: AppsSetup, apps: AppsFactory) -> AppsPanel:
    """What Get started shows of the apps (the starter page renders it)."""
    settings = service.settings
    folder = ops.odysseus()
    statuses = {s.app: s for s in apps(lambda _line: None).statuses()}
    offers = offered(folder is not None)
    error = None
    try:
        remote = ops.remote([o.app for o in offers], settings)
    except InstanceError as exc:
        remote, error = {}, str(exc)
    rows = [
        AppOffer(
            o, o.paragraph(folder or Path("~/odysseus")), statuses[o.app], remote.get(o.app, [])
        )
        for o in offers
    ]
    projects = settings.expanded_projects_dir()
    refused = projects_problem(projects, settings)
    return AppsPanel(rows, str(projects), str(folder) if folder else None, error, refused)


def register_starter_app_routes(
    app: FastAPI,
    render: Render,
    service: InstanceService,
    jobs: JobRegistry,
    ops: AppsSetup,
) -> None:
    def job(label: str, work: Callable[[], list[str]], back: str) -> RedirectResponse:
        busy = jobs.active_for(APPS_JOB)
        if busy is not None:
            raise InstanceError(f"{busy.title} is still running; wait for it to finish.")

        def run(report: Callable[[str], None]) -> list[str]:
            lines = work()  # an AppsError fails the job with its own words
            for line in lines:
                report(line)
            return lines[-40:]

        jobs.start_unless_active(JobKind.APPS, APPS_JOB, run, label=label, keep_lines=40)
        return RedirectResponse(back, status_code=303)

    @app.post("/starter/apps/goose")
    def connect_goose() -> RedirectResponse:
        return job(
            "write Goose's file", lambda: ops.connect_goose(service.settings), "/system/apps"
        )

    @app.get("/starter/odysseus", response_class=HTMLResponse)
    def odysseus_page(request: Request) -> HTMLResponse:
        if ops.odysseus() is None:
            raise InstanceError("No Odysseus folder here (~/odysseus): mnm never installs it.")
        link = ops.link_odysseus(service.settings)
        return render(request, "odysseus_connect.html", link=link, job=jobs.latest_for(APPS_JOB))

    @app.post("/starter/odysseus")
    def odysseus_write() -> RedirectResponse:
        settings = service.settings

        def work() -> list[str]:
            link = ops.link_odysseus(settings)  # read again: what is there now is written
            return ops.apply_odysseus(settings, link)

        return job("connect Odysseus", work, "/starter/odysseus")

    @app.post("/starter/apps/librechat/user")
    def librechat_user(
        email: Annotated[str, Form()],
        password: Annotated[str, Form()],
        again: Annotated[str, Form()],
    ) -> RedirectResponse:
        if password != again:
            raise InstanceError("The two passwords differ: nothing was made.")
        label = f"{LABELS[App.LIBRECHAT]} account {email.strip()}"
        return job(label, lambda: [ops.add_user(email.strip(), password)], "/system/apps")
