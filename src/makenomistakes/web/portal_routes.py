"""The portal: the Status page's first panel, the one bookmark for the box. Where clients
point (the router from this machine and from its containers, and whether it asks a key), the
desk's roles (each a model id clients send, linked to its instance), and the workspace apps
with their state, address and login note (each app's own `notes`: where its password is, how
its first account is made). Who is signed in to an app is the app's own business: mnm reads no
app's sessions. Loaded after the page (`hx-trigger="load"`): an app's state asks Docker or
systemd, about a second with several installed."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from makenomistakes.aliases import Aliases
from makenomistakes.apps.defs import Kind
from makenomistakes.apps.hub import AppStatus
from makenomistakes.apps.ledger import AppsError
from makenomistakes.client_view import DOCKER_HOST
from makenomistakes.consumers import shared_base_url
from makenomistakes.desk import read_aliases
from makenomistakes.instances import InstanceService
from makenomistakes.json_folder import cause
from makenomistakes.web.app_routes import AppsFactory
from makenomistakes.web.common import Render
from makenomistakes.workflows import WorkflowStore


@dataclass(frozen=True)
class Portal:
    here: str  # the router's base URL from this machine
    containers: str  # and from a container on it
    keyed: bool  # the router asks every request for its key
    desk: Aliases
    apps: list[AppStatus]
    apps_problem: str | None = None

    @property
    def installed(self) -> list[AppStatus]:
        return [a for a in self.apps if a.installed is not None or a.error]

    @property
    def not_installed(self) -> list[AppStatus]:
        """Apps mnm installs that are not installed here."""
        return [a for a in self._rest if a.definition.kind is not Kind.CONNECT]

    @property
    def connected_only(self) -> list[AppStatus]:
        """The apps mnm only connects (Goose, Odysseus): set up in their own config, which mnm
        writes or shows; whether they are connected is theirs to say, so the panel never offers
        to connect them whatever their state."""
        return [a for a in self._rest if a.definition.kind is Kind.CONNECT]

    @property
    def _rest(self) -> list[AppStatus]:
        return [a for a in self.apps if a.installed is None and not a.error]


def portal_now(service: InstanceService, apps: AppsFactory) -> Portal:
    settings = service.settings
    try:  # the desk's roles leave out a role an instance is named after
        names = service.store.names()
    except OSError as exc:
        desk = Aliases(problem=f"the instances could not be listed ({cause(exc)})")
    else:
        desk = read_aliases(WorkflowStore(settings.expanded_state_dir()), names)
    try:
        found, problem = apps(lambda _line: None).statuses(), None
    except (AppsError, OSError) as exc:
        found, problem = [], str(exc)
    return Portal(
        here=shared_base_url(),
        containers=shared_base_url(DOCKER_HOST),
        keyed=settings.api_key_value() is not None,
        desk=desk,
        apps=found,
        apps_problem=problem,
    )


def register_portal_routes(
    app: FastAPI, render: Render, service: InstanceService, apps: AppsFactory
) -> None:
    @app.get("/portal", response_class=HTMLResponse)
    def portal(request: Request) -> HTMLResponse:
        return render(request, "_portal.html", portal=portal_now(service, apps))
