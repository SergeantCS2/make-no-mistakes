"""`GET /__mnm/health` on the web UI: the router's answer with the web UI up, or,
when the router does not answer, `down` with the rest read here (`health_now.for_web`).
No page: JSON for an outside monitor, never cached."""

from __future__ import annotations

from fastapi import FastAPI, Response

from makenomistakes.config import Settings
from makenomistakes.health import HEALTH_PATH
from makenomistakes.health_now import HealthReader, for_web
from makenomistakes.web.system_data import SystemProbe


def register_health_routes(app: FastAPI, settings: Settings, probe: SystemProbe) -> None:
    reader = HealthReader(settings, probe.runner)

    @app.get(HEALTH_PATH)
    def health() -> Response:
        found = for_web(reader, lambda: probe.health(settings), lambda: probe.router_up(settings))
        return Response(
            found.model_dump_json(),
            media_type="application/json",
            headers={"Cache-Control": "no-store"},
        )
