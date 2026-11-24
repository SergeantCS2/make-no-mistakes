"""The instance page's Why panel (`mnm why`): the plan explained in words, planned again
for this box. It reads the model's header from its source, so the page asks for it on a click
and swaps this fragment into the panel."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from makenomistakes.instances import InstanceService
from makenomistakes.web.common import Render
from makenomistakes.why import explain


def register_why_routes(app: FastAPI, render: Render, service: InstanceService) -> None:
    @app.get("/instances/{name}/why", response_class=HTMLResponse)
    def why_panel(request: Request, name: str) -> HTMLResponse:
        recipe = service.get(name)
        return render(request, "_why.html", recipe=recipe, why=explain(service, name))
