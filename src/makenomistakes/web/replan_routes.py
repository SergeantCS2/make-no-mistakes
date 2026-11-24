"""Web routes for re-planning an instance on new hardware: see what moves, then save."""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from makenomistakes.busy_card import replan_note
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.plan_budget import planning_budget
from makenomistakes.replan import replan, save
from makenomistakes.settings_notes import doomed_plan
from makenomistakes.web.common import Render


def register_replan_routes(app: FastAPI, service: InstanceService, render: Render) -> None:
    @app.post("/instances/{name}/replan", response_class=HTMLResponse)
    def replan_page(
        request: Request, name: str, any_quant: Annotated[str, Form()] = ""
    ) -> HTMLResponse:
        """Plan it again for this machine and show what would move; nothing is saved.
        `any_quant` lets the plan pick another quant (`mnm replan --any-quant`)."""
        budget = planning_budget(service)
        done = replan(service, name, budget, keep_quant=any_quant != "on")
        return render(
            request,
            "replan.html",
            done=done,
            any_quant=any_quant == "on",
            busy_note=replan_note(budget) if budget is not None else None,
            doomed=doomed_plan(service, done.new) if done.changed else None,
        )

    @app.post("/instances/{name}/replan/save")
    def replan_save(
        name: str, shown: Annotated[str, Form()] = "", any_quant: Annotated[str, Form()] = ""
    ) -> RedirectResponse:
        """Plan again and save when it is the plan the page showed: the card may have
        changed between the two clicks. The settings apply on the next start."""
        done = replan(service, name, planning_budget(service), keep_quant=any_quant != "on")
        if done.fingerprint != shown:
            raise InstanceError(
                f"The card changed since the page was shown, so {name}'s plan did too. Press "
                "Re-plan again to see the new one."
            )
        save(service, done)
        return RedirectResponse(f"/instances/{name}#settings", status_code=303)
