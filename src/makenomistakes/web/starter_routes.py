"""The web UI's Get started page: the starter models `mnm setup` and `mnm starter`
offer, each added with one press: planned for this card, saved, downloaded as a job; and the
workspace apps setup offers (`starter_app_routes.py`)."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from makenomistakes.busy_card import plan_note
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.jobs import JobKind, JobRegistry
from makenomistakes.plan_budget import planning_budget
from makenomistakes.recipes import Recipe
from makenomistakes.starter import BY_KEY, CATALOG, LARGE, floor_words, offered_on, room_words
from makenomistakes.starter_add import PlanFn, add_item, chosen_profile, plan_item, saved_words
from makenomistakes.web.common import Render
from makenomistakes.web.starter_app_routes import AppsPanel


def register_starter_routes(
    app: FastAPI,
    service: InstanceService,
    jobs: JobRegistry,
    render: Render,
    planner: PlanFn,
    apps_panel: Callable[[], AppsPanel],
) -> None:
    @app.get("/starter", response_class=HTMLResponse)
    def starter_page(request: Request) -> HTMLResponse:
        snapshot = service.memory()
        gpu = snapshot.gpu
        offered = offered_on(snapshot)
        statuses = service.statuses()
        have = {s.recipe.name for s in statuses}
        budget = planning_budget(service, snapshot, statuses)
        # the free VRAM a starter's plan gets (mnm's running models counted free)
        free = budget.vram_free_bytes if budget else gpu.vram_free_bytes if gpu else None
        return render(
            request,
            "starter.html",
            offered=offered,
            hidden=[s for s in CATALOG if s not in offered],
            floors={s.key: floor_words(s) for s in (*CATALOG, *LARGE)},
            short={s.key: room_words(s, free) for s in offered},
            have=have,
            snapshot=snapshot,
            busy_note=plan_note(budget) if budget is not None else None,
            apps=apps_panel(),
        )

    @app.post("/starter/{key}")
    def add_starter(key: str) -> RedirectResponse:
        starter = BY_KEY.get(key)
        if starter is None:
            raise InstanceError(f"No starter model is called {key!r}.")
        have = {s.recipe.name for s in service.statuses()}
        added: list[Recipe] = []
        for item in starter.items:
            if item.name in have:
                continue
            plan = plan_item(service, item, planner)
            saved = saved_words(chosen_profile(plan, item), plan)  # the row it saves
            recipe = add_item(service, item, plan)
            added.append(recipe)  # its download starts now: a later item's error cannot strand it

            def download(
                report: Callable[[str], None], recipe: Recipe = recipe, saved: str = saved
            ) -> list[str]:
                service.download(recipe, report)
                return [saved]

            jobs.start_unless_active(JobKind.DOWNLOAD, recipe.name, download)
        # The home board shows every download's progress, not only the first one's.
        return RedirectResponse("/#activity" if added else "/starter", status_code=303)
