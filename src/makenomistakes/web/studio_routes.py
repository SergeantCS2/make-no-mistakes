"""System → Studio: the studio tools image (install,
rollback, uninstall, each a job the page polls), what it carries, and the stages registry
with the user's picks, as `mnm studio status` and `mnm studio stage` show them."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from makenomistakes.jobs import BUILD_LINES, JobRegistry
from makenomistakes.recipes import Recipe
from makenomistakes.studio import stages
from makenomistakes.studio.music_models import Models
from makenomistakes.studio.stage_instances import playing
from makenomistakes.studio.stages import StudioError
from makenomistakes.studio.tools import CARRIES, IMAGE_GIB, StudioTools, image_tag
from makenomistakes.web.common import Render
from makenomistakes.web.image_routes import image_job, image_state, register_image_routes

STUDIO_JOB = "studio-tools"


def register_studio_routes(
    app: FastAPI,
    render: Render,
    tools: StudioTools,
    jobs: JobRegistry,
    recipes_now: Callable[[], list[Recipe]],
    models_now: Callable[[], Models],
) -> None:
    job = image_job(jobs, STUDIO_JOB, "The studio tools'")

    @app.get("/system/studio", response_class=HTMLResponse)
    def studio(request: Request) -> HTMLResponse:
        state = image_state(tools)
        folder = tools.settings.expanded_state_dir()
        try:  # an unreadable picks file is said once here; each row then reads unplayed
            stages.read_picks(folder)
        except StudioError as exc:
            state["problem"] = str(exc)
        recipes = recipes_now()
        rows = [
            (row, tier, playing(row.stage, tier, folder, recipes), stages.tier_words(row, tier))
            for row in stages.STAGES.values()
            for tier in row.tiers
        ]
        return render(
            request,
            "studio.html",
            tab="system",
            **state,
            current=image_tag(),
            image_gib=IMAGE_GIB,
            carries=CARRIES,
            rows=rows,
            models=models_now(),
            workflow_pick=stages.WORKFLOW_PICK,
            job=jobs.latest_for(STUDIO_JOB),
        )

    @app.post("/system/studio/install")
    def install() -> RedirectResponse:
        job(
            "install the studio tools",
            lambda report: f"built {tools.install(report).image}",
            BUILD_LINES,
        )
        return RedirectResponse("/system/studio", status_code=303)

    register_image_routes(app, "/system/studio", lambda: tools, job, "the runs keep their scores")
