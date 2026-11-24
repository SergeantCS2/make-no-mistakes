"""Optimise for quality on the web: the page with the plan `mnm tune
--quality` shows (the Hub read, nothing downloaded) and its approval as a job, which plans
again, refuses a plan that moved since the page showed it, downloads what is missing and
queues the tune for the router's tuner (`quality.approve`); with no router answering it waits
in the queue, never run in the web UI's process."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.jobs import JobKind, JobRegistry, Report
from makenomistakes.quality import FLOOR_MAX, FLOOR_MIN, QualityPlanError
from makenomistakes.quality.approve import approve
from makenomistakes.quality.prepare import Offer, offer
from makenomistakes.quality.words import plan_rows, plan_words
from makenomistakes.tune.session import untunable
from makenomistakes.web.common import Render

Floor = float | None
NO_ROUTER = (
    "The router is not running: the quality tune waits in its queue and runs once the router "
    "does (System → Services, or `mnm service install`)."
)


@dataclass(frozen=True)
class QualityAsk:
    """The page's choices for a quality tune: another floor, the quants named, `--keep`."""

    floor: Floor = None
    quants: str = ""  # comma-separated, as `--quants` takes them
    keep: bool = False

    @property
    def names(self) -> list[str]:
        return [q.strip() for q in self.quants.split(",") if q.strip()]

    def offer(self, service: InstanceService, name: str) -> Offer:
        return offer(service, name, self.floor, self.names, self.keep)


def short_words(offered: Offer | None) -> str | None:
    """The disk's refusal, when the models disk cannot hold the tune."""
    if offered is None or not offered.room_short:
        return None
    return offered.short_words[0].upper() + offered.short_words[1:] + "."


def register_quality_routes(
    app: FastAPI, render: Render, service: InstanceService, jobs: JobRegistry
) -> None:
    @app.get("/instances/{name}/quality", response_class=HTMLResponse)
    def quality_page(
        request: Request,
        name: str,
        floor: Annotated[Floor, Query(ge=FLOOR_MIN, le=FLOOR_MAX)] = None,
        quants: str = "",
        keep: str = "",
    ) -> HTMLResponse:
        """The plan, and the button that approves it."""
        service.get(name)  # a 404 page for an unknown name
        asked = QualityAsk(floor=floor, quants=quants, keep=keep == "on")
        try:
            offered: Offer | None = asked.offer(service, name)
            error = None
        except (QualityPlanError, InstanceError) as exc:
            offered, error = None, str(exc)
        return render(
            request,
            "quality.html",
            instance=name,
            asked=asked,
            offered=offered,
            error=error,
            rows=plan_rows(offered) if offered else [],
            words=plan_words(offered) if offered else [],
            short=short_words(offered),
            floor_min=f"{FLOOR_MIN:g}",
            floor_max=f"{FLOOR_MAX:g}",
        )

    @app.post("/instances/{name}/quality")
    def quality_approve(
        name: str,
        shown: Annotated[str, Form()],
        floor: Annotated[Floor, Form(ge=FLOOR_MIN, le=FLOOR_MAX)] = None,
        quants: Annotated[str, Form()] = "",
        keep: Annotated[str, Form()] = "",
    ) -> RedirectResponse:
        recipe = service.get(name)
        if (why := untunable(recipe)) is not None:
            raise InstanceError(f"{name}: {why}.")
        asked = QualityAsk(floor=floor, quants=quants, keep=keep == "on")

        def work(report: Report) -> list[str]:
            offered = asked.offer(service, name)
            if offered.plan.fingerprint != shown:
                raise InstanceError(
                    f"{name}'s quality plan moved since the page showed it (the card, the "
                    "instance or the Hub): Optimise for quality shows the new one."
                )
            if offered.room_short:
                raise QualityPlanError(offered.short_words)
            approved = approve(service, name, offered, True, report)
            said = [f"The quality tune {approved.replaced}."] if approved.replaced else []
            if approved.answer is not None:
                return [*said, approved.answer[1]]
            # Never run here: the plan waits in the queue for the router (a tune of hours in
            # the web UI's process would hold the card beside whatever the router loads).
            return [*said, approved.waiting or NO_ROUTER.format(name=name)]

        jobs.start(JobKind.TUNE, name, work, busy="tune it", label=f"quality tune {name}")
        return RedirectResponse(f"/instances/{name}#tuning", status_code=303)
