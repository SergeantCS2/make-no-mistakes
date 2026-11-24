"""Pipelines in the web UI: the Short film form, a run's page with its steps,
and the review points (edit the shots, redraw a still, continue). Every move that sends
requests runs as a job keyed `pipeline:<run id>`, so a page never blocks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.datastructures import FormData

from makenomistakes.instance_types import InstanceError
from makenomistakes.jobs import JobKind, JobRegistry
from makenomistakes.pipeline import short_film as film
from makenomistakes.pipeline.runner import SONG, FilmRunner, instances_for
from makenomistakes.pipeline.short_film import FilmInputs, ShotList
from makenomistakes.pipeline.steps import StepError
from makenomistakes.runs import RunError, RunKind, RunRecord, RunState, StepState
from makenomistakes.web.common import Render

Work = Callable[[Callable[[str], None]], RunRecord]


def _key(run_id: str) -> str:
    return f"pipeline:{run_id}"


def register_pipeline_routes(  # noqa: PLR0915 - the pages and the review moves, one place
    app: FastAPI, render: Render, runner: FilmRunner, jobs: JobRegistry
) -> None:
    store = runner.store

    def in_background(run_id: str, title: str, work: Work) -> None:
        busy = jobs.active_for(_key(run_id))
        if busy is not None:
            raise InstanceError(f"{run_id} is busy ({busy.title}); wait for it.")

        def job(report: Callable[[str], None]) -> list[str]:
            run = work(report)
            if run.state is RunState.FAILED:
                raise InstanceError(run.error)
            return [_where(run)]

        jobs.start_unless_active(JobKind.PIPELINE, _key(run_id), job, label=title)

    def page(run_id: str) -> RedirectResponse:
        return RedirectResponse(f"/pipelines/{run_id}", status_code=303)

    @app.get("/pipelines", response_class=HTMLResponse)
    def pipelines(request: Request) -> HTMLResponse:
        # a song's mixes are runs under it, shown on its page and in the gallery
        runs = [r for r in store.all() if r.kind is RunKind.PIPELINE and r.parent is None]
        return render(
            request,
            "pipelines.html",
            tab="runs",
            runs=runs,
            choices=instances_for(
                runner.service,
                [m.name for m in runner.comfy.workflows.all()] if runner.comfy else [],
            ),
            max_shots=film.MAX_SHOTS,
            labels=film.STAGE_LABELS,
        )

    @app.post("/pipelines")
    async def start(request: Request) -> RedirectResponse:
        form = await request.form()
        fields = {k: str(v).strip() for k, v in form.items() if isinstance(v, str)}
        values: dict[str, object] = {k: v for k, v in fields.items() if v}
        values["auto"] = fields.get("auto") == "true"
        try:
            inputs = FilmInputs.model_validate(values)
        except ValueError as exc:
            raise InstanceError(f"The film's form: {exc}") from exc
        run = runner.start(inputs)
        in_background(run.id, "short film", lambda report: runner.advance(run.id, report))
        return page(run.id)

    @app.get("/pipelines/{run_id}", response_class=HTMLResponse, response_model=None)
    def one(request: Request, run_id: str) -> HTMLResponse | RedirectResponse:
        try:
            run = store.get(run_id)
        except RunError as exc:
            raise InstanceError(str(exc)) from exc
        if run.kind is not RunKind.PIPELINE:
            raise InstanceError(f"{run_id} is not a pipeline run (`Runs` has it).")
        if run.target == SONG:  # A song's own page
            return RedirectResponse(f"/songs/{run_id}", status_code=303)
        shots = None
        if any(s.stage == film.SHOTLIST and s.state is StepState.DONE for s in run.steps):
            try:
                shots = runner.shots(run)
            except StepError:  # unreadable: the page offers to redo the shot list
                shots = None
        return render(
            request,
            "pipeline.html",
            tab="runs",
            run=run,
            shots=shots,
            job=jobs.latest_for(_key(run_id)),
            labels=film.STAGE_LABELS,
            stages=film.STAGES,
            per_shot=(film.STILLS, film.LINES, film.CLIPS),
            RunState=RunState,
        )

    @app.post("/pipelines/{run_id}/go")
    def proceed(run_id: str) -> RedirectResponse:
        in_background(run_id, "short film", lambda report: runner.proceed(run_id, report))
        return page(run_id)

    @app.post("/pipelines/{run_id}/resume")
    def resume(run_id: str) -> RedirectResponse:
        in_background(run_id, "short film", lambda report: runner.advance(run_id, report))
        return page(run_id)

    @app.post("/pipelines/{run_id}/redo")
    def redo(
        run_id: str,
        stage: Annotated[str, Form()],
        index: Annotated[int, Form(ge=0, lt=film.MAX_SHOTS)] = 0,
    ) -> RedirectResponse:
        if stage not in film.STAGES or stage == film.FILM:
            raise InstanceError(f"'{stage}' is not a step that can be made again.")
        in_background(
            run_id, f"redo {stage}", lambda report: runner.redo(run_id, stage, index, report)
        )
        return page(run_id)

    @app.post("/pipelines/{run_id}/shots")
    async def edit_shots(request: Request, run_id: str) -> RedirectResponse:
        busy = jobs.active_for(_key(run_id))
        if busy is not None:  # Go on another tab: its record and this edit would collide
            raise InstanceError(f"{run_id} is busy ({busy.title}); edit the shots after it.")
        runner.edit_shots(run_id, _shots_from(await request.form()))
        return page(run_id)


def _shots_from(form: FormData) -> ShotList:
    """The shot list form: `still_<n>`, `motion_<n>`, `line_<n>`; an empty picture prompt
    drops that shot."""
    shots: list[dict[str, str]] = []
    for number in range(film.MAX_SHOTS):
        still = str(form.get(f"still_{number}", "")).strip()
        if still:
            motion = str(form.get(f"motion_{number}", "")).strip() or still
            line = str(form.get(f"line_{number}", "")).strip()
            shots.append({"still_prompt": still, "motion_prompt": motion, "line": line})
    if not shots:
        raise InstanceError("Keep at least one shot (its picture prompt filled in).")
    title = str(form.get("title", "")).strip()[:200]
    try:
        return ShotList.model_validate({"title": title, "shots": shots})
    except ValueError as exc:
        raise InstanceError(f"The shots: {exc}") from exc


def _where(run: RunRecord) -> str:
    match run.state:
        case RunState.WAITING:
            return f"waiting for you at the {film.STAGE_LABELS[run.waiting_at or '']}"
        case RunState.DONE:
            return run.answer
        case _:
            return run.state.value
