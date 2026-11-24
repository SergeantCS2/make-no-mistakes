"""Web routes for instances: the home board, the status bar, one instance's page with its
log, and start / stop / stop all / remove. A start is a job under the box's start lock
(`start_turn`, or the swap group's), as `mnm start` runs it."""

from __future__ import annotations

import shlex
from collections.abc import Callable
from typing import Annotated, Literal, Protocol
from urllib.parse import urlsplit

from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from makenomistakes import router_client
from makenomistakes.capacity import Verdict
from makenomistakes.comfy.status import ComfyFeed
from makenomistakes.config import Settings
from makenomistakes.consumers import CONSUMER_LABELS, Consumer
from makenomistakes.controls import controls_for, settings_rows
from makenomistakes.diagnose import LogReview
from makenomistakes.harness import consumer_urls
from makenomistakes.harness_agents import AGENT_CLIENTS, AGENT_MODALITIES
from makenomistakes.held_back import held_back_in
from makenomistakes.instance_types import InstanceError, InstanceStatus, Started, by_type
from makenomistakes.instances import InstanceService
from makenomistakes.jobs import JobKind, JobRegistry, JobState, changed
from makenomistakes.machine import drift_now
from makenomistakes.owner_start import start_for_owner
from makenomistakes.parts import instance_parts
from makenomistakes.progress_words import parked_for_room
from makenomistakes.readings_filter import (
    instance_readings,
    left_out,
    left_out_words,
    memory_readings,
    store_here,
)
from makenomistakes.recipes import AudioCppArgs, Recipe, docker_run_argv
from makenomistakes.recipes.audiocpp import voice_names
from makenomistakes.settings_notes import ROUTER_KEYS, Line, saved
from makenomistakes.supervisor import ContainerState
from makenomistakes.web.common import Render, local
from makenomistakes.web.system_data import SystemProbe, router_view
from makenomistakes.workflows import WorkflowStore

CONSUMER_VALUES = [c.value for c in Consumer]


def known_consumer(consumer: str) -> None:
    """A hand-edited `?consumer=`: an HTML page that says why, not FastAPI's 422 JSON."""
    if consumer not in CONSUMER_VALUES:
        raise InstanceError(
            f"Unknown harness '{consumer}'; pick one of: {', '.join(CONSUMER_VALUES)}."
        )


LogView = Literal["tail", "all", "issues"]


class Forget(Protocol):
    """What the remove route clears beside the recipe (`try_routes.TryResults`)."""

    def forget(self, name: str) -> None: ...


COMFY_SAID = {  # what the ComfyUI panel's buttons did (`web/comfy_status_routes.py`)
    "stopped": "Stopped ComfyUI: its card is free.",
    "freed": "ComfyUI unloaded its models. Its Now line says what it holds since.",
}
LOG_LINES = {"tail": 400, "all": 0, "issues": 0}  # `mnm logs`' default, `--tail 0`, `--review`


def start_notes(started: Started) -> list[str]:
    """A start job's notes: what auto-fix changed, then what else the start had to say."""
    return [changed(what) for what in started.fixes] + started.notes


def start_job(
    service: InstanceService,
    recipe: Recipe,
    report: Callable[[str], None],
    room: bool = False,
    anyway: bool = False,
) -> list[str]:
    """The page's start is `mnm start`'s (`owner_start`): `room` is `--stop-others`,
    `anyway` is `--anyway`; without either, mnm's own instances holding the card refuse it."""
    done = start_for_owner(service, recipe, report, stop_others=room, anyway=anyway)
    made = [parked_for_room(done.parked)] if done.parked else []
    return made + start_notes(done.started)


def refuse_if_busy(jobs: JobRegistry, name: str, what: str) -> None:
    """A second job for an instance would be swallowed by the first: say so instead."""
    jobs.refuse_if_busy(name, what)


def instance_logs(
    service: InstanceService, recipe: Recipe, state: ContainerState, view: LogView
) -> tuple[str, LogReview, bool]:
    """The engine's log for the page, and whether it is the one `mnm stop` kept (the
    crash case, where the container is gone). Docker failing is said in the text."""
    try:
        text, review = service.logs(recipe, LOG_LINES[view])
    except InstanceError as exc:
        if state is ContainerState.ABSENT:  # nothing kept either: no log panel
            return "", LogReview(), False
        return f"({exc})", LogReview(), False
    return text, review, state is ContainerState.ABSENT and bool(text)


def register_board_routes(
    app: FastAPI,
    render: Render,
    settings: Settings,
    service: InstanceService,
    jobs: JobRegistry,
    system: SystemProbe,
    comfy: ComfyFeed | None = None,
) -> None:
    """The home board, the hardware panel, the status bar and the job lines they poll."""
    feed = comfy or ComfyFeed()

    @app.get("/", response_class=HTMLResponse)
    def instances(
        request: Request,
        removed: str = "",
        freed: int | None = None,
        kept: int = 0,
        comfy_done: Annotated[str, Query(alias="comfy")] = "",
    ) -> HTMLResponse:
        statuses = service.statuses()
        snapshot = service.memory()
        return render(
            request,
            "instances.html",
            statuses=statuses,
            groups=by_type(statuses),
            docker_error=next((s.error for s in statuses if s.error), None),
            verdicts=service.verdicts(
                statuses,
                snapshot,
                held_back_in(WorkflowStore(settings.expanded_state_dir()), statuses),
            ),
            snapshot=snapshot,
            jobs=jobs.recent(6),
            running=jobs.running_jobs(),
            router=router_view(settings, system),
            # a link's `?removed=` names only an instance that is gone
            removed="" if any(s.recipe.name == removed for s in statuses) else removed,
            freed=None if freed is None else max(freed, 0),  # None: the files stayed
            kept=max(kept, 0),
            comfy=feed.status(),
            comfy_said=COMFY_SAID.get(comfy_done, ""),
            # From machine.json as last recorded, never a reading per page load
            machine=drift_now(settings, [s.recipe for s in statuses]),
        )

    @app.get("/machine", response_class=HTMLResponse)
    def machine(request: Request) -> HTMLResponse:
        """The hardware panel alone (nvidia-smi + meminfo, no Docker)."""
        return render(request, "_machine.html", snapshot=service.memory())

    @app.get("/bar", response_class=HTMLResponse)
    def bar(request: Request) -> HTMLResponse:
        """The status bar every page polls (nvidia-smi, meminfo, the router's /health and busy
        count; no Docker). On the home page it also refreshes the machine panel."""
        current = request.headers.get("hx-current-url")
        home = current is not None and urlsplit(current).path in ("", "/")
        return render(
            request,
            "_bar.html",
            snapshot=service.memory(),
            router=router_view(settings, system),
            busy_jobs=len(jobs.running_jobs()),
            comfy_held=feed.held(),
            with_machine=home,
        )

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job(request: Request, job_id: str) -> HTMLResponse:
        found = jobs.get(job_id)
        page = render(request, "_job.html", 200 if found else 404, job=found)
        ended = found is None or found.state is not JobState.RUNNING
        if ended and request.headers.get("hx-request"):
            # Only a running job is polled, so this answer is the moment it ended (or the UI
            # restarted and forgot it): reload the page so its state, fit verdict and
            # Start/Stop button match, instead of one line changing (or polling forever).
            page.headers["HX-Refresh"] = "true"
        return page


def register_instance_routes(
    app: FastAPI, render: Render, service: InstanceService, jobs: JobRegistry, tried: Forget
) -> None:
    """One instance: its page and log panel, settings, start, stop (and stop all), remove."""
    said: dict[str, list[Line]] = {}  # what a settings save says, shown once on the page

    @app.get("/instances/{name}", response_class=HTMLResponse)
    def instance(
        request: Request,
        name: str,
        consumer: str = Consumer.ODYSSEUS.value,
        log: LogView = "tail",
    ) -> HTMLResponse:
        recipe = service.get(name)
        known_consumer(consumer)
        status = service.status(recipe)
        logs, review, kept = instance_logs(service, recipe, status.state, log)
        agents = recipe.modality in AGENT_MODALITIES
        job = jobs.latest_for(name)
        tried = job is not None and job.kind is JobKind.TRY
        ran = job is not None and job.kind is JobKind.RUN
        return render(
            request,
            "instance.html",
            recipe=recipe,
            status=status,
            verdict=_verdict(service, status),
            urls=consumer_urls(recipe, Consumer(consumer)),
            consumer=Consumer(consumer),
            consumers={c: n for c, n in CONSUMER_LABELS.items() if c not in AGENT_CLIENTS},
            agents={c: n for c, n in CONSUMER_LABELS.items() if c in AGENT_CLIENTS}
            if agents
            else {},
            controls=settings_rows(recipe, service.native_context(recipe)),
            parts=instance_parts(recipe, service.models_dir),
            logs=logs,
            review=review,
            log_view=log,
            kept_log=kept,
            job=None if tried or ran else job,  # a test's or a run's line is its tab's
            try_job=job if tried else None,
            run_job=job if ran else None,
            command=_command(service, recipe),
            voices=voice_names(recipe) if isinstance(recipe.args, AudioCppArgs) else [],
            readings=memory_readings(instance_readings(service.measurements, name))[-5:],
            elsewhere=left_out_words(
                left_out(service.measurements.readings(name), store_here(service.measurements))
            ),
            tools=service.tools.get(name) if agents else None,
            saved=said.pop(name, []),
        )

    @app.get("/instances/{name}/log", response_class=HTMLResponse)
    def log_panel(request: Request, name: str, view: LogView = "tail") -> HTMLResponse:
        """The log panel's inside: its view switcher and Refresh swap it into the panel."""
        recipe = service.get(name)
        status = service.status(recipe)
        logs, review, kept = instance_logs(service, recipe, status.state, view)
        return render(
            request,
            "_log_body.html",
            recipe=recipe,
            logs=logs,
            review=review,
            log_view=view,
            kept_log=kept,
        )

    @app.post("/instances/{name}/settings")
    async def settings_form(request: Request, name: str) -> RedirectResponse:
        """Async only to read the form; the store's file reads and writes run on a worker
        thread, never on the event loop that answers every other page."""
        recipe = await run_in_threadpool(service.get, name)
        form = await request.form()
        updates = {
            c.key: str(form[f"c_{c.key}"])
            for c in controls_for(recipe.modality, recipe.engine)
            if f"c_{c.key}" in form and isinstance(form[f"c_{c.key}"], str)
        }
        updated = await run_in_threadpool(service.update_settings, recipe, updates)
        done = await run_in_threadpool(saved, service, recipe, updated)  # what `mnm set` says
        said[name] = done.lines
        only_router = bool(updates) and updates.keys() <= ROUTER_KEYS  # a pin's own post
        if done.router_moved or only_router:  # as `mnm pin`/`unpin`; the
            # whole form's save tells it only when idle or keep moved
            await run_in_threadpool(router_client.settings_saved, service.settings, name)
        return RedirectResponse(f"/instances/{name}#settings", status_code=303)

    @app.post("/instances/stop-all")
    def stop_all(
        back: Annotated[str, Form(alias="next")] = "",
        names: Annotated[str, Form()] = "",
    ) -> RedirectResponse:
        """`mnm stop --all` (every instance with a container), or `mnm stop a b` by names."""
        service.stop_many(names.split() or None)
        return RedirectResponse(local(back, "/"), status_code=303)

    @app.post("/instances/{name}/start")
    def start(
        name: str,
        back: Annotated[str, Form(alias="next")] = "",
        room: Annotated[str, Form(alias="make_room")] = "",
        anyway: Annotated[str, Form()] = "",
    ) -> RedirectResponse:
        """Start (refused when mnm's own instances hold the room), Make room and start
        (`--stop-others`), or Start anyway (`--anyway`, after the page asked)."""
        recipe = service.get(name)
        refuse_if_busy(jobs, name, "start it")

        def run(report: Callable[[str], None]) -> list[str]:
            return start_job(service, recipe, report, room == "on", anyway == "on")

        jobs.start_unless_active(JobKind.START, name, run)
        return RedirectResponse(local(back, f"/instances/{name}"), status_code=303)

    @app.post("/instances/{name}/stop")
    def stop(name: str, back: Annotated[str, Form(alias="next")] = "") -> RedirectResponse:
        """Stop it, or switch a parked one off: the router then refuses to load it."""
        service.stop_many([name])
        return RedirectResponse(local(back, f"/instances/{name}"), status_code=303)

    @app.post("/instances/{name}/remove")
    def remove(name: str, files: Annotated[str, Form()] = "") -> RedirectResponse:
        recipe = service.get(name)
        refuse_if_busy(jobs, name, "remove it")  # its job would write after it was gone
        cleanup = service.remove(recipe, files=files == "on")
        tried.forget(name)  # its Test it result and files go with it
        jobs.forget(name)  # and its last jobs: one added again starts with none
        where = f"/?removed={name}"
        if cleanup is not None:
            where += f"&freed={cleanup.freed_bytes}&kept={len(cleanup.shared)}"
        return RedirectResponse(where, status_code=303)


def _verdict(service: InstanceService, status: InstanceStatus) -> Verdict:
    """One instance's Fits now, judged with every instance in view: which running ones to
    stop for it, and whether its swap group's mates are what holds the card. The page's own
    instance is not asked of Docker twice."""
    others = [service.status(r) for r in service.all() if r.name != status.recipe.name]
    statuses = [status, *others]
    kept = held_back_in(WorkflowStore(service.settings.expanded_state_dir()), statuses)
    return service.verdicts(statuses, service.memory(), kept)[status.recipe.name]


def _command(service: InstanceService, recipe: Recipe) -> str | None:
    """`mnm show`'s exact `docker run`, once the files are there (it names them)."""
    if not service.files_present(recipe):
        return None
    try:
        return shlex.join(docker_run_argv(recipe, service.models_dir))
    except ValueError:  # a recipe whose engine needs every file first
        return None
