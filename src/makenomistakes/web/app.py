"""FastAPI app: the unified model terminal on :7800. Instances, add, workflows, the System
page. HTMX polls jobs and the status bar; everything else is plain forms.

Long operations (download, start, Test it, an engine update) run as background jobs so a page
never blocks. Planning is synchronous (a few Hub calls) and stateless: the create step
re-plans from the submitted choice instead of keeping server-side session state. Failures
raise typed errors that one handler renders; no error travels through a URL. The routes live
in their own modules; this one wires them, the guards and the template globals.
"""

from __future__ import annotations

import hashlib
import re
import urllib.parse
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.trustedhost import TrustedHostMiddleware

from makenomistakes import __version__
from makenomistakes.apps.setup_ops import AppsSetup, LiveApps
from makenomistakes.backend_now import no_card
from makenomistakes.comfy.engine import ComfyEngine
from makenomistakes.comfy.github_packs import GitHub
from makenomistakes.comfy.guard import ComfyGuard
from makenomistakes.comfy.models import Models
from makenomistakes.comfy.registry import Registry
from makenomistakes.comfy.room import park_for_comfy
from makenomistakes.comfy.run import ComfyRunner
from makenomistakes.comfy.status import ComfyFeed, ComfyStatus, comfy_status, held_for_bar
from makenomistakes.comfy.store import WorkflowStore
from makenomistakes.config import Settings
from makenomistakes.consumers import browser_origin, copy_url, engine_origin, shared_base_url
from makenomistakes.controls import Kind
from makenomistakes.doctor import Check, Status
from makenomistakes.fit import FitLevel
from makenomistakes.hardware import SubprocessRunner, probe
from makenomistakes.hardware import platform as platforms
from makenomistakes.hardware.box_facts import read_box
from makenomistakes.hub.civitai import CivitaiClient
from makenomistakes.hub.kits import FAMILY_LABELS
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.jobs import JobRegistry, JobState, ago
from makenomistakes.loras import LoraService
from makenomistakes.machine_checks import machine_checks
from makenomistakes.make import RUNNABLE
from makenomistakes.model_uses import Use
from makenomistakes.notify import job_notices
from makenomistakes.pipeline.runner import FilmRunner, film_runner
from makenomistakes.pipeline.short_film import FILM_NAMES
from makenomistakes.pipeline.song_runner import SongRunner, song_runner
from makenomistakes.plan import PlanError, make_plan
from makenomistakes.reading_stamp import beside_words
from makenomistakes.runs import RunRecord, RunStore
from makenomistakes.sizes import size_words
from makenomistakes.studio.music_models import models as studio_models
from makenomistakes.studio.style_trainer import StyleTrainer, box_style_trainer
from makenomistakes.studio.trainer import StudioTrainer, Variant, box_trainer
from makenomistakes.supervisor import ContainerState
from makenomistakes.try_figures import run_figures, try_figures
from makenomistakes.validation_words import FORM_PARTS, worded
from makenomistakes.web import view
from makenomistakes.web.add_routes import PlanFn, register_add_routes
from makenomistakes.web.app_routes import AppsFactory, apps_service, register_app_routes
from makenomistakes.web.backup_routes import register_backup_routes
from makenomistakes.web.comfy_model_routes import register_comfy_model_routes, statuses_for
from makenomistakes.web.comfy_pack_routes import register_comfy_pack_routes
from makenomistakes.web.comfy_routes import register_comfy_routes
from makenomistakes.web.comfy_status_routes import register_comfy_status_routes
from makenomistakes.web.comfy_workflow_routes import (
    Fetch,
    default_fetch,
    register_comfy_workflow_routes,
)
from makenomistakes.web.common import Pages
from makenomistakes.web.health_routes import register_health_routes
from makenomistakes.web.instance_routes import register_board_routes, register_instance_routes
from makenomistakes.web.lora_routes import register_lora_routes
from makenomistakes.web.machine_routes import register_machine_routes
from makenomistakes.web.nightly_routes import register_nightly_routes
from makenomistakes.web.pipeline_routes import register_pipeline_routes
from makenomistakes.web.portal_routes import register_portal_routes
from makenomistakes.web.replan_routes import register_replan_routes
from makenomistakes.web.run_routes import (
    RunMaker,
    Runs,
    register_gallery_routes,
    register_run_routes,
)
from makenomistakes.web.sandbox import SandboxGuard
from makenomistakes.web.song_edit_routes import register_song_edit_routes
from makenomistakes.web.song_master_routes import register_song_master_routes
from makenomistakes.web.song_mix_routes import register_song_mix_routes
from makenomistakes.web.song_routes import register_song_routes
from makenomistakes.web.song_vocals_routes import register_song_vocals_routes
from makenomistakes.web.starter_app_routes import apps_panel, register_starter_app_routes
from makenomistakes.web.starter_routes import register_starter_routes
from makenomistakes.web.studio_routes import register_studio_routes
from makenomistakes.web.system_data import SystemProbe
from makenomistakes.web.system_routes import Doctor, register_system_routes
from makenomistakes.web.train_routes import register_train_routes
from makenomistakes.web.try_routes import (
    MAX_UPLOAD,
    TryResults,
    TryRunner,
    register_try_routes,
    too_large,
)
from makenomistakes.web.tune_routes import register_tune_routes
from makenomistakes.web.voice_routes import (
    CONSENT_UPLOAD,
    register_voice_routes,
    voice_upload_too_large,
)
from makenomistakes.web.why_routes import register_why_routes
from makenomistakes.web.workflow_routes import register_workflow_routes
from makenomistakes.workflows import WorkflowService

HERE = Path(__file__).parent
TEMPLATES = HERE / "templates"
STATIC = HERE / "static"
LOCAL_HOSTS = ("localhost", "127.0.0.1")  # the names the UI is opened under on the box
_SAFE_FETCH_SITES = {"same-origin", "none"}
GIB = 1024**3
HTMX = "/static/htmx-1.9.12.min.js"  # vendored: the UI works with no internet
HTMX_SRI = "sha384-ujb1lZYygJmzgSwoxRggbCHcjc0rB2XoQrxeTUQyRjrOnlCoYta87iKBWq3EsdM2"


def default_doctor(settings: Settings) -> list[Check]:
    report = probe(settings.expanded_models_dir(), host_ram=True)
    return machine_checks(settings, report, read_box(SubprocessRunner(), report.is_wsl))


def static_version() -> str:
    """A short hash of the stylesheets and script, so a browser fetches them again after an
    update instead of pairing new templates with an old cached file."""
    digest = hashlib.sha256()
    for name in ("mnm.css", "mnm.js"):
        digest.update((STATIC / name).read_bytes())
    return digest.hexdigest()[:10]


def create_app(  # noqa: PLR0913, PLR0915 - every outside dependency can be replaced
    settings: Settings,
    service: InstanceService | None = None,
    jobs: JobRegistry | None = None,
    *,
    planner: PlanFn = make_plan,
    doctor: Doctor = default_doctor,
    loras: LoraService | None = None,
    allowed_hosts: tuple[str, ...] = LOCAL_HOSTS,
    system: SystemProbe | None = None,
    tryer: TryRunner | None = None,
    maker: RunMaker | None = None,
    films: FilmRunner | None = None,
    songs: SongRunner | None = None,
    comfy: ComfyEngine | None = None,
    comfy_fetch: Fetch | None = None,
    comfy_runner: ComfyRunner | None = None,
    comfy_models: Models | None = None,
    registry: Registry | None = None,
    github: GitHub | None = None,
    apps: AppsFactory | None = None,
    app_setup: AppsSetup | None = None,
    trainer: Callable[[Variant | None], StudioTrainer] | None = None,
    style_trainer: Callable[[Variant | None], StyleTrainer] | None = None,
) -> FastAPI:
    app = FastAPI(title="makenomistakes", version=__version__, docs_url=None, redoc_url=None)
    templates = Jinja2Templates(directory=str(TEMPLATES))  # autoescapes .html by default
    templates.env.globals.update(
        gib=lambda b: f"{b / GIB:.1f} GiB",
        beside=beside_words,  # A reading taken beside something outside mnm
        version=__version__,
        static_v=static_version(),
        htmx_src=HTMX,
        htmx_sri=HTMX_SRI,
        platform=platforms.current,  # read per render: tests pin the platform per test
        FitLevel=FitLevel,
        ContainerState=ContainerState,
        JobState=JobState,
        Kind=Kind,
        Status=Status,
        copy_url=copy_url,
        browser_origin=browser_origin,
        engine_origin=engine_origin,
        router_url=shared_base_url,
        family_labels=FAMILY_LABELS,
        state_badge=view.state_badge,
        verdict_badge=view.verdict_badge,
        repo_page=view.repo_page,
        repo_site=view.repo_site,
        percent=view.percent,
        fit_badge=view.fit_badge,
        check_badge=view.check_badge,
        tools_badge=view.tools_badge,
        unit_badge=view.unit_badge,
        meter_tone=view.meter_tone,
        exposure_words=view.EXPOSURE_WORDS,
        try_shows=view.try_shows,
        try_figures=try_figures,
        run_figures=run_figures,
        ago=ago,
        size_words=size_words,
        runnable=frozenset(m.value for m in RUNNABLE),
        film_names=FILM_NAMES,
        what_uses=list(Use),  # Find a model's uses, on every Add page
    )
    pages = Pages(templates)
    service = service or InstanceService(settings)
    jobs = jobs or JobRegistry(on_finish=job_notices(settings))
    system = system or SystemProbe()
    tried = TryResults()
    templates.env.globals["try_result"] = tried.get
    templates.env.filters["ticks"] = view.ticks  # `code` spans in words the CLI shares
    templates.env.globals["no_card"] = no_card  # The bar on a box without a card
    _guards(app, pages)
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
    engine = comfy or ComfyEngine(settings)
    store = RunStore.for_settings(settings)

    def comfy_now() -> ComfyStatus:
        return comfy_status(engine, store, lambda run: _step(jobs, run))

    feed = ComfyFeed(comfy_now, lambda: held_for_bar(engine))
    register_board_routes(app, pages.render, settings, service, jobs, system, feed)
    register_comfy_status_routes(app, engine, comfy_now)
    register_instance_routes(app, pages.render, service, jobs, tried)
    register_why_routes(app, pages.render, service)
    register_add_routes(app, pages.render, settings, service, jobs, planner)
    register_try_routes(app, service, jobs, tried, tryer)
    runs = Runs(service, store, jobs, maker)
    templates.env.globals["runs_for"] = runs.runs_for
    register_run_routes(app, runs)
    register_gallery_routes(app, pages.render, runs, engine.clear_history)
    register_pipeline_routes(app, pages.render, films or film_runner(service), jobs)
    song_pipeline = songs or song_runner(service)
    register_song_routes(app, pages.render, song_pipeline, jobs)
    register_song_mix_routes(app, song_pipeline, jobs)  # The mixer and the export
    register_song_edit_routes(app, song_pipeline, jobs)
    register_song_master_routes(app, song_pipeline, jobs)
    register_song_vocals_routes(app, song_pipeline, jobs)
    register_studio_routes(  # System → Studio, with its Models table
        app, pages.render, song_pipeline.tools, jobs, service.all, lambda: studio_models(service)
    )
    register_voice_routes(app, pages.render, song_pipeline.library)
    register_train_routes(
        app,
        pages.render,
        trainer or (lambda v: box_trainer(settings, v)),
        jobs,
        settings,
        style_trainer or (lambda v: box_style_trainer(settings, v)),
    )
    register_comfy_routes(app, pages.render, service, jobs, engine)
    state = settings.expanded_state_dir()
    comfy_runner = comfy_runner or ComfyRunner(
        engine, WorkflowStore(state), store, park_for_comfy(service)
    )
    fetcher = comfy_fetch or default_fetch(settings.civitai_token_value)
    models = comfy_models or Models(  # the key is read at each Find: one added later counts
        settings.expanded_models_dir(),
        state,
        civitai=lambda: CivitaiClient(settings.civitai_token_value()),
    )
    register_comfy_pack_routes(  # before `/comfy/{name}`, which would match `/comfy/packs`
        app,
        pages.render,
        settings,
        engine,
        comfy_runner.workflows,
        service.all,
        jobs,
        registry=registry,
        github=github,
    )
    register_comfy_workflow_routes(
        app,
        pages.render,
        comfy_runner,
        jobs,
        fetcher,
        lambda name: statuses_for(models, comfy_runner.workflows, name, service.all),
    )
    register_comfy_model_routes(app, settings, comfy_runner.workflows, models, service.all, jobs)
    register_system_routes(app, pages.render, settings, service, jobs, system, doctor)
    register_backup_routes(app, pages.render, settings, service, jobs, system)
    register_nightly_routes(app, templates, settings, service, jobs, system)
    register_machine_routes(app, pages.render, service, jobs)
    register_health_routes(app, settings, system)  # `/__mnm/health`
    apps = apps or apps_service(service)
    app_setup = app_setup or LiveApps()
    register_app_routes(app, pages.render, service, jobs, apps)
    register_portal_routes(app, pages.render, service, apps)
    register_starter_app_routes(app, pages.render, service, jobs, app_setup)
    register_lora_routes(app, loras or LoraService(service), jobs)
    register_replan_routes(app, service, pages.render)
    register_tune_routes(app, templates, pages.render, service, jobs)
    register_starter_routes(
        app,
        service,
        jobs,
        pages.render,
        planner,
        lambda: apps_panel(service, app_setup, apps),
    )
    register_workflow_routes(app, WorkflowService(service), jobs, pages.render)
    # A browser on another site can rebind its own name to 127.0.0.1; its requests then pass
    # the same-origin check but name that site in Host.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))
    # added last, so it runs first: ComfyUI's container meets nothing else (web/sandbox.py)
    app.add_middleware(SandboxGuard, guard=ComfyGuard(settings.expanded_state_dir()))
    return app


def _step(jobs: JobRegistry, run: RunRecord) -> str:
    """A workflow run's last progress line, when the web started it (its job's message)."""
    job = jobs.active_for(f"comfy:{run.target}")
    return job.message if job is not None else ""


def _same_origin(request: Request) -> bool:
    """No `Origin` (a terminal's curl, a form some browsers send without one), or this page's
    own: never another site's, nor the `null` of a sandboxed frame."""
    origin = request.headers.get("origin")
    if origin is None:
        return True
    return urllib.parse.urlsplit(origin).netloc == request.headers.get("host", "")


FORM_BYTES = 2**20  # a form's fields around its file
# The pages that take one file, the most each takes and what takes more: refused by the
# length a request says, before Starlette spools a body of any size to disk
ONE_FILE_PAGES: tuple[tuple[re.Pattern[str], int, str], ...] = (
    (re.compile(r"/songs"), MAX_UPLOAD,
     "A recording here is up to 32 MiB: `mnm song new --recording FILE` takes a larger one."),
    (re.compile(r"/songs/[^/]+/master"), MAX_UPLOAD,
     "A reference here is up to 32 MiB: `mnm song master <run> --reference FILE` takes a "
     "larger one."),
    (re.compile(r"/system/voices/[^/]+/consent"), CONSENT_UPLOAD,
     "A consent document here is up to 32 MiB: `mnm voices consent <name> FILE` takes a "
     "larger one."),
)  # fmt: skip


def _too_large(request: Request) -> str | None:
    """Why an upload is refused for its size, before it is read; None when it is not."""
    path, length = request.url.path, request.headers.get("content-length")
    if too_large(path, length):
        return (
            "That upload is too large for Test it: use smaller files, or `mnm try` from a terminal."
        )
    if voice_upload_too_large(path, length):
        return (
            "That upload is too large for this page: `mnm voices add` takes the files from "
            "this machine."
        )
    if length is not None and length.isdigit():  # one file each
        for page, limit, words in ONE_FILE_PAGES:
            if page.fullmatch(path) and int(length) > limit + FORM_BYTES:
                return words
    return None


def _guards(app: FastAPI, pages: Pages) -> None:
    """The CSRF check, and one error page for every typed failure (never a traceback)."""

    @app.middleware("http")
    async def same_origin_posts(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """A localhost tool still needs CSRF protection: other sites can POST forms to it."""
        site = request.headers.get("sec-fetch-site")
        if request.method == "POST" and site is not None and site not in _SAFE_FETCH_SITES:
            return Response("cross-site requests are not accepted", status_code=403)
        if request.method == "POST" and site is None and not _same_origin(request):
            # a browser without Fetch Metadata (Safari before 16.4) still sends Origin
            return Response("cross-site requests are not accepted", status_code=403)
        refused = _too_large(request) if request.method == "POST" else None
        if refused is not None:  # before Starlette spools the body to disk
            return pages.render(
                request, "error.html", 413, message=refused, diagnoses=[], guidance=None
            )
        return await call_next(request)

    @app.exception_handler(InstanceError)
    async def instance_error(request: Request, exc: InstanceError) -> HTMLResponse:
        return pages.render(
            request, "error.html", 400, message=str(exc), diagnoses=exc.diagnoses, guidance=None
        )

    @app.exception_handler(RequestValidationError)
    async def _bad_form(request: Request, exc: RequestValidationError) -> HTMLResponse:
        fields = worded(exc.errors(), skip=FORM_PARTS)
        return pages.render(
            request,
            "error.html",
            400,
            message=f"The form sent a value this page does not take ({fields}). Reload the page "
            "and try again.",
            diagnoses=[],
            guidance=None,
        )

    @app.exception_handler(PlanError)
    async def plan_error(request: Request, exc: PlanError) -> HTMLResponse:
        return pages.render(
            request, "error.html", 400, message=str(exc), diagnoses=[], guidance=exc.guidance
        )
