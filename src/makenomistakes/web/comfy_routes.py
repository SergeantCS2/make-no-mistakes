"""System → ComfyUI: link the picture models, and install,
start or stop ComfyUI itself. A build or a start runs as a job; the page polls it."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from makenomistakes import comfyui
from makenomistakes.comfy.engine import IMAGE_GIB, ComfyEngine, ComfyError, State
from makenomistakes.comfy.models import ModelError, Models
from makenomistakes.comfy.packs import approved_ids
from makenomistakes.comfy.preflight import preflight, refusals
from makenomistakes.comfy.variants import install_variant
from makenomistakes.hardware.probe import probe_memory
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.jobs import BUILD_LINES, JobRegistry
from makenomistakes.web.comfy_jobs import COMFY_JOB, start_comfy_job
from makenomistakes.web.common import Render


def register_comfy_routes(
    app: FastAPI,
    render: Render,
    service: InstanceService,
    jobs: JobRegistry,
    engine: ComfyEngine,
) -> None:
    def job(
        title: str, work: Callable[[Callable[[str], None]], str], keep_lines: int = 0
    ) -> RedirectResponse:
        def run(report: Callable[[str], None]) -> list[str]:
            try:
                return [work(report)]
            except ComfyError as exc:
                raise InstanceError(str(exc)) from exc

        start_comfy_job(jobs, engine, COMFY_JOB, title, run, keep_lines)  # one at a time
        return RedirectResponse("/system/comfyui", status_code=303)

    def layout() -> comfyui.Layout:
        models = Models(service.models_dir, service.settings.expanded_state_dir())
        try:
            return models.layout(service.all())
        except ModelError as exc:
            raise InstanceError(str(exc)) from exc

    @app.get("/system/comfyui", response_class=HTMLResponse)
    def comfyui_page(request: Request, linked: int = -1) -> HTMLResponse:
        state = engine.state()
        variant = install_variant(False, probe_memory().gpu is not None)
        try:  # the page that says why ComfyUI reads as not installed (an unreadable ledger)
            ledger, ledger_error = engine.ledger(), ""
        except ComfyError as exc:
            ledger, ledger_error = None, str(exc)
        return render(
            request,
            "comfyui.html",
            layout=layout(),
            block=comfyui.yaml_block(service.models_dir),
            linked=linked,
            state=state,
            State=State,
            ledger=ledger,
            ledger_error=ledger_error,
            stats=engine.stats() if state is State.RUNNING else None,
            variant=variant,
            image_gib=IMAGE_GIB[variant.value],
            job=jobs.latest_for(COMFY_JOB),
        )

    @app.post("/system/comfyui/link")
    def comfyui_link() -> RedirectResponse:
        """`mnm comfyui`: link the image models under <models>/comfyui/ (writing ComfyUI's
        own yaml, `--write`, stays in a terminal: it is a file outside mnm's folders)."""
        planned = layout()
        try:
            comfyui.write_links(service.models_dir, planned)
        except OSError as exc:
            raise InstanceError(f"Could not link the models for ComfyUI: {exc}") from exc
        return RedirectResponse(f"/system/comfyui?linked={planned.count}", status_code=303)

    @app.post("/system/comfyui/install")
    def install() -> RedirectResponse:
        gpus = probe_memory().gpus
        variant = install_variant(False, isinstance(gpus, list) and bool(gpus))
        refused = refusals(preflight(variant, gpus, engine.runner))
        if refused:  # said before a job starts a 10-30 minute build that cannot work
            raise InstanceError(" ".join(refused))

        def work(report: Callable[[str], None]) -> str:
            ledger = engine.install(variant, report)
            packs = approved_ids(service.settings.expanded_state_dir())
            if ledger.packs_image is None and packs:
                return (
                    f"built {ledger.image}; your approved node packs are built over one image: "
                    "Node packs → Install builds them over this one"
                )
            return f"built {ledger.image}"

        return job("install ComfyUI", work, keep_lines=BUILD_LINES)  # the build as it goes

    @app.post("/system/comfyui/start")
    def start() -> RedirectResponse:
        return job("start ComfyUI", lambda report: f"ready at {engine.start(report)}")

    @app.post("/system/comfyui/stop")
    def stop() -> RedirectResponse:
        """A job, refused at once while another holds the engine (`job`), and Docker's
        refusal is the job's worded failure."""

        def work(_report: Callable[[str], None]) -> str:
            engine.stop()
            return "stopped"

        return job("stop ComfyUI", work)
