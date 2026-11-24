"""The System tab: the router and services with which settings are set, the engines (status,
update with its self-test, test, rollback), the calibration (and its export), the doctor
(and its GPU container test), and the ComfyUI links. `mnm update`, `mnm service install` and
`mnm setup` stay in a terminal: the first two restart this very page, the last needs sudo.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from makenomistakes import engines
from makenomistakes.backend_now import box_backend
from makenomistakes.bridge.card import mcp_card
from makenomistakes.calibration import MIN_READINGS, MachineSummary, export
from makenomistakes.config import Settings
from makenomistakes.disk import disk_report
from makenomistakes.disk_settings import when_words
from makenomistakes.disk_types import gb
from makenomistakes.doctor import Check, Status
from makenomistakes.engine_check import check_engines, save_report
from makenomistakes.engine_selftest import Outcome
from makenomistakes.engine_update import FAILURES, Line, self_test, update_engines
from makenomistakes.hub.facts import Engine
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.jobs import JobFailed, JobKind, JobRegistry
from makenomistakes.local_time import local
from makenomistakes.quality.readings import QUALITY_CAPTION
from makenomistakes.queue_summary import QUEUE_CAPTION
from makenomistakes.readings_filter import readings_here, store_here
from makenomistakes.web.common import Render
from makenomistakes.web.system_data import (
    STARTED_AT,
    SystemProbe,
    calibration_view,
    deep_words,
    engines_view,
    router_view,
    setting_rows,
)

Doctor = Callable[[Settings], list[Check]]
# Job keys a colon keeps apart from instance names (which never hold one), as `workflow:`.
ENGINES_JOB = "system:engines"  # one engine job at a time: updates and tests share builds
CHECK_JOB = "system:engines-check"  # `mnm engines check` (asks the registry, pulls nothing)
DOCTOR_JOB = "system:doctor"
JOURNAL_LINES = (50, 100, 400)


def register_system_routes(  # noqa: PLR0915 - the routes share these seven
    app: FastAPI,
    render: Render,
    settings: Settings,
    service: InstanceService,
    jobs: JobRegistry,
    system: SystemProbe,
    doctor: Doctor,
) -> None:
    state_dir = settings.expanded_state_dir

    @app.get("/system", response_class=HTMLResponse)
    def overview(request: Request) -> HTMLResponse:
        return render(
            request,
            "system.html",
            router=router_view(settings, system),
            units=system.units(),
            settings_rows=setting_rows(settings, system),
            revision=system.revision(),
            started_at=STARTED_AT,
            uptime=_uptime(STARTED_AT),
            journal_lines=JOURNAL_LINES,
            mcp_card=mcp_card(settings),
            mcp_tools=system.mcp_tools(service),
        )

    @app.get("/system/journal", response_class=HTMLResponse)
    def journal(request: Request, lines: int = 100) -> HTMLResponse:
        """`mnm service logs -n N`: both services' newest lines (read-only)."""
        count = lines if lines in JOURNAL_LINES else 100
        return render(request, "_journal.html", text=system.journal(count), lines=count)

    @app.get("/system/engines", response_class=HTMLResponse)
    def engines_page(request: Request, rolled: str = "") -> HTMLResponse:
        return render(
            request,
            "engines.html",
            found=engines_view(service),
            job=jobs.latest_for(ENGINES_JOB),
            official=engines.box_images(),  # Test runs the official image, not a custom one
            rolled=rolled,
            checking=jobs.latest_for(CHECK_JOB),
        )

    @app.post("/system/engines/check")
    def check() -> RedirectResponse:
        """`mnm engines check` as a job: the registry asked once per image, nothing pulled."""

        def run(report: Callable[[str], None]) -> list[str]:
            report("asking the registry for each engine's newest build…")
            try:
                found = check_engines(service.supervisor, service.all())
            except FAILURES as exc:
                raise JobFailed(f"Docker could not be asked: {exc}", []) from exc
            save_report(state_dir(), found)
            return [row.words for row in found.rows] + [line.line for line in found.instances]

        jobs.start_unless_active(JobKind.CHECK, CHECK_JOB, run, label="engines check")
        return RedirectResponse("/system/engines", status_code=303)

    def engine_job(label: str, work: Callable[[Callable[[Line], None]], bool]) -> None:
        busy = jobs.active_for(ENGINES_JOB)
        if busy is not None:
            raise InstanceError(f"{busy.title} is still running; wait for it to finish.")

        def run(report: Callable[[str], None]) -> list[str]:
            lines: list[str] = []

            def say(line: Line) -> None:
                lines.append(line.text)
                report(line.text.strip())

            try:
                failed = work(say)
            except FAILURES as exc:
                raise JobFailed(str(exc), lines) from exc
            if failed:
                raise JobFailed(f"{label}: something failed (see the lines below).", lines)
            return lines

        jobs.start_unless_active(JobKind.ENGINES, ENGINES_JOB, run, label=label)

    @app.post("/system/engines/update")
    def update(engine: Annotated[Engine | None, Form()] = None) -> RedirectResponse:
        """`mnm engines update [ENGINE]`: pulls, then the self-test; a failure rolls back."""
        which = {engine} if engine else set()
        engine_job(
            f"update {engine.value if engine else 'engines'}",
            lambda say: update_engines(service, service.supervisor, state_dir(), which, say),
        )
        return RedirectResponse("/system/engines", status_code=303)

    @app.post("/system/engines/{engine}/test")
    def test(engine: Engine) -> RedirectResponse:
        """`mnm engines test ENGINE`: a copy of the smallest instance on the current build."""
        image = engines.image_for(engine.value)

        def work(say: Callable[[Line], None]) -> bool:
            result = self_test(service, engine, image, state_dir(), None, say, roll_back=False)
            return result.outcome is Outcome.FAILED

        engine_job(f"test {engine.value}", work)
        return RedirectResponse("/system/engines", status_code=303)

    @app.post("/system/engines/{engine}/rollback")
    def rollback(engine: Engine) -> RedirectResponse:
        """`mnm engines rollback ENGINE`: the tag back on the build the last update replaced."""
        try:
            done = engines.rollback(
                service.supervisor, engines.image_for(engine.value), state_dir()
            )
        except FAILURES as exc:
            raise InstanceError(str(exc)) from exc
        return RedirectResponse(
            f"/system/engines?rolled={engine.value}+{done.restored.short_id}", status_code=303
        )

    @app.get("/system/calibration", response_class=HTMLResponse)
    def calibration(request: Request) -> HTMLResponse:
        return render(
            request,
            "calibration.html",
            found=calibration_view(service),
            min_readings=MIN_READINGS,
            queue_caption=QUEUE_CAPTION,
            quality_caption=QUALITY_CAPTION,
        )

    @app.get("/system/calibration/export")
    def calibration_export() -> Response:
        """`mnm calibration export`, as a download instead of a file in a folder."""
        gpu = service.memory().gpu
        kept, here = service.measurements.every(), store_here(service.measurements)
        written = export(
            kept,
            gpu.name if gpu else None,
            gpu.vram_total_bytes if gpu else None,
            here={name: readings_here(found, here) for name, found in kept.items()},
            machine=MachineSummary(id=here.id, words=here.words),
        )
        name = f"mnm-calibration-{local():%Y%m%d-%H%M%S}.json"
        return Response(
            written.model_dump_json(indent=2) + "\n",
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )

    @app.get("/system/disk", response_class=HTMLResponse)
    def disk_page(request: Request) -> HTMLResponse:
        """`mnm disk`: the models folder by instance, folder and unlisted; the disk policy."""
        return render(request, "disk.html", report=disk_report(service), gb=gb, when=when_words)

    @app.get("/doctor", response_class=HTMLResponse)
    def doctor_page(request: Request) -> HTMLResponse:
        return render(
            request,
            "doctor.html",
            checks=doctor(settings),
            job=jobs.latest_for(DOCTOR_JOB),
            deep=deep_words(box_backend()),
        )

    @app.post("/doctor/deep")
    def doctor_deep() -> RedirectResponse:
        """`mnm doctor --deep`: the checks again with a GPU container (a job: a pull)."""

        def run(report: Callable[[str], None]) -> list[str]:
            report(deep_words(box_backend()).running)
            checks = system.deep(settings)
            lines = [f"{c.name}: {c.status.value.upper()} · {c.detail}" for c in checks]
            if any(c.status is Status.FAIL for c in checks):
                raise JobFailed("A check failed.", lines)
            return lines

        jobs.start_unless_active(JobKind.CHECK, DOCTOR_JOB, run, label="doctor --deep")
        return RedirectResponse("/doctor", status_code=303)


def _uptime(moment: datetime) -> str:
    seconds = max(int((datetime.now(UTC) - moment).total_seconds()), 0)  # WSL's clock steps back
    if seconds < 3600:
        return f"{seconds // 60} min"
    if seconds < 86400:
        return f"{seconds // 3600} h {seconds % 3600 // 60} min"
    return f"{seconds // 86400} d {seconds % 86400 // 3600} h"
