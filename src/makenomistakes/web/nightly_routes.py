"""System → Nightly: the last night's report (or a chosen one), its verdicts with what
mnm did or offers, each step's result, a link per earlier night, the readings graph of every
chat-like instance, and Run now (`mnm nightly run --now` as a job, refused at once with the
reason while the router is in use). The instance page's readings graph is the template
global `speed_graph_for`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from makenomistakes.config import Settings
from makenomistakes.hub.facts import CHAT_LIKE
from makenomistakes.instances import InstanceService
from makenomistakes.jobs import JobFailed, JobKind, JobRegistry
from makenomistakes.nightly import settings as nightly_settings
from makenomistakes.nightly import timer
from makenomistakes.nightly.gate import Gate, gate_for
from makenomistakes.nightly.graph import Graph, merged, speed_graph
from makenomistakes.nightly.report import (
    NightlyReport,
    NightlyStore,
    Outcome,
    StepName,
    StepState,
    Verdict,
)
from makenomistakes.nightly.wiring import NightlyBusyError, run_nightly
from makenomistakes.readings_filter import instance_readings, store_here
from makenomistakes.web.common import Pages
from makenomistakes.web.system_data import SystemProbe
from makenomistakes.web.view import Badge, Tone

NIGHTLY_JOB = "system:nightly"
Runner = Callable[[Settings, Callable[[str], None]], NightlyReport]

OUTCOME_BADGES = {
    Outcome.QUIET: Badge("quiet", "✓", Tone.OK),
    Outcome.REGRESSION: Badge("regression", "✕", Tone.BAD),
    Outcome.TO_READ: Badge("to read", "!", Tone.INV),
    Outcome.FAILED: Badge("failed", "✕", Tone.BAD),
    Outcome.WAITED: Badge("waited", "!", Tone.INV),
    Outcome.RUNNING: Badge("running", "◔", Tone.BUSY),
}
STATE_BADGES = {
    StepState.OK: Badge("ok", "✓", Tone.OK),
    StepState.WARN: Badge("warn", "!", Tone.INV),
    StepState.FAILED: Badge("failed", "✕", Tone.BAD),
    StepState.SKIPPED: Badge("skipped", "○", Tone.DIM),
    StepState.NOT_BUILT: Badge("not built", "○", Tone.DIM),
}


def verdict_class(verdict: Verdict) -> str:
    """A regression or a failure on the alert colour; a steady row bright; the rest plain."""
    if verdict.regression or verdict.told:
        return "alert"
    return "hi" if verdict.kind.value == "steady" else ""


def run_now(settings: Settings, say: Callable[[str], None]) -> NightlyReport:
    """Run now: no idle minute, and no wait: a router in use refuses it at once."""
    return run_nightly(settings, say, now=True, wait=False)


@dataclass(frozen=True)
class Hooks:
    """What Run now calls (tests replace them): the run, and the router's gate."""

    runner: Runner
    gate: Callable[[Settings], Gate]


def register_nightly_routes(
    app: FastAPI,
    templates: Jinja2Templates,
    settings: Settings,
    service: InstanceService,
    jobs: JobRegistry,
    system: SystemProbe,
    hooks: Hooks | None = None,
) -> None:
    render = Pages(templates).render
    store = NightlyStore(settings.expanded_state_dir())
    hooked = hooks or Hooks(run_now, gate_for)  # read now: tests replace them

    def graph_for(name: str, reports: list[NightlyReport] | None = None) -> Graph | None:
        readings = instance_readings(service.measurements, name)  # this box's
        nights = store.reports() if reports is None else reports
        here = store_here(service.measurements)
        return speed_graph(name, merged(readings, nights, name, here))

    templates.env.globals["speed_graph_for"] = graph_for

    def graphs() -> list[Graph]:
        chat = [r.name for r in service.all() if r.modality in CHAT_LIKE]
        nights = store.reports()  # read once for every graph
        return [g for name in chat if (g := graph_for(name, nights)) is not None]

    refusals: list[str] = []  # Run now's refusal, shown once by the page it redirects to

    def page(request: Request, chosen: str | None) -> HTMLResponse:
        refused = refusals.pop() if refusals else None
        every = store.every()
        shown = next((f for f in every if f.id == chosen), None) if chosen else None
        if shown is None:
            shown = next((f for f in reversed(every) if f.report is not None), None)
        try:
            clean = nightly_settings.read(settings.expanded_state_dir()).clean_words
        except nightly_settings.NightlySettingsError as exc:
            clean = str(exc)
        return render(
            request,
            "nightly.html",
            shown=shown,
            report=(report := shown.report if shown is not None else None),
            tried=report.step(StepName.TRY) if report is not None else None,
            every=list(reversed(every)),
            timer=timer.state(system.runner).line,
            clean=clean,
            graphs=graphs(),
            job=jobs.latest_for(NIGHTLY_JOB),
            refused=refused,
            outcome_badges=OUTCOME_BADGES,
            state_badges=STATE_BADGES,
            verdict_class=verdict_class,
        )

    @app.get("/system/nightly", response_class=HTMLResponse)
    def nightly_page(request: Request, id: str | None = None) -> HTMLResponse:
        return page(request, id)

    @app.post("/system/nightly/run")
    def nightly_run() -> RedirectResponse:
        """`mnm nightly run --now` as a job; refused with the reason while the router is in
        use (a tune, a request in flight, a start or stop, a ComfyUI job)."""
        if (why := hooked.gate(settings).in_use()) is not None:
            refusals[:] = [f"Not run: {why}. Try again once it is idle."]
            return RedirectResponse("/system/nightly", status_code=303)

        def work(report: Callable[[str], None]) -> list[str]:
            try:
                done = hooked.runner(settings, report)
            except NightlyBusyError as exc:
                raise JobFailed(str(exc), []) from exc
            lines = [f"{done.id}: {done.outcome.value}", *(v.line for v in done.told)]
            if done.waited:
                raise JobFailed(done.waited, lines)
            return lines

        jobs.start_unless_active(JobKind.NIGHTLY, NIGHTLY_JOB, work, label="nightly run --now")
        return RedirectResponse("/system/nightly", status_code=303)
