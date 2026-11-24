"""The tuner on the web: the instance page's Tuning tab (its trials, the
settings chosen, Optimise for speed and Cancel as jobs), the Status board's "tuning X (…)" line
and System → Router's queue. The page reads through template globals (`tuning_for`,
`tuning_line`, `tuning_queue`), so no other route changes for it. Optimise for
quality is a page with the plan `mnm tune --quality` shows and its approval as a job."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.jobs import Job, JobKind, JobRegistry, Report
from makenomistakes.recipes import Recipe
from makenomistakes.recipes.tuned import Tuned
from makenomistakes.router_client import router_tune, router_tuning
from makenomistakes.tune.apply import key_or_none, tuned_word
from makenomistakes.tune.cancel import cancel_tune
from makenomistakes.tune.orphans import Refused, closing_now, queue_speed
from makenomistakes.tune.session import untunable
from makenomistakes.tune.store import Entry, GaveUp, Queued, TuneState, TuneStore, TuneStoreError
from makenomistakes.tune.trial_types import TrialResult
from makenomistakes.web.common import Render
from makenomistakes.web.quality_routes import register_quality_routes
from makenomistakes.web.tune_here import HereRuns, answered

SHOWN_TRIALS = 30
CANCEL = "cancel:"  # a Cancel's job key: never the tune job's own, which it must reach


@dataclass(frozen=True)
class TuningView:
    """What the Tuning tab shows."""

    why_not: str | None  # the tuner leaves it alone: why
    tuned: Tuned | None
    word: str | None  # `tuned +18%`, `tuned for another build`
    queued: Entry | None
    state: TuneState | None
    trials: list[TrialResult] = field(default_factory=list)  # newest first
    job: Job | None = None
    problem: str | None = None  # a tuning file mnm cannot read
    gave_up: GaveUp | None = None  # its last tune ended without a result, and why
    closing: str | None = None  # an approval holds it, or its run is ending it (`closing_words`)


def tuning_view(service: InstanceService, jobs: JobRegistry, recipe: Recipe) -> TuningView:
    store = TuneStore(service.settings.expanded_state_dir())
    why = untunable(recipe)
    found = [jobs.latest_for(recipe.name), jobs.latest_for(CANCEL + recipe.name)]
    tuning = [j for j in found if j is not None and j.kind is JobKind.TUNE]
    job = max(tuning, key=lambda j: j.started_at) if tuning else None
    problem, closing = None, None
    try:
        queued, state = store.queued(recipe.name), store.state(recipe.name)
        if queued is not None:
            closing = closing_now(service, store, queued)
    except TuneStoreError as exc:
        queued, state, problem = None, None, str(exc)
    key = key_or_none(service, recipe, None) if recipe.tuned is not None else None
    return TuningView(
        why_not=why,
        tuned=recipe.tuned,
        word=tuned_word(recipe, key),
        queued=queued,
        state=state,
        trials=list(reversed(store.trials(recipe.name)))[:SHOWN_TRIALS],
        job=job,
        problem=problem,
        gave_up=store.gave_up(recipe.name),
        closing=closing,
    )


def tuning_line(service: InstanceService) -> str | None:
    """The tune running now: the router's, else a terminal's (`mnm tune --now`)."""
    line = router_tuning(service.settings)
    if line is not None:
        return line
    running = TuneStore(service.settings.expanded_state_dir()).running()
    return running[0].line if running else None


def tuning_queue(service: InstanceService) -> list[Entry]:
    """System's queue; one whose folder cannot be listed shows nothing there (each instance's
    Tuning tab still reads its own entry by name, and the router's journal says why)."""
    try:
        return TuneStore(service.settings.expanded_state_dir()).waiting()
    except TuneStoreError:
        return []


def register_tune_routes(
    app: FastAPI,
    templates: Jinja2Templates,
    render: Render,
    service: InstanceService,
    jobs: JobRegistry,
) -> None:
    globals_ = templates.env.globals
    globals_["tuning_for"] = lambda recipe: tuning_view(service, jobs, recipe)
    globals_["tuning_line"] = lambda: tuning_line(service)
    globals_["tuning_queue"] = lambda: tuning_queue(service)
    globals_["tuning_closing"] = lambda entry: _closing(service, entry)
    here = HereRuns(service)

    @app.post("/instances/{name}/tune")
    def tune_now(name: str) -> RedirectResponse:
        """Optimise for speed: asked of the router's tuner, else run in this process."""
        recipe = service.get(name)
        if (why := untunable(recipe)) is not None:
            raise InstanceError(f"{name}: {why}.")

        def work(report: Report) -> list[str]:
            body = json.dumps({"action": "tune", "now": True}).encode()
            asked = router_tune(service.settings, name, body)
            if asked is not None:
                return answered(asked)
            store = TuneStore(service.settings.expanded_state_dir())
            entry = Queued(name=name, now=True, reason="asked")
            queued = queue_speed(service, store, entry, report)  # a quality plan waits
            if isinstance(queued, Refused):
                raise InstanceError(queued.words)
            return here.run(queued, report)

        jobs.start(JobKind.TUNE, name, work, busy="tune it", label=f"tune {name}")
        return RedirectResponse(f"/instances/{name}#tuning", status_code=303)

    @app.post("/instances/{name}/tune/cancel")
    def tune_cancel(name: str) -> RedirectResponse:
        service.get(name)  # a 404 page for an unknown name

        def work(report: Report) -> list[str]:
            return cancel_here(service, here, name)

        jobs.start_unless_active(JobKind.TUNE, CANCEL + name, work, label=f"cancel {name}")
        return RedirectResponse(f"/instances/{name}#tuning", status_code=303)

    register_quality_routes(app, render, service, jobs)  # Optimise for quality


def _closing(service: InstanceService, entry: Entry) -> str | None:
    """System's word on a queued tune its run or approval is ending, as `mnm tune --status`
    and the Tuning tab say it (`closing_now`); None for one simply waiting."""
    return closing_now(service, TuneStore(service.settings.expanded_state_dir()), entry)


def cancel_here(service: InstanceService, here: HereRuns, name: str) -> list[str]:
    """The Tuning tab's Cancel: the router's when it answers, unless this process runs the
    tune itself; else `tune.cancel.cancel_tune` here, which cuts a run of this process's own
    so it looks at its entry's mark now."""
    if not here.runs(name):
        asked = router_tune(service.settings, name, json.dumps({"action": "cancel"}).encode())
        if asked is not None:
            return answered(asked)
    store = TuneStore(service.settings.expanded_state_dir())
    here.cancel_asked(name)  # before it is judged, while this process's run is registered
    cancelled = cancel_tune(service, store, name, here.cut)
    if not cancelled.ok:
        raise InstanceError(cancelled.words)
    return list(cancelled.lines)
