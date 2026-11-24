"""`mnm status`: the box on one screen (`status_now`): the card, the health verdict,
the doctor's worst lines, the models by state, ComfyUI, what waits in the router, the tuner,
the last nightly, the web UI's address and, last, when the hardware changed since plans were
made. Exit 1 when the router is down or a doctor check fails, as `mnm doctor` exits; exit 2
when a recipe cannot be read, said in the models' place after health and the doctor."""

from __future__ import annotations

import typer
from rich.markup import escape

from makenomistakes.cli._app import app, console, print_said
from makenomistakes.cli.ls_lines import (
    box_drift,
    print_comfy,
    print_machine,
    print_queue,
    print_tuning,
    said_drift,
)
from makenomistakes.config import Settings
from makenomistakes.doctor import Status, worst
from makenomistakes.health import HealthStatus
from makenomistakes.health_now import web_listens
from makenomistakes.held_back import held_back_in
from makenomistakes.instance_types import InstanceError, InstanceStatus
from makenomistakes.instances import InstanceService
from makenomistakes.nightly.report import NightlyStore
from makenomistakes.router_client import router_now
from makenomistakes.status_now import (
    doctor_checks,
    doctor_words,
    health_here,
    health_words,
    instance_words,
    nightly_words,
    web_words,
)
from makenomistakes.workflows import WorkflowStore


@app.command()
def status() -> None:
    """The box on one screen: health, the doctor's worst lines, the models by state, what
    waits in the router, the last nightly and the web UI's address."""
    settings = Settings()
    service = InstanceService(settings)
    statuses: list[InstanceStatus] | None = None
    problem: str | None = None
    try:
        statuses = service.statuses()
    except InstanceError as exc:  # said in its place; health and the doctor still shown
        problem = str(exc)
    snapshot = service.memory()
    print_machine(snapshot)
    moved = box_drift(service, snapshot, [s.recipe for s in statuses or []])
    health = health_here(settings)
    checks = doctor_checks(settings)
    verdicts = {}
    if statuses:
        held = held_back_in(WorkflowStore(settings.expanded_state_dir()), statuses)
        verdicts = service.verdicts(statuses, snapshot, held)
    drifted = moved.words() if moved is not None else None
    print_said([*health_words(health, settings, drifted), *doctor_words(checks)])
    print_said(instance_words(statuses, verdicts, problem))
    now = router_now(settings)
    print_comfy(service)
    print_queue(now)
    print_tuning(service, now)
    if now is not None and now.training:
        console.print(f"Trainer: {escape(now.training)}")
    latest = NightlyStore(settings.expanded_state_dir()).latest()
    print_said([*nightly_words(latest), *web_words(settings, web_listens(settings))])
    said_drift(moved)
    if problem is not None:  # as `mnm ls` exits on a recipe it cannot read
        raise typer.Exit(code=2)
    failed = health.status is HealthStatus.DOWN or worst(checks) is Status.FAIL
    raise typer.Exit(code=1 if failed else 0)
