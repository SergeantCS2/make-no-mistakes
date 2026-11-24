"""`mnm replan`: plan a saved instance again for the machine it runs on now; `--all`
re-plans each instance in turn (after the card was busy while models were added)."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.markup import escape

from makenomistakes.busy_card import replan_note
from makenomistakes.cli._app import (
    GpuOption,
    RamOption,
    VramOption,
    app,
    budget_override,
    console,
    fail_with,
    print_said,
)
from makenomistakes.cli.complete import instance_names
from makenomistakes.config import Settings
from makenomistakes.fit import Budget
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.plan_budget import planning_budget
from makenomistakes.replan import planned_as_it_is
from makenomistakes.replan import replan as replan_instance
from makenomistakes.replan import save as save_replan
from makenomistakes.replan_words import replan_lines
from makenomistakes.settings_notes import doomed_plan


@app.command()
def replan(
    name: Annotated[
        str | None, typer.Argument(show_default=False, autocompletion=instance_names)
    ] = None,
    every: Annotated[
        bool, typer.Option("--all", help="Every instance, one after another.")
    ] = False,
    vram_gb: VramOption = None,
    ram_gb: RamOption = 14.0,
    gpu: GpuOption = "RTX 4070",
    any_quant: Annotated[
        bool,
        typer.Option("--any-quant", help="Let the plan pick the quant too (may download)."),
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Save without asking.")] = False,
    shrink: Annotated[
        bool,
        typer.Option(
            "--shrink",
            help="Take the new plan even when it is smaller and the saved one still fits.",
        ),
    ] = False,
) -> None:
    """Plan a saved instance again for this machine now (a new card, more RAM, a server, a
    card freed): same name, port, type and profile; the settings the plan decides are its
    again."""
    if every and name is not None:
        fail_with(InstanceError("Name one instance or pass --all, not both."))
    service = InstanceService(Settings())
    try:
        budget = budget_override(vram_gb, ram_gb, gpu) or planning_budget(service)
        if every:
            names = [r.name for r in service.all()]
        elif name is not None:
            names = [name]
        else:
            fail_with(InstanceError("Name the instance to re-plan, or pass --all."))
    except InstanceError as exc:
        fail_with(exc)
    if not names:
        console.print("No instances to re-plan.")
        return
    note = replan_note(budget) if budget is not None else None
    if note:
        console.print(f"[yellow]{escape(note)}[/yellow]")
    failed = [
        n
        for n in names
        if not _replan_one(service, n, budget, any_quant=any_quant, yes=yes, shrink=shrink)
    ]
    if failed:
        fail_with(InstanceError(f"Not re-planned: {', '.join(failed)} (see above)."))


def _replan_one(
    service: InstanceService,
    name: str,
    budget: Budget | None,
    *,
    any_quant: bool,
    yes: bool,
    shrink: bool,
) -> bool:
    """Plan one instance again and save it when told to; False when it could not be."""
    try:
        done = replan_instance(
            service, name, budget, keep_quant=not any_quant, keep_if_fits=not shrink
        )
    except InstanceError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        return False
    # a withheld plan whose load mode alone moved is `changed`: saved below
    print_said(replan_lines(done, name, f"Re-planned {name} for this machine:"))
    if not done.changed:
        return _as_it_is(service, name)
    if not yes and not typer.confirm("Save these settings?", default=True):
        console.print("Kept the old settings.")
        return True
    try:
        saved = save_replan(service, done)
    except InstanceError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        return False
    console.print(f"Saved. They apply on the next start: [bold]mnm start {escape(name)}[/bold]")
    if (doomed := doomed_plan(service, saved)) is not None:
        console.print(f"[yellow]{escape(doomed)}[/yellow]")
    return True


def _as_it_is(service: InstanceService, name: str) -> bool:
    """The plan was judged on this box and kept: the recipe notes the box."""
    try:
        planned_as_it_is(service, name)
    except InstanceError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        return False
    return True
