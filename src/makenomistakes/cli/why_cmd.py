"""`mnm why <name>`: an instance's plan explained in words (`why.explain`): what it runs,
why the planner picks that for this box, what a plan for the box as it is now would change,
and what more room would change. Read-only; it reads the model's header from the Hub."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.markup import escape

from makenomistakes.cli._app import app, console, fail_with
from makenomistakes.cli.complete import instance_names
from makenomistakes.config import Settings
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.why import EXTRA_GIB, explain


@app.command()
def why(
    name: Annotated[
        str,
        typer.Argument(help="The instance whose plan to explain.", autocompletion=instance_names),
    ],
    more: Annotated[
        int,
        typer.Option(
            "--more", min=1, max=256, help="How many GiB more room to plan with for the last part."
        ),
    ] = EXTRA_GIB,
) -> None:
    """Explain an instance's plan in words: why this quant, this context, this much on the
    card, and what more room would change. Plans and saves nothing."""
    service = InstanceService(Settings())
    try:
        said = explain(service, name, more)
    except InstanceError as exc:
        fail_with(exc)
    console.print(escape(said.runs))
    if said.problem is not None:
        console.print(f"[yellow]{escape(said.problem)}[/yellow]")
        raise typer.Exit(code=1)
    for title, lines in (("Why", said.reasons), ("Now", said.today), ("More room", said.more)):
        console.print(f"\n[bold]{title}[/bold]")
        for line in lines:
            console.print(f"  {escape(line)}")
