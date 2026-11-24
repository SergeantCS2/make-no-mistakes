"""`mnm starter`: the starter models, the same menu `mnm setup` ends with. Each pick is
an ordinary `mnm add`, so the planner fits it to this machine; one already added is kept."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.markup import escape

from makenomistakes.cli._app import app, console, fail_with, print_progress
from makenomistakes.cli.hub_cmds import print_plan
from makenomistakes.config import Settings
from makenomistakes.hardware import probe_memory
from makenomistakes.hardware.probe import MemorySnapshot
from makenomistakes.instance_types import InstanceError
from makenomistakes.instance_vram import vram_holders
from makenomistakes.instances import InstanceService
from makenomistakes.plan import PlanError
from makenomistakes.plan_budget import planning_budget
from makenomistakes.quality.readings import measured_quants
from makenomistakes.starter import (
    CATALOG,
    GIB,
    Item,
    download_gb,
    floor_words,
    offered_on,
    parse_picks,
    room_words,
)
from makenomistakes.starter_add import add_item, chosen_profile, plan_item, saved_words


def add_starter_item(item: Item) -> bool:
    """`mnm add` for one starter item, the plan shown first; False when it did not get
    added (the reason printed). An instance already under that name is kept, and counts."""
    service = InstanceService(Settings())
    try:
        service.get(item.name)
    except InstanceError:
        pass
    else:
        console.print(f"[dim]{escape(item.name)} is already added; kept as it is.[/dim]")
        return True
    console.print(f"\n[bold]mnm add {escape(item.reference)} --name {escape(item.name)}[/bold]")
    try:
        plan = plan_item(service, item)
        chosen = chosen_profile(plan, item)  # the row it saves, never the other two
        measured = measured_quants(service.measurements, plan.facts.repo_id)
        print_plan(plan, [chosen], vram_holders(plan), measured)
        recipe = add_item(service, item, plan)
        console.print(f"{escape(saved_words(chosen, plan))}")
        console.print(f"{escape(item.name)}: saved, port {recipe.port}.")
        service.download(recipe, print_progress)
    except (PlanError, InstanceError) as exc:
        console.print(f"[red]{escape(item.name)} not added:[/red] {escape(str(exc))}")
        return False
    return True


def _free_now(snapshot: MemorySnapshot) -> int | None:
    """The VRAM a starter's plan gets now (mnm's running models counted free, as `mnm add`
    counts them); the card's own free figure when the instances or the RAM cannot be read."""
    gpu = snapshot.gpu
    if gpu is None:
        return None
    try:
        budget = planning_budget(InstanceService(Settings()), snapshot)
    except InstanceError:  # an unreadable recipe: `mnm ls` says which
        budget = None
    return budget.vram_free_bytes if budget is not None else gpu.vram_free_bytes


@app.command()
def starter(
    picks: Annotated[
        list[str] | None,
        typer.Argument(help="Numbers or names from the list (chat, memory …), `all`, or none."),
    ] = None,
    list_only: Annotated[bool, typer.Option("--list", help="Show the menu only.")] = False,
) -> None:
    """Add starter models that fit this card (each is `mnm add`, planned for this machine)."""
    snapshot = probe_memory()
    gpu = snapshot.gpu
    offered = offered_on(snapshot)
    free = _free_now(snapshot)
    for number, item in enumerate(offered, 1):
        names = ", ".join(i.name for i in item.items)
        console.print(
            f" {number}. [bold]{escape(item.title):<13}[/bold] {escape(names):<12} "
            f"~{item.download_gb:>4.1f} GB, {floor_words(item):<26}  {escape(item.what)}"
        )
        if (short := room_words(item, free)) is not None:
            console.print(f"    [red]{escape(short)}[/red]")
        if item.note:
            console.print(f"    [yellow]{escape(item.note)}[/yellow]")
    hidden = [f"{s.title} ({floor_words(s)})" for s in CATALOG if s not in offered]
    if hidden and gpu is not None:
        console.print(
            f"[dim]Not offered on this card: {escape(', '.join(hidden))}; it has "
            f"{gpu.vram_total_bytes / GIB:.1f} GiB.[/dim]"
        )
    if list_only:
        return
    answer = " ".join(picks) if picks else typer.prompt("Which? (Enter: chat)", default="")
    chosen = parse_picks(answer, offered)
    if chosen is None:
        fail_with(ValueError(f"{answer!r} names nothing in the list."))
    if not chosen:
        console.print("Nothing added.")
        return
    for picked in chosen:
        if picked.note:
            console.print(f"[yellow]{escape(picked.title)}:[/yellow] {escape(picked.note)}")
    console.print(f"About {download_gb(chosen):.1f} GB to download.")
    left = [i.name for s in chosen for i in s.items if not add_starter_item(i)]
    if left:
        fail_with(InstanceError(f"Not added: {', '.join(left)} (see above)."))
    console.print(
        "[green]Done.[/green] `mnm start <name>`, or send it a request: the router loads it."
    )
