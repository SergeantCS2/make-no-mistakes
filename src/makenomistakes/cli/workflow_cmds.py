"""`mnm sets`: named sets of instances with a role each, started and stopped together
(`mnm workflow` is a hidden alias)."""

from __future__ import annotations

from typing import Annotated

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from makenomistakes.capacity import FitNow
from makenomistakes.cli._app import (
    app,
    console,
    fail_with,
    print_block,
    print_changes,
    spinner_progress,
)
from makenomistakes.cli.instance_cmds import VERDICT_STYLE
from makenomistakes.cli.ls_lines import state_cell
from makenomistakes.config import Settings
from makenomistakes.consumers import Consumer
from makenomistakes.desk import aliases_of, shadowed_note
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.progress_words import parked_for_room
from makenomistakes.workflows import ROLE_ORDER, Workflow, WorkflowService, docker_error

workflow_app = typer.Typer(
    help="Sets: a chat model, a coder, an image model, a voice … started together, planned "
    "to fit together, one card per harness."
)
app.add_typer(workflow_app, name="sets")
app.add_typer(workflow_app, name="workflow", hidden=True)  # another name for it


def _service() -> WorkflowService:
    return WorkflowService(InstanceService(Settings()))


def _members(pairs: list[str]) -> dict[str, str]:
    members: dict[str, str] = {}
    for pair in pairs:
        role, sep, instance = pair.partition("=")
        if not sep or not role or not instance:
            raise InstanceError(
                f"'{pair}' is not role=instance (for example chat=qwen3-8b); the roles are: "
                f"{', '.join(ROLE_ORDER)}."
            )
        if role in members:
            raise InstanceError(f"the role '{role}' is given twice.")
        members[role] = instance
    return members


@workflow_app.command("create")
def create(
    name: Annotated[str, typer.Argument(help="The set's name.")],
    members: Annotated[
        list[str], typer.Argument(help="role=instance pairs: chat=qwen3-8b image=zimage …")
    ],
    swap: Annotated[
        bool,
        typer.Option(
            "--swap",
            help="The members take turns on the GPU: each starts at its first request, "
            "stopping the others when the card is short.",
        ),
    ] = False,
    persistent: Annotated[
        bool,
        typer.Option(
            "--persistent",
            help="Its members are never unloaded while they run: not for room, not when idle.",
        ),
    ] = False,
) -> None:
    """Create a set from existing instances. Nothing starts."""
    service = _service()
    try:
        workflow = service.create(name, _members(members), swap=swap, persistent=persistent)
    except InstanceError as exc:
        fail_with(exc)
    roles = ", ".join(f"{role}={instance}" for role, instance in workflow.ordered())
    console.print(f"Saved set [bold]{escape(name)}[/bold]: {escape(roles)}.")
    if swap:
        console.print(
            "The members take turns: a request to a stopped one waits while it loads "
            "(20-60 s, longer on vLLM's first start)."
        )
    if persistent:
        console.print(PERSISTENT)
    console.print(f"Next: [bold]mnm sets start {escape(name)}[/bold]")


PERSISTENT = (
    "Persistent: while a member runs, nothing unloads it: not to make room for another model "
    "(the router, `mnm start --stop-others`, a ComfyUI run), not when it is idle. `mnm stop` "
    "still stops it."
)


@workflow_app.command("edit")
def edit(
    name: Annotated[str, typer.Argument(help="The set's name.")],
    members: Annotated[
        list[str] | None,
        typer.Argument(help="role=instance pairs to add or replace: image=klein …"),
    ] = None,
    drop: Annotated[
        list[str] | None, typer.Option("--drop", help="A role to take out (repeat for more).")
    ] = None,
    persistent: Annotated[
        bool | None,
        typer.Option(
            "--persistent/--no-persistent",
            help="Its members are never unloaded while they run (or are again).",
        ),
    ] = None,
) -> None:
    """Change a set's members or whether it is persistent (its instances stay; nothing starts)."""
    service = _service()
    try:
        workflow = service.edit(name, _members(members or []), drop or [], persistent)
    except InstanceError as exc:
        fail_with(exc)
    roles = ", ".join(f"{role}={instance}" for role, instance in workflow.ordered())
    kept = " (persistent)" if workflow.persistent else ""
    console.print(f"Saved set [bold]{escape(name)}[/bold]{kept}: {escape(roles)}.")
    if persistent:
        console.print(PERSISTENT)


@workflow_app.command("ls")
def list_cmd() -> None:
    """Every set with its members and whether they fit together right now."""
    service = _service()
    try:
        workflows = service.all()
        desk = service.store.desk.name()
        snapshot = service.instances.memory()
        rows = []
        for workflow in workflows:
            members = service.members(workflow, snapshot)
            rows.append((workflow, members, service.listed(workflow, members, snapshot)))
    except InstanceError as exc:
        fail_with(exc)
    if not workflows:
        console.print("No sets yet. Try: mnm sets create studio chat=<name> image=<name>")
        return
    table = Table(title="sets", box=box.SIMPLE_HEAD)
    table.add_column("Name", min_width=6)
    table.add_column("Members", overflow="fold")
    table.add_column("Running", no_wrap=True)
    table.add_column("Fit together", overflow="fold")
    for workflow, members, verdict in rows:
        style = VERDICT_STYLE[verdict.level]
        names = ", ".join(f"{m.role}={m.name}{' (gone)' if m.missing else ''}" for m in members)
        table.add_row(
            escape(workflow.name)
            + (" [bold](desk)[/bold]" if workflow.name == desk else "")
            + (" [dim](persistent)[/dim]" if workflow.persistent else ""),
            escape(names),
            f"{sum(m.running for m in members)}/{len(members)}",
            f"[{style}]{escape(verdict.words)}[/{style}]",
        )
    console.print(table)
    for workflow, _members, verdict in rows:
        if verdict.level not in (FitNow.FITS, FitNow.RUNNING):
            console.print(f"[dim]{escape(workflow.name)}: {escape(verdict.note)}[/dim]")


@workflow_app.command("show")
def show(name: Annotated[str, typer.Argument()]) -> None:
    """The members, each one's state and fit verdict, and the fit-together verdict."""
    service = _service()
    try:
        workflow = service.get(name)
        snapshot = service.instances.memory()
        members = service.members(workflow, snapshot)
    except InstanceError as exc:
        fail_with(exc)
    kind = (
        " (persistent: its members are never unloaded, for room or when idle)"
        if workflow.persistent
        else ""
    )
    console.print(f"[bold]{escape(name)}[/bold]{kind}")
    if _desk_name(service) == name:
        console.print(DESK_NOTE.format(roles=", ".join(r for r, _ in workflow.ordered())))
        names = (r.name for r in service.instances.all())
        shadowed = shadowed_note(aliases_of(workflow, names))
        if shadowed:  # a restored backup can bring one back: the instance's name wins
            console.print(f"[yellow]{escape(shadowed)}[/yellow]")
    error = docker_error(members)
    if error:  # once, above the table, as `mnm ls` prints it
        console.print(f"[red]Docker could not be asked:[/red] {escape(error)}")
    table = Table(box=box.SIMPLE_HEAD)
    for column in ("Role", "Instance", "Type", "Port", "State", "Fits now"):
        table.add_column(column, overflow="fold")
    for m in members:
        style = VERDICT_STYLE[m.verdict.level]
        if m.missing:
            table.add_row(
                m.role, escape(m.name), "gone", "", "removed", f"[{style}]unknown[/{style}]"
            )
            continue
        state = state_cell(m.status) if m.status else "?"
        table.add_row(
            m.role,
            escape(m.name),
            m.recipe.modality.label,
            str(m.recipe.port),
            state,
            f"[{style}]{escape(m.verdict.words)}[/{style}]",
        )
    console.print(table)
    verdict = service.fit_together(members, snapshot, owner=service.owner(workflow))
    style = VERDICT_STYLE[verdict.level]
    console.print(
        f"Fit together: [{style}]{escape(verdict.words)}[/{style}]. {escape(verdict.note)}"
    )


DESK_NOTE = (
    "The desk: its roles ({roles}) are model ids every app can name; `mnm sets edit` swaps the "
    "model behind one."
)


def _desk_name(service: WorkflowService) -> str | None:
    try:
        return service.store.desk.name()
    except InstanceError:
        return None


@workflow_app.command("default")
def default(
    name: Annotated[str | None, typer.Argument(help="The set that becomes the desk.")] = None,
    clear: Annotated[bool, typer.Option("--clear", help="No desk: no role is a model id.")] = False,
) -> None:
    """The desk: the set whose roles (coder, chat, memory …) are model ids an app's
    config names, so the app survives a model swap. Without a name: which set it is."""
    service = _service()
    try:
        if clear:
            had = service.store.desk.clear()
            console.print("No desk now: no role is a model id." if had else "There was no desk.")
            return
        if name is not None:
            workflow = service.choose_desk(name)
        else:
            current = service.store.desk.name()
            if current is None:
                console.print(
                    "No desk yet: `mnm sets default <set>` makes a set's roles model ids "
                    "(`mnm sets create desk coder=<name> chat=<name> memory=<name>` first)."
                )
                return
            workflow = service.get(current)
    except InstanceError as exc:
        fail_with(exc)
    console.print(f"The desk is the set [bold]{escape(workflow.name)}[/bold]:")
    for role, instance in workflow.ordered():
        console.print(f"  {escape(role)} → {escape(instance)}")
    console.print(
        "[dim]Clients name the role as the model (`/v1/models` lists each, `meta.alias_of` its "
        "instance); `mnm url --for opencode|librechat|open-webui|openhands|goose|odysseus` "
        "writes the apps' files by role.[/dim]"
    )


@workflow_app.command("start")
def start(
    name: Annotated[str, typer.Argument()],
    anyway: Annotated[
        bool, typer.Option("--anyway", help="Start even when the members will not fit together.")
    ] = False,
    no_room: Annotated[
        bool,
        typer.Option(
            "--no-room",
            help="Park nothing for the set: refuse when mnm's other running models hold its room.",
        ),
    ] = False,
) -> None:
    """Start every member that is not running, the largest first, and print their URLs.
    When they do not fit, mnm's other running models are parked first, as the router would
    park them (never a pinned one or a member of a set in use)."""
    service = _service()
    try:
        workflow = service.get(name)
        with console.status(f"Starting {name}…") as status:
            done = service.start(workflow, spinner_progress(status), anyway, room=not no_room)
    except InstanceError as exc:
        fail_with(exc)
    if done.parked:
        console.print(escape(parked_for_room(done.parked, name)))
    for item in done.started:
        print_changes(item.fixes)
        for note in item.notes:
            console.print(f"[yellow]Note:[/yellow] {escape(note)}")
    console.print(f"[green]{escape(name)} is up.[/green]")
    _print_cards(service, workflow, Consumer.ODYSSEUS)


@workflow_app.command("stop")
def stop(name: Annotated[str, typer.Argument()]) -> None:
    """Stop every member that has a container; recipes and files stay."""
    service = _service()
    try:
        stopped = service.stop(service.get(name))
    except InstanceError as exc:
        fail_with(exc)
    if stopped:
        console.print(f"Stopped {escape(', '.join(stopped))}.")
    else:
        console.print(f"{escape(name)}: nothing was running.")


@workflow_app.command("rm")
def remove(name: Annotated[str, typer.Argument()]) -> None:
    """Remove the set; its instances stay."""
    service = _service()
    was_desk = _desk_name(service) == name
    try:
        service.remove(name)
    except InstanceError as exc:
        fail_with(exc)
    console.print(f"Removed set {escape(name)}; its instances are untouched.")
    if was_desk:
        console.print(
            "It was the desk: its roles are no model ids now (`mnm sets default <set>` names "
            "another), so an app that names one gets 404 until then."
        )


@workflow_app.command("url")
def url(
    name: Annotated[str, typer.Argument()],
    consumer: Annotated[
        Consumer, typer.Option("--for", help="Who pastes it: a chat client or a coding agent.")
    ] = Consumer.ODYSSEUS,
) -> None:
    """Every member's URLs for one harness, under its role."""
    service = _service()
    try:
        workflow = service.get(name)
        _print_cards(service, workflow, consumer)
    except InstanceError as exc:
        fail_with(exc)


def _print_cards(service: WorkflowService, workflow: Workflow, consumer: Consumer) -> None:
    for role, recipe, cards in service.cards(workflow, consumer):
        console.print(
            f"\n[bold]{escape(role)}[/bold]: {escape(recipe.name)} ({recipe.modality.label})"
        )
        for item in cards:
            console.print(f"  [bold]{escape(item.label)}[/bold]")
            if item.url:  # the stitch line has no address of its own
                print_block(f"    {item.url}")
            console.print(f"    [dim]{escape(item.note)}[/dim]")
            if item.snippet:
                print_block(item.snippet)
