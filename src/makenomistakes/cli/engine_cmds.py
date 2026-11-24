"""`mnm engines`: which engine builds this box runs, update them, roll one back."""

from __future__ import annotations

from typing import Annotated

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from makenomistakes import engine_check, engines
from makenomistakes.cli._app import Failed, app, console, fail_with
from makenomistakes.cli.notify_cmd import notified
from makenomistakes.config import Settings
from makenomistakes.engine_selftest import Outcome
from makenomistakes.engine_update import FAILURES, Line, Tone, self_test, update_engines
from makenomistakes.hardware.probe import SubprocessRunner
from makenomistakes.hub.facts import Engine
from makenomistakes.instances import InstanceService
from makenomistakes.supervisor import DockerSupervisor

engines_app = typer.Typer(
    help="Engine images (llama.cpp, stable-diffusion.cpp, whisper.cpp, audio.cpp, vLLM): status, "
    "check (a newer build upstream?), update (which tests the new build), test, rollback.",
    invoke_without_command=True,
)
app.add_typer(engines_app, name="engines")


def _docker() -> DockerSupervisor:
    return DockerSupervisor(SubprocessRunner())


@engines_app.callback()
def status(ctx: typer.Context) -> None:
    """Show each engine image: pulled or not, its build, which instances use it."""
    if ctx.invoked_subcommand is not None:
        return
    settings = Settings()
    try:
        ledger = engines.load_ledger(settings.expanded_state_dir())
    except engines.EngineError as exc:  # status still renders; rollback is just unknown
        console.print(f"[yellow]{escape(str(exc))}[/yellow]")
        ledger = engines.EngineLedger()
    try:
        rows = engines.engine_status(_docker(), InstanceService(settings).all(), ledger)
    except FAILURES as exc:
        fail_with(exc)
    # Every official image is on ghcr.io; without it publisher, name and tag fit unwrapped.
    table = Table(title="engines", box=box.SIMPLE_HEAD)
    for column in ("Engine", "Image", "Backend", "Build", "Created"):
        table.add_column(column, no_wrap=True)
    table.add_column("Instances")
    table.add_column("Rollback", no_wrap=True)
    for row in rows:
        table.add_row(
            row.engine.value,
            escape(row.image.removeprefix("ghcr.io/")),
            row.backend,
            row.local.short_id if row.local else "[dim]not pulled[/dim]",
            row.local.created_day if row.local else "",
            escape(", ".join(row.instances)) or "[dim]none[/dim]",
            "yes" if row.can_roll_back else "",
        )
    console.print(escape(engines.box_words()))  # The backend the rows are for
    console.print(table)
    console.print(
        "[dim]Builds change only when you run `mnm engines update`; "
        "`mnm engines rollback <engine>` undoes the last one.[/dim]"
    )


@engines_app.command("update")
@notified("engines update")
def update_cmd(
    which: Annotated[
        list[Engine] | None,
        typer.Argument(help="Engines to update (default: every engine this box has or uses)."),
    ] = None,
) -> None:
    """Pull the newest build of each engine image; the previous build is kept for rollback."""
    settings = Settings()
    service = InstanceService(settings)
    try:
        failed = update_engines(
            service, _docker(), settings.expanded_state_dir(), set(which or []), _say
        )
    except FAILURES as exc:
        fail_with(exc)
    if failed:
        raise Failed("an engine's new build failed its test and was rolled back", code=1)


_TONE_STYLE = {Tone.OK: "green", Tone.BAD: "red", Tone.WARN: "yellow", Tone.DIM: "dim"}


def _say(line: Line) -> None:
    """One line of the update's story, coloured for the terminal."""
    indent = "  " if line.indent else ""
    style = _TONE_STYLE.get(line.tone)
    head = f"[{style}]{escape(line.head)}[/{style}]" if style else escape(line.head)
    console.print(f"{indent}{head}{escape(line.rest)}")


@engines_app.command("check")
def check_cmd(
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Whether the registry holds a newer build of each engine image, never pulled,
    and whether it runs a saved instance the installed build lacks."""
    settings = Settings()
    try:
        recipes = InstanceService(settings).all()
        report = engine_check.check_engines(_docker(), recipes)
    except FAILURES as exc:
        fail_with(exc)
    engine_check.save_report(settings.expanded_state_dir(), report)
    if as_json:
        print(report.model_dump_json(indent=2))
        return
    table = Table(title="engines against the registry", box=box.SIMPLE_HEAD)
    for column in ("Engine", "This box", "Registry", "State"):
        table.add_column(column, no_wrap=True)
    table.add_column("Why")
    for row in report.rows:
        table.add_row(
            row.engine.value,
            " ".join(filter(None, (row.installed_build, row.installed_day))) or "-",
            row.upstream_day or "-",
            f"[{_STATE_STYLE[row.state]}]{row.state.value}[/]",
            escape(row.why or ""),
        )
    console.print(table)
    for line in report.instances:
        console.print(f"[yellow]{escape(line.line)}[/yellow]")
    for row in report.newer:
        console.print(f"{escape(row.words)}: [bold]{engine_check.update_command(row.engine)}[/]")
    if report.newer:
        console.print(f"[dim]{escape(engine_check.UPDATE_WORDS)}.[/dim]")
    elif not any(row.state is engine_check.State.UNKNOWN for row in report.rows):
        console.print("Every pulled engine is the registry's newest build.")


_STATE_STYLE = {
    engine_check.State.NEWER: "yellow",
    engine_check.State.CURRENT: "green",
    engine_check.State.NOT_PULLED: "dim",
    engine_check.State.UNKNOWN: "red",
}


@engines_app.command("test")
def selftest_cmd(
    engine: Annotated[Engine, typer.Argument(help="The engine whose current build to test.")],
) -> None:
    """Test an engine's current build with a copy of its smallest downloaded instance,
    against that instance's readings (the ledger's other build may be one a rollback
    rejected, so it never sets the bar here)."""
    settings = Settings()
    service = InstanceService(settings)
    image = engines.image_for(engine.value)
    state_dir = settings.expanded_state_dir()
    result = self_test(service, engine, image, state_dir, None, _say, roll_back=False)
    if result.outcome is Outcome.FAILED:
        raise typer.Exit(code=1)


@engines_app.command("rollback")
def rollback_cmd(
    target: Annotated[
        str,
        typer.Argument(
            help="llamacpp | sdcpp | whispercpp | audiocpp | vllm, or an image a recipe names"
        ),
    ],
) -> None:
    """Point an engine's tag back at the build the last `mnm engines update` replaced."""
    settings = Settings()
    try:
        done = engines.rollback(_docker(), engines.image_for(target), settings.expanded_state_dir())
    except FAILURES as exc:
        fail_with(exc)
    was = f"was {done.replaced.short_id}" if done.replaced else "the tag was missing"
    console.print(
        f"[green]{escape(done.image)}[/green] now points at {done.restored.short_id} ({was}). "
        "Restart the instances that use it to switch."
    )
