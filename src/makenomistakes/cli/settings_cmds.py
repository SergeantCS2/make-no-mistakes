"""`mnm show` and `mnm set`: inspect and change an instance's settings."""

from __future__ import annotations

import shlex
from typing import Annotated

import typer
from rich.markup import escape
from rich.table import Table

from makenomistakes.cli._app import app, console, fail_with, print_block
from makenomistakes.cli.complete import instance_names
from makenomistakes.cli.large_lines import PER_CARD_WORDS
from makenomistakes.config import Settings
from makenomistakes.consumers import instance_origin
from makenomistakes.controls import (
    SettingsError,
    current_value,
    parse_assignments,
    settings_rows,
)
from makenomistakes.engine_args import LlamaCppArgs
from makenomistakes.hub.facts import CHAT_LIKE
from makenomistakes.hub.kits import FAMILY_LABELS, ImageFamily
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.machine import planned_words
from makenomistakes.machine_facts import MachineFactsError, load_file
from makenomistakes.parts import Part, instance_parts
from makenomistakes.prompt_cache import words
from makenomistakes.readings_filter import left_out, store_here
from makenomistakes.recipes import AudioCppArgs, Recipe, docker_run_argv
from makenomistakes.recipes.audiocpp import voice_names
from makenomistakes.recipes.llamacpp_cache import keeps_slots
from makenomistakes.router_client import settings_saved
from makenomistakes.settings_notes import saved
from makenomistakes.slot_folder import SlotFolder
from makenomistakes.tune.apply import shown as tuned_lines


def _planned(service: InstanceService, recipe: Recipe) -> str | None:
    try:
        return planned_words(recipe, load_file(service.settings.expanded_state_dir()))
    except MachineFactsError:
        return None


def _chat_lines(service: InstanceService, recipe: Recipe) -> None:
    """A chat-like instance's: `mnm try --tools`' last verdict and its prompt cache
    (the caps, and what is saved now)."""
    console.print(f"  {escape(tools_line(service, recipe))}")
    if isinstance(recipe.args, LlamaCppArgs):
        saved_now = SlotFolder.of(service.models_dir, recipe.name).describe()
        said = words(recipe.args, keeps_slots(recipe))
        console.print(f"  prompt cache: {escape(said)}; {saved_now}")


@app.command()
def show(name: Annotated[str, typer.Argument(autocompletion=instance_names)]) -> None:
    """Show an instance's recipe, its settings with recommendations, and the exact command."""
    service = InstanceService(Settings())
    try:
        recipe = service.get(name)
    except InstanceError as exc:
        fail_with(exc)
    console.print(
        f"[bold]{escape(recipe.name)}[/bold]  {escape(recipe.repo_id)} @ {recipe.revision}  "
        f"({recipe.modality.label}, {escape(recipe.quant)}, {recipe.profile.value} profile, "
        f"port {recipe.port})"
    )
    console.print(f"  source: {recipe.source_url}")
    if recipe.image_family is not ImageFamily.UNKNOWN:  # what the engine's `Version:` should say
        console.print(f"  family: {FAMILY_LABELS[recipe.image_family]}")
    present = service.files_present(recipe)
    console.print(f"  downloaded: {'yes' if present else 'no'}")
    if recipe.modality in CHAT_LIKE:
        _chat_lines(service, recipe)
    if (planned := _planned(service, recipe)) is not None:  # The box it was planned on
        console.print(f"  planned on: {escape(planned)}")
    kept = service.measurements.readings(recipe.name)  # Another box's are only counted
    if others := left_out(kept, store_here(service.measurements)):
        console.print(
            f"  readings: {len(kept) - others} on this box; {others} on another machine, kept "
            "for when it returns"
        )
    for label, text in _tune_lines(recipe):
        console.print(f"  {label.lower()}: {escape(text)}")
    print_parts(instance_parts(recipe, service.models_dir))
    placement = recipe.llama_args.placement if isinstance(recipe.args, LlamaCppArgs) else None
    if placement is not None:  # Each card's blocks and experts
        console.print(f"  across cards, {PER_CARD_WORDS}:")
        for line in placement.lines():
            console.print(f"    {escape(line)}")
    if isinstance(recipe.args, AudioCppArgs) and not recipe.args.music:  # a request's `voice`
        console.print(f"  voices: {escape(', '.join(voice_names(recipe)))}")
    print_controls(recipe, service.native_context(recipe))
    running = service.status(recipe).running
    console.print(port_line(recipe))
    if present:
        try:
            argv = docker_run_argv(recipe, service.models_dir)
        except ValueError as exc:  # MNM_GPU is not a card number
            console.print(f"\n[yellow]{escape(str(exc))}[/yellow]")
            return
        what = "the engine runs now" if running else "the next `mnm start` runs"
        console.print(f"\n[dim]Exact command {what}:[/dim]")
        print_block(shlex.join(argv))  # quoted: a negative prompt has spaces


def tools_line(service: InstanceService, recipe: Recipe) -> str:
    found = service.tools.get(recipe.name)
    if found is None:
        return f"tools not tested (`mnm try {recipe.name} --tools`)"
    return f"{found.summary} ({found.when(recipe)})"


@app.command(name="set")
def set_(
    name: Annotated[str, typer.Argument(autocompletion=instance_names)],
    assignments: Annotated[list[str], typer.Argument(help="key=value … (see `mnm show`).")],
) -> None:
    """Change settings; they persist and apply on the next start."""
    service = InstanceService(Settings())
    try:
        recipe = service.get(name)
        updated = service.update_settings(recipe, parse_assignments(assignments))
    except (InstanceError, SettingsError) as exc:
        fail_with(exc)
    done = saved(service, recipe, updated)
    for control in done.changed:
        before, after = current_value(recipe, control), current_value(updated, control)
        console.print(  # values are the user's text: `[blurry]` is prompt syntax, not markup
            f"{control.key}: {escape(before) if before else '(empty)'} → "
            f"{escape(after) if after else '(empty)'}"
        )
    if {"idle", "keep"} & parse_assignments(assignments).keys():
        settings_saved(service.settings, name)  # the user's word wins over a client's
    for line in done.lines:
        text = escape(line.text).replace(f"`mnm start {name}`", f"[bold]mnm start {name}[/bold]")
        console.print(f"[yellow]{text}[/yellow]" if line.warn else text)


def port_line(recipe: Recipe) -> str:
    """Where the engine listens, and where clients reach it."""
    return (
        f"  port {recipe.port} on 127.0.0.1 only; clients reach it through the router at "
        f"{instance_origin(recipe)}/"
    )


def print_parts(parts: list[Part]) -> None:
    """Each file the next start loads: its part, size, where the engine keeps it."""
    table = Table(title="Parts", title_justify="left", box=None, pad_edge=False)
    for column, width in (("Part", 14), ("File", 18), ("Size", 9), ("Where", 10), ("From", 12)):
        table.add_column(column, min_width=width, overflow="fold")
    for part in parts:
        size = f"{part.size_bytes / 2**30:.2f} GiB" if part.size_bytes else "?"
        table.add_row(*(escape(v) for v in (part.role, part.file, size, part.place, part.repo_id)))
    console.print(table)


def print_controls(recipe: Recipe, native_ctx: int | None) -> None:
    table = Table(title="settings (mnm set <name> key=value)")
    for column in ("Key", "Current", "Recommended", "What it does"):
        table.add_column(column)
    for control, value, note in settings_rows(recipe, native_ctx):
        table.add_row(
            control.key,
            escape(value) if value else "[dim](empty)[/dim]",
            escape(f"{control.recommended}. {note}" if note else control.recommended),
            escape(control.help),
        )
    console.print(table)


def _tune_lines(recipe: Recipe) -> list[tuple[str, str]]:
    """The tuner's block, then the quality tune's choice."""
    tuned = tuned_lines(recipe.tuned) if recipe.tuned is not None else []
    return tuned + (recipe.quality.lines() if recipe.quality is not None else [])
