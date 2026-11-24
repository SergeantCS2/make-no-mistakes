"""`mnm studio train style` and `mnm studio styles ls|show|add|rm`: a style taught to
ACE-Step from the user's songs by the router's trainer, and the library of styles an ACE-Step
instance plays (`mnm set <instance> style=<name>`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from makenomistakes.cli import studio_train_cmds
from makenomistakes.cli._app import build_progress, console, fail_with
from makenomistakes.cli.studio_cmds import studio_app
from makenomistakes.cli.studio_train_cmds import train_app
from makenomistakes.config import Settings
from makenomistakes.hardware.probe import probe_memory
from makenomistakes.local_time import local
from makenomistakes.router_client import router_train
from makenomistakes.runs import RunError
from makenomistakes.studio import style_data, style_jobs, style_use
from makenomistakes.studio.style_library import (
    SOURCE_WORDS,
    StyleError,
    StyleLibrary,
    StyleSource,
)
from makenomistakes.studio.style_trainer import CARRIES as STYLE_CARRIES
from makenomistakes.studio.style_trainer import IMAGE_GIB as STYLE_GIB
from makenomistakes.studio.style_trainer import (
    StyleTrainer,
    StyleTrainerError,
    style_variant,
)
from makenomistakes.studio.train_data import TrainDataError
from makenomistakes.studio.train_jobs import TrainError
from makenomistakes.studio.trainer import VARIANT_WORDS

GIB = 2**30
styles_app = typer.Typer(help="The style library: LoRAs that teach ACE-Step a style.")
studio_app.add_typer(styles_app, name="styles")


def _machine() -> tuple[float | None, float | None]:
    """The card's and the RAM's size in GiB, each None when unread (tests replace it)."""
    snapshot = probe_memory()
    gpu, memory = snapshot.gpu, snapshot.memory
    return (
        gpu.vram_total_bytes / GIB if gpu is not None else None,
        memory.total_bytes / GIB if memory is not None else None,
    )


def _library(settings: Settings) -> StyleLibrary:
    return StyleLibrary(settings.expanded_state_dir(), settings.expanded_models_dir())


def install_styles(cpu: bool, ca: Path | None, yes: bool) -> None:
    """`mnm studio trainer install --for styles`: ACE-Step 1.5's trainer built for this box."""
    try:
        variant = style_variant(cpu)
    except StyleTrainerError as exc:
        fail_with(exc)
    trainer = StyleTrainer(Settings(), variant)
    console.print(
        f"This builds {escape(trainer.image_tag())} for {VARIANT_WORDS[variant]}: about "
        f"{STYLE_GIB[variant]:.0f} GB on disk, 10-30 minutes; it fetches ACE-Step 1.5 from "
        "GitHub at a commit checked and packages from PyPI and PyTorch's index, each file "
        "checked by sha256. The checkpoint it trains is downloaded when a style is trained "
        "(about 6 GB, once). It carries:"
    )
    for line in STYLE_CARRIES:
        console.print(f"  {escape(line)}")
    if not yes and not typer.confirm("Build it now?"):
        raise typer.Exit(code=1)
    try:
        ledger = trainer.install(build_progress, ca)
    except StyleTrainerError as exc:
        fail_with(exc)
    size = f", {ledger.size_bytes / 1e9:.1f} GB" if ledger.size_bytes else ""
    console.print(
        f"[green]Built[/green] {escape(ledger.image)}{size}. `mnm studio train style <name> "
        '--from <folder> --caption "…" --source own` teaches ACE-Step a style from your songs.'
    )


@train_app.command("style")
def train_style(
    name: Annotated[str, typer.Argument(help="The style's name in the library.")],
    folder: Annotated[
        Path,
        typer.Option("--from", exists=True, file_okay=False, help="A folder of the songs."),
    ],
    caption: Annotated[str, typer.Option(help="What the songs sound like: the style, in words.")],
    source: Annotated[StyleSource, typer.Option(help="Whose songs: own, licensed, generated.")],
    trigger: Annotated[
        str | None, typer.Option(help="The word that calls the style up (the name if unset).")
    ] = None,
    variant: Annotated[
        Literal["base", "turbo"], typer.Option(help="The ACE-Step build it plays on.")
    ] = "base",
    epochs: Annotated[int, typer.Option(help="Passes over the songs.")] = style_jobs.DEFAULT_EPOCHS,
    max_seconds: Annotated[
        float, typer.Option("--max-seconds", help="Each song is cut here (30 to 600).")
    ] = style_data.DEFAULT_MAX_SECONDS,
    licence: Annotated[str, typer.Option(help="The terms, in your words.")] = "",
    now: Annotated[
        bool, typer.Option("--now", help="Start without the router's idle minute.")
    ] = False,
) -> None:
    """Teach ACE-Step a style from a folder of songs (five or more, 30 s to 4 min, WAV or FLAC
    keep the sound; a `<song>.txt` beside a song is its lyrics). It downloads ACE-Step's
    checkpoint for the build (about 6 GB, once); the router trains while nobody uses the
    card, its first epoch measuring the card's real peak; a request pauses it."""
    settings = Settings()
    card_gib, ram_gib = _machine()
    try:
        queued = style_jobs.queue_style(
            settings, studio_train_cmds._style_trainer(settings), name, folder, source,
            caption=caption, card_gib=card_gib, ram_gib=ram_gib, trigger=trigger, variant=variant,
            epochs=epochs, max_seconds=max_seconds, licence=licence,
            say=lambda line: console.print(escape(line)),
        )  # fmt: skip
    except (TrainError, TrainDataError, StyleTrainerError, StyleError, RunError) as exc:
        fail_with(exc)
    run = queued.run
    console.print(f"Songs: {escape(queued.survey.words)}, copied into {run.id}.")
    for skipped in queued.survey.skipped:
        console.print(f"[dim]Left out: {escape(skipped)}[/dim]")
    for warning in queued.survey.warnings:
        console.print(f"[yellow]{escape(warning)}[/yellow]")
    console.print(
        f"Style: {escape(SOURCE_WORDS[source])}, trigger '{escape(run.settings['trigger'])}'; "
        f"{escape(queued.estimate)}."
    )
    answer = router_train(settings, run.id, json.dumps({"now": now}).encode())
    if answer is None:
        console.print(
            "No router answers on this machine: the training waits for it (`mnm service "
            "status`); it starts once the router runs and has been idle a minute."
        )
    else:
        console.print(escape(answer[1]))


@styles_app.command("ls")
def styles_ls() -> None:
    """The styles: each one's build, trigger word and whose songs it learned from."""
    library = _library(Settings())
    rows = library.all()
    for problem in library.problems():
        console.print(f"[red]{escape(problem)}[/red]")
    if not rows:
        console.print("No styles yet: `mnm studio train style <name> --from <folder> …`.")
        return
    table = Table(box=box.SIMPLE_HEAD)
    for column in ("Style", "Build", "Trigger", "Rank", "From", "Added"):
        table.add_column(column)
    for row in rows:
        table.add_row(
            escape(row.name), row.variant, escape(row.trigger), str(row.rank),
            escape(SOURCE_WORDS[row.source]), f"{local(row.created_at):%Y-%m-%d}",
        )  # fmt: skip
    console.print(table)


@styles_app.command("show")
def styles_show(name: Annotated[str, typer.Argument(help="The style.")]) -> None:
    """One style: its files by hash, how it was trained, how to play it."""
    library = _library(Settings())
    try:
        row = library.get(name)
    except StyleError as exc:
        fail_with(exc)
    console.print(escape(row.words()))
    console.print(f"  folder: {escape(str(library.folder(row.name)))}")
    for file in row.files:
        console.print(f"  {file.name} (sha256 {file.sha256[:16]}…)")
    if row.caption:
        console.print(f"  caption: {escape(row.caption)}")
    if row.training is not None:
        songs = len(row.training.dataset)
        console.print(f"  trained: {row.training.epochs} epochs on {songs} songs, rank {row.rank}")
    if missing := library.missing_files(row):
        console.print(f"[yellow]  missing: {escape(', '.join(missing))}[/yellow]")
    console.print(
        f"  plays on ACE-Step 1.5 {row.variant}: `mnm set <instance> style={row.name}` "
        "(applies at its next start)."
    )


@styles_app.command("add")
def styles_add(
    name: Annotated[str, typer.Argument(help="The style's name in the library.")],
    adapter: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="A PEFT adapter's folder."),
    ],
    variant: Annotated[Literal["base", "turbo"], typer.Option(help="The build it was trained on.")],
    trigger: Annotated[str, typer.Option(help="The word its training put in front.")],
    source: Annotated[StyleSource, typer.Option(help="Whose songs: own, licensed, generated.")],
    licence: Annotated[str, typer.Option(help="The terms, in your words.")] = "",
) -> None:
    """Add an ACE-Step LoRA trained elsewhere (PEFT's `adapter_model.safetensors` and
    `adapter_config.json`), checked as audio.cpp loads it."""
    settings = Settings()
    try:
        row = _library(settings).add(name, adapter, variant, trigger, source, licence=licence)
    except StyleError as exc:
        fail_with(exc)
    console.print(f"Added {escape(row.words())}.")


@styles_app.command("rm")
def styles_rm(
    name: Annotated[str, typer.Argument(help="The style.")],
    keep_files: Annotated[
        bool, typer.Option("--keep-files", help="Keep its folder (only the row goes).")
    ] = False,
) -> None:
    """Remove a style from the library (its files too unless `--keep-files`)."""
    try:
        settings = Settings()
        style_use.refuse_removal(settings, name)
        row = _library(settings).remove(name, files=not keep_files)
    except StyleError as exc:
        fail_with(exc)
    console.print(f"Removed {escape(row.name)}" + (" (its files stay)." if keep_files else "."))
