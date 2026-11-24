"""`mnm studio trainer install|status|rollback|uninstall` and `mnm studio train voice|ls|show|
cancel`: the trainer's image, built on this box, and the voices it trains from
the user's recordings, run by the router when the card is free."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from makenomistakes.cli._app import build_progress, console, fail_with
from makenomistakes.cli.image_cmds import remove_all, roll_back, show_status
from makenomistakes.cli.notify_cmd import notified
from makenomistakes.cli.studio_cmds import studio_app
from makenomistakes.config import Settings
from makenomistakes.hardware.runner import SubprocessRunner
from makenomistakes.local_time import local
from makenomistakes.router_client import router_train, router_training
from makenomistakes.runs import RunError, RunKind, RunRecord, RunStore
from makenomistakes.studio import train_jobs
from makenomistakes.studio.built_image import BuiltImage
from makenomistakes.studio.style_trainer import STYLE, StyleTrainer, box_style_trainer
from makenomistakes.studio.train_data import TrainDataError
from makenomistakes.studio.train_kinds import EPOCH_SECONDS, OUTSIDE, PROBED
from makenomistakes.studio.trainer import (
    CARRIES,
    IMAGE_GIB,
    VARIANT_WORDS,
    StudioTrainer,
    TrainerError,
    TrainingContainer,
    Variant,
    box_trainer,
    image_tag,
    install_variant,
    read_progress,
    variant_of,
)
from makenomistakes.studio.voice_library import SOURCE_WORDS, VoiceError, VoiceSource

GB = 1_000_000_000


class For(StrEnum):
    """Which trainer an image command means: Applio's voices or ACE-Step's styles."""

    VOICES = "voices"
    STYLES = "styles"


FOR = Annotated[For, typer.Option("--for", help="voices (Applio) or styles (ACE-Step).")]

trainer_app = typer.Typer(help="The trainer's image: Applio, built on this box.")
studio_app.add_typer(trainer_app, name="trainer")
train_app = typer.Typer(
    help="Train a voice from recordings or a style from songs (the router runs it)."
)
studio_app.add_typer(train_app, name="train")


def _trainer(settings: Settings) -> StudioTrainer:
    """This box's voice trainer (tests replace it)."""
    return box_trainer(settings)


def _style_trainer(settings: Settings) -> StyleTrainer:
    """This box's style trainer (tests replace it; `studio_style_cmds` calls it here)."""
    return box_style_trainer(settings)


def _image(settings: Settings, which: For) -> BuiltImage:
    """The image an image command means: the two seams above, one per kind."""
    return _style_trainer(settings) if which is For.STYLES else _trainer(settings)


@trainer_app.command("install")
@notified("studio trainer install")
def install(
    cpu: Annotated[
        bool, typer.Option("--cpu", help="Build it for the processor (no card: hours to days).")
    ] = False,
    ca: Annotated[
        Path | None,
        typer.Option(
            exists=True, dir_okay=False, help="A proxy's CA certificate, for a build behind one."
        ),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask first.")] = False,
    which: FOR = For.VOICES,
) -> None:
    """Build a trainer's image on this box: for voices, Applio at a pinned release with its
    weights checked by sha256; for styles (`--for styles`), ACE-Step 1.5's own trainer at a
    pinned commit (its checkpoint is downloaded when a style is trained). PyTorch for this
    card, or the processor."""
    if which is For.STYLES:
        from makenomistakes.cli.studio_style_cmds import install_styles  # noqa: PLC0415

        install_styles(cpu, ca, yes)
        return
    try:
        variant = install_variant(cpu)
    except TrainerError as exc:
        fail_with(exc)
    trainer = StudioTrainer(Settings(), variant)
    console.print(
        f"This builds {escape(image_tag(variant))} for {VARIANT_WORDS[variant]}: about "
        f"{IMAGE_GIB[variant]:.0f} GB on disk, 10-30 minutes; it downloads Applio from GitHub at "
        "a commit checked, packages from PyPI and PyTorch's index and Applio's weights from "
        "Hugging Face, each file checked by sha256. It carries:"
    )
    for line in CARRIES:
        console.print(f"  {escape(line)}")
    if not yes and not typer.confirm("Build it now?"):
        raise typer.Exit(code=1)
    try:
        ledger = trainer.install(build_progress, ca)
    except TrainerError as exc:
        fail_with(exc)
    size = f", {ledger.size_bytes / GB:.1f} GB" if ledger.size_bytes else ""
    console.print(
        f"[green]Built[/green] {escape(ledger.image)}{size}. `mnm studio train voice <name> "
        "--from <folder> --source own` trains a voice from your recordings."
    )


@trainer_app.command("status")
def status(which: FOR = For.VOICES) -> None:
    """A trainer's image: which one, for what, and whether this mnm would build another."""
    trainer = _image(Settings(), which)
    try:
        ledger = trainer.ledger()
    except TrainerError:
        ledger = None  # said by show_status
    variant = (variant_of(ledger.image) if ledger else None) or Variant.CPU
    show_status(trainer, trainer.image_tag(), f"for {VARIANT_WORDS[variant]}")


@trainer_app.command("rollback")
def rollback(which: FOR = For.VOICES) -> None:
    """Go back to the image before the last install."""
    roll_back(_image(Settings(), which))


@trainer_app.command("uninstall")
def uninstall(
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask first.")] = False,
    which: FOR = For.VOICES,
) -> None:
    """Remove every image of a trainer; what it trained and the trainings' logs stay."""
    kept = (
        "the styles it trained stay in the library"
        if which is For.STYLES
        else ("the voices it trained stay in the library")
    )
    remove_all(_image(Settings(), which), yes, kept)


@train_app.command("voice")
def train_voice(
    name: Annotated[str, typer.Argument(help="The voice's name in the library.")],
    folder: Annotated[
        Path,
        typer.Option(
            "--from", exists=True, file_okay=False, help="A folder of the voice's recordings."
        ),
    ],
    source: Annotated[
        VoiceSource, typer.Option(help="Whose voice: own, hired, licensed, likeness, generated.")
    ],
    consent: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="The singer's consent (only its hash)."),
    ] = None,
    rate: Annotated[int, typer.Option(help="32000, 40000 or 48000.")] = train_jobs.DEFAULT_RATE,
    epochs: Annotated[int, typer.Option(help="Passes over the recordings.")] = (
        train_jobs.DEFAULT_EPOCHS
    ),
    batch: Annotated[int, typer.Option(help="Chunks per step.")] = train_jobs.DEFAULT_BATCH,
    licence: Annotated[str, typer.Option(help="The terms, in your words.")] = "",
    now: Annotated[
        bool, typer.Option("--now", help="Start without the router's idle minute.")
    ] = False,
) -> None:
    """Train an RVC voice from a folder of recordings (10 to 60 minutes of the voice alone,
    dry, WAV or FLAC is what RVC voices are trained on). The router runs it while nobody uses
    the card; a request pauses it, and it resumes from its last checkpoint."""
    settings = Settings()
    try:
        queued = train_jobs.queue_voice(
            settings, _trainer(settings), name, folder, source,
            consent=consent, sample_rate=rate, epochs=epochs, batch_size=batch, licence=licence,
        )  # fmt: skip
    except (train_jobs.TrainError, TrainDataError, TrainerError, VoiceError, RunError) as exc:
        fail_with(exc)
    run = queued.run
    console.print(f"Recordings: {escape(queued.survey.words)}, copied into {run.id}.")
    for skipped in queued.survey.skipped:
        console.print(f"[dim]Left out: {escape(skipped)}[/dim]")
    for warning in queued.survey.warnings:
        console.print(f"[yellow]{escape(warning)}[/yellow]")
    console.print(f"Voice: {escape(SOURCE_WORDS[source])}; {escape(queued.estimate)}.")
    answer = router_train(settings, run.id, json.dumps({"now": now}).encode())
    if answer is None:
        console.print(
            "No router answers on this machine: the training waits for it (`mnm service "
            "status`); it starts once the router runs and has been idle a minute."
        )
    else:
        console.print(escape(answer[1]))


@train_app.command("ls")
def train_ls() -> None:
    """The trainings: queued, running (with the epoch), paused, done or failed."""
    settings = Settings()
    store = RunStore.for_settings(settings)
    runs = train_jobs.trainings(store)
    if not runs:
        console.print(
            "No trainings yet: `mnm studio train voice <name> --from <folder> …` or "
            "`mnm studio train style <name> --from <folder> …`."
        )
        return
    training = router_training(settings)
    table = Table(box=box.SIMPLE_HEAD)
    for column in ("Run", "Name", "Kind", "State", "Asked", "Epochs"):
        table.add_column(column)
    for run in runs:
        table.add_row(
            escape(run.id),
            escape(run.target),
            train_jobs.kind_of(run),
            escape(train_jobs.state_words(store, run, training)),
            f"{local(run.created_at):%Y-%m-%d %H:%M}",
            run.settings.get("epochs", ""),
        )
    console.print(table)


@train_app.command("show")
def train_show(run_id: Annotated[str, typer.Argument(help="The training's run id.")]) -> None:
    """One training: its settings, recordings, warnings, progress and outcome."""
    settings = Settings()
    store = RunStore.for_settings(settings)
    run = _training(store, run_id)
    kind = train_jobs.kind_of(run)
    console.print(
        f"{kind.title()}: {escape(run.target)} ({escape(run.settings.get('source', ''))})"
    )
    state = train_jobs.state_words(store, run, router_training(settings))
    console.print(f"State: {escape(state)}")
    keys = train_jobs.KEYS + train_jobs.STYLE_KEYS + (PROBED, OUTSIDE, EPOCH_SECONDS)
    shown = {k: v for k, v in run.settings.items() if k in keys}
    console.print("Settings: " + escape(", ".join(f"{k} {v}" for k, v in shown.items())))
    what = "Songs" if kind == STYLE else "Recordings"
    console.print(f"{what}: {len(run.inputs_recorded)}, each kept by sha256 on the {kind}")
    for note in run.notes:
        console.print(f"[yellow]{escape(note)}[/yellow]")
    progress = read_progress(store.folder(run))
    if progress is not None:
        console.print(f"Last stage: {escape(progress.stage)}")
    if run.answer:
        console.print(f"[green]{escape(run.answer)}[/green]")
    if run.error:
        console.print(f"[red]{escape(run.error)}[/red]")
    console.print(f"Folder: {escape(str(store.folder(run)))}")


@train_app.command("cancel")
def train_cancel(run_id: Annotated[str, typer.Argument(help="The training's run id.")]) -> None:
    """Stop a training (running or queued); its folder stays until `mnm clean`."""
    settings = Settings()
    store = RunStore.for_settings(settings)
    run = _training(store, run_id)
    answer = router_train(settings, run.id, json.dumps({"action": "cancel"}).encode())
    if answer is not None:
        if answer[0] >= 400:
            fail_with(train_jobs.TrainError(answer[1]))
        console.print(escape(answer[1]))
        return
    if run.going and run.pid is not None:
        fail_with(
            train_jobs.TrainError(
                f"The training {run.id} runs in another process ({run.pid}): ask the router."
            )
        )
    if not (run.going or run.interrupted):
        console.print(f"The training {run.id} has ended ({run.state.value}).")
        return
    if run.interrupted:  # a router took it and stopped: its container may hold the card still
        container = _container()
        container.stop(run.id)
        container.finished(run.id)
    store.fail(run, train_jobs.CANCELLED)
    console.print(f"The training of {run.target} is cancelled; its folder stays until `mnm clean`.")


def _container() -> TrainingContainer:
    return TrainingContainer(SubprocessRunner())


def _training(store: RunStore, run_id: str) -> RunRecord:
    try:
        run = store.get(run_id)
    except RunError as exc:
        fail_with(exc)
    if run.kind is not RunKind.TRAINING:
        fail_with(train_jobs.TrainError(f"{run_id} is not a training (`mnm studio train ls`)."))
    return run
