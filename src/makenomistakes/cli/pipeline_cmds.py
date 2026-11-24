"""`mnm pipeline`: mnm's own multi-step runs. One template so far: Short film."""

from __future__ import annotations

from typing import Annotated

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from makenomistakes.cli._app import app, console, fail_with, print_run_folder
from makenomistakes.cli.notify_cmd import notified
from makenomistakes.config import Settings
from makenomistakes.instances import InstanceService
from makenomistakes.pipeline import short_film as film
from makenomistakes.pipeline.runner import FilmRunner, PipelineError, film_runner
from makenomistakes.pipeline.short_film import FilmInputs
from makenomistakes.pipeline.steps import StepError
from makenomistakes.progress_words import with_ellipsis
from makenomistakes.runs import RunError, RunKind, RunRecord, RunState

pipeline_app = typer.Typer(help="Pipelines: multi-step runs over your instances (Short film).")
app.add_typer(pipeline_app, name="pipeline")


def _runner() -> FilmRunner:
    """The runner for this box; tests replace it."""
    return film_runner(InstanceService(Settings()))


def _said(run: RunRecord, runner: FilmRunner) -> None:
    """Where the run stands, and what to do next."""
    match run.state:
        case RunState.DONE:
            console.print(f"[green]Done[/green]: {escape(run.answer)}")
            made = run.step(film.FILM)
            name = made.file if made is not None and made.file else film.FILM_FILE
            console.print(escape(str(runner.store.folder(run) / name)))
        case RunState.FAILED:
            console.print(f"[red]{escape(run.error)}[/red]")
            console.print(f"Fix it, then `mnm pipeline resume {escape(run.id)}`.")
        case RunState.WAITING:
            console.print(f"Waiting for you at the {film.STAGE_LABELS[run.waiting_at or '']}.")
            if run.waiting_at == film.SHOTLIST:
                try:
                    shots = runner.shots(run).shots
                except StepError as exc:  # its folder cleaned away: said, never a traceback
                    console.print(f"[yellow]{escape(str(exc))}[/yellow]")
                    console.print(
                        f"`mnm pipeline redo {escape(run.id)} {film.SHOTLIST}` makes it again."
                    )
                    return
                for number, shot in enumerate(shots, start=1):
                    console.print(f"  {number}. {escape(shot.still_prompt)}")
                    console.print(f"     moves: {escape(shot.motion_prompt)}")
                    if shot.line:
                        console.print(f'     says: "{escape(shot.line)}"')
            else:
                folder = runner.store.folder(run)
                for step in run.steps:
                    if step.stage == run.waiting_at and step.file:
                        console.print(f"  {escape(str(folder / step.file))}")
            console.print(
                f"`mnm pipeline go {escape(run.id)}` goes on; `mnm pipeline redo "
                f"{escape(run.id)} {run.waiting_at} <n>` makes one again; the web's Runs → "
                "Pipelines edits the shots."
            )
        case RunState.RUNNING:
            console.print(f"{escape(run.id)} is running.")


def _progress(message: str) -> None:
    console.print(escape(with_ellipsis(message)))  # a step gets "…", a `Whole` line none


@pipeline_app.command("run")
@notified("pipeline run")
def run_cmd(
    idea: Annotated[str, typer.Argument(help="The film's idea, in a sentence or two.")],
    chat: Annotated[str, typer.Option(help="The chat instance that writes the shot list.")],
    image: Annotated[str, typer.Option(help="The image instance that draws each still.")],
    video: Annotated[
        str,
        typer.Option(
            help="The video instance that animates each still, or comfy:<workflow> for an "
            "imported ComfyUI workflow (a Wan 14B one, say)."
        ),
    ],
    voice: Annotated[
        str | None, typer.Option(help="A text-to-speech instance; none: a silent film.")
    ] = None,
    shots: Annotated[int, typer.Option(min=1, max=film.MAX_SHOTS)] = 3,
    style: Annotated[str, typer.Option(help="The look every still shares.")] = (
        "cinematic, soft light"
    ),
    auto: Annotated[
        bool, typer.Option("--auto", help="Run straight through; no stop at the review points.")
    ] = False,
    clip_size: Annotated[
        str | None,
        typer.Option(
            help="A comfy:<workflow>'s clip size, WxH (default 832x480); the stills are drawn "
            "at it. An instance's clips are its own size."
        ),
    ] = None,
    clip_frames: Annotated[
        int | None,
        typer.Option(help="A comfy:<workflow>'s frames per clip (default 33).", min=1, max=1000),
    ] = None,
) -> None:
    """Make a short film: a shot list, a still per shot, the lines, a clip per still, one film.
    It stops for you after the shot list and after the stills unless --auto."""
    runner = _runner()
    try:
        inputs = FilmInputs(
            idea=idea, shots=shots, style=style, chat=chat, image=image, video=video,
            voice=voice, auto=auto, clip_size=clip_size, clip_frames=clip_frames,
        )  # fmt: skip
        run = runner.start(inputs)
        print_run_folder(run.id, runner.store.folder(run))
        run = runner.advance(run.id, _progress)
    except (PipelineError, ValueError) as exc:
        fail_with(exc)
    _said(run, runner)


@pipeline_app.command("go")
@notified("pipeline go", key="run_id")
def go(run_id: Annotated[str, typer.Argument()]) -> None:
    """Past the review point the run waits at, and on."""
    runner = _runner()
    try:
        run = runner.proceed(run_id, _progress)
    except PipelineError as exc:
        fail_with(exc)
    _said(run, runner)


@pipeline_app.command("resume")
@notified("pipeline resume", key="run_id")
def resume(run_id: Annotated[str, typer.Argument()]) -> None:
    """Carry on a run that failed or was interrupted, from the first item not done."""
    runner = _runner()
    try:
        run = runner.advance(run_id, _progress)
    except PipelineError as exc:
        fail_with(exc)
    _said(run, runner)


@pipeline_app.command("redo")
@notified("pipeline redo", key="run_id")
def redo(
    run_id: Annotated[str, typer.Argument()],
    stage: Annotated[str, typer.Argument(help="shotlist, stills, lines or clips.")],
    number: Annotated[int, typer.Argument(help="Which one (1 = the first shot).", min=1)] = 1,
) -> None:
    """Make one item again (and the film); a still at the stills review stays there."""
    if stage not in film.STAGES or stage == film.FILM:
        fail_with(ValueError(f"'{stage}' is not one of: shotlist, stills, lines, clips."))
    runner = _runner()
    try:
        run = runner.redo(run_id, stage, number - 1, _progress)
    except PipelineError as exc:
        fail_with(exc)
    _said(run, runner)


@pipeline_app.command("ls")
def list_cmd() -> None:
    """The pipeline runs, newest first."""
    runner = _runner()
    table = Table(box=box.SIMPLE_HEAD)
    for column in ("Run", "Idea", "State"):
        table.add_column(column)
    for run in runner.store.all():
        if run.kind is RunKind.PIPELINE and run.parent is None:  # a song's mixes: under it
            state = run.state.value + (f" at {run.waiting_at}" if run.waiting_at else "")
            state += " (interrupted)" if run.interrupted else ""
            table.add_row(escape(run.id), escape((run.prompt or "")[:60]), state)
    console.print(table)


@pipeline_app.command("show")
def show(run_id: Annotated[str, typer.Argument()]) -> None:
    """One run's steps: each item, its state and its file."""
    runner = _runner()
    try:
        run = runner.store.get(run_id)
    except RunError as exc:
        fail_with(exc)
    table = Table(box=box.SIMPLE_HEAD)
    for column in ("Step", "State", "File", "Seconds"):
        table.add_column(column)
    for step in run.steps:
        label = film.STAGE_LABELS.get(step.stage, step.stage)
        many = step.stage in (film.STILLS, film.LINES, film.CLIPS)
        table.add_row(
            f"{label} {step.index + 1}" if many else label,
            step.state.value + (f": {escape(step.error)}" if step.error else ""),
            escape(step.file or "-"),
            f"{step.seconds:.0f}" if step.seconds else "",
        )
    console.print(table)
    _said(run, runner)
