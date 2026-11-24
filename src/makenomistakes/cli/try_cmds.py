"""`mnm try` (send an instance its card's request) and `mnm vram` (read the GPU)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from makenomistakes import try_run
from makenomistakes.cli._app import app, console, fail_with
from makenomistakes.cli.complete import instance_names
from makenomistakes.config import Settings
from makenomistakes.consumers import own_origin
from makenomistakes.gpu_peak import vram_readings
from makenomistakes.hub.facts import CHAT_LIKE, Engine
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.prefix_check import (
    PREFIX_TOKENS,
    PrefixError,
    check_prefix,
    prefix_lines,
    prefix_tokens,
    record,
)
from makenomistakes.recipes import Recipe
from makenomistakes.smoke import Picture, TryError
from makenomistakes.tool_check import Outcome, Verdict
from makenomistakes.try_all import HEADERS, Row, steps_for, try_all, write_report
from makenomistakes.try_run import TryInputs, TryRun, run_try

MIB = 2**20
TOOL_TONES = {Outcome.PASSED: "green", Outcome.FAILED: "red", Outcome.NOT_SUPPORTED: "yellow"}
VERDICT_TONES = {Verdict.WORKS: "green", Verdict.PARTIAL: "yellow", Verdict.NO: "red"}


def http_client() -> httpx.Client:
    """The try client for this box's settings (`try_run.http_client`); tests replace it."""
    return try_run.http_client(Settings())


def used_bytes() -> int | None:
    """GPU memory in use now (`try_run.used_bytes`); tests replace it."""
    return try_run.used_bytes()


@app.command(name="try")
def try_command(
    name: Annotated[
        str | None, typer.Argument(show_default=False, autocompletion=instance_names)
    ] = None,
    every: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Every saved instance, largest first, each alone (the rest parked; a pinned "
            "one or a set in use is left as it is and not tried); one table, also saved as "
            "try-all-<time>.txt here.",
        ),
    ] = False,
    audio: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="A clip to transcribe (speech-to-text instances).",
        ),
    ] = None,
    speed: Annotated[
        bool, typer.Option("--speed", help="Chat: 400 tokens, and llama.cpp's own tok/s.")
    ] = False,
    tools: Annotated[
        bool,
        typer.Option(
            "--tools",
            help="Chat, coding, vision: the tool-call test an agent depends on (one call, the "
            "tool's result, two calls at once, streamed, with thinking on); the verdict shows "
            "on the instance's page. `--all` always runs it on chat-like instances.",
        ),
    ] = False,
    measure: Annotated[
        bool, typer.Option("--measure", help="Sample GPU memory every 0.25 s during the request.")
    ] = False,
    agent: Annotated[
        bool,
        typer.Option(
            "--agent",
            help="Chat, coding: an agent's long prefix read cold, again warm, and (cache=on) "
            "after a park with its saved slot put back: what the prompt cache saves; kept as a "
            "reading. `--speed` is the decode rate.",
        ),
    ] = False,
    prefix: Annotated[
        int | None,
        typer.Option(
            "--prefix",
            min=512,
            max=200_000,
            help=f"With --agent: its tokens ({PREFIX_TOKENS:,}, or what one request's context "
            "holds).",
        ),
    ] = None,
    say: Annotated[
        str | None,
        typer.Option("--say", help="Text-to-speech: the sentence to read (default: the card's)."),
    ] = None,
    image: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="A picture: an image model repaints it, an editor edits it, a video model "
            "animates it (image-to-video).",
        ),
    ] = None,
    mask: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="With --image: white is redrawn, black kept (inpainting).",
        ),
    ] = None,
) -> None:
    """Send a running instance the request its curl card shows, and print what came back.

    Images are saved as <name>.png, clips as <name>.webm, speech as <name>.wav, in the
    current folder. `--all` tries every saved instance in turn (a chat-like one's tool calls
    too: the Tools column)."""
    service = InstanceService(Settings())
    if every:
        if name is not None or any(x is not None for x in (audio, say, image, mask)):
            fail_with(
                ValueError(
                    "--all sends each instance its card's own request (and a chat-like one the "
                    "tool-call test, --tools or not): leave out the name, --audio, --say, --image "
                    "and --mask (--speed applies; VRAM is always read)."
                )
            )
        _try_every(service, speed)
        return
    if name is None:
        fail_with(ValueError("Name the instance to try (`mnm ls` lists them), or pass --all."))
    try:
        recipe = service.get(name)
    except InstanceError as exc:
        fail_with(exc)
    if agent:  # Its own long prompt, three times
        if tools or any(x is not None for x in (audio, say, image, mask)):
            fail_with(ValueError("--agent sends its own long prompt: leave out the other inputs."))
        _try_prefix(service, recipe, prefix)
        return
    here = Path.cwd().resolve()
    for given, what in ((audio, "clip"), (image, "picture"), (mask, "mask")):
        if given is not None and not given.resolve().is_relative_to(here):
            # A coding agent may run `mnm try` unasked: it must not become a way to read files
            # outside the folder it works in (links are resolved first).
            fail_with(ValueError(f"The {what} must be in the current folder: {given}"))
    if mask is not None and image is None:
        fail_with(ValueError("A mask needs the picture it masks: add `--image <file>`."))
    picture = Picture(image.read_bytes(), mask.read_bytes() if mask else None) if image else None
    inputs = TryInputs(audio=audio, speed=speed, say=say, picture=picture, tools=tools)
    with http_client() as http:
        try:
            run = run_try(
                service, recipe, http, Path.cwd(), inputs, used_bytes if measure else None
            )
        except (TryError, InstanceError) as exc:
            fail_with(exc)
    _print_run(name, run)
    if run.tools is not None and run.tools.verdict is Verdict.NO:
        raise typer.Exit(1)  # no call an agent could run: a script reads the exit code


def _try_prefix(service: InstanceService, recipe: Recipe, asked: int | None) -> None:
    """`--agent`: the three legs, said, and kept as this box's reading."""
    if recipe.engine is not Engine.LLAMACPP or recipe.modality not in CHAT_LIKE:
        fail_with(
            ValueError(
                f"--agent reads a llama.cpp chat model's prompt cache; {recipe.name} is "
                f"{recipe.modality.value} on {recipe.engine.value}."
            )
        )
    try:
        tokens = prefix_tokens(recipe, asked)
    except PrefixError as exc:
        fail_with(exc)
    console.print(f"Sending {recipe.name} a prefix of about {tokens:,} tokens three times …")
    with http_client() as http:
        try:
            check = check_prefix(service, recipe, http, own_origin(recipe), tokens)
        except PrefixError as exc:
            fail_with(exc)
    record(service, recipe, check)
    for line in prefix_lines(check):
        console.print(escape(line))
    console.print("Kept as a reading of this box (`mnm calibration` shows it).")


def _print_run(name: str, run: TryRun) -> None:
    result = run.result
    tone = VERDICT_TONES[run.tools.verdict] if run.tools else "green"
    console.print(
        f"[{tone}]{escape(name)}[/{tone}] in {result.seconds:.2f} s: {escape(result.answer)}"
    )
    for step in run.tools.steps if run.tools else []:
        mark = TOOL_TONES[step.outcome]
        console.print(f"  [{mark}]{escape(step.step.value)}[/{mark}]: {escape(step.words)}")
    if run.vram_line is not None:
        console.print(run.vram_line)
    if run.company is not None:
        console.print(f"[dim]{escape(run.company)}[/dim]")
    if run.verdict is not None:
        console.print(escape(run.verdict))


def _try_every(service: InstanceService, speed: bool) -> None:
    """`mnm try --all`: the rows as they come, then the table, then the file."""
    here = Path.cwd()

    def said(row: Row) -> None:
        if row.why is None:
            console.print(f"{escape(row.name)}: [green]tried[/green]")
        else:
            console.print(f"{escape(row.name)}: [yellow]not tried[/yellow]: {escape(row.why)}")

    with http_client() as http, console.status("Trying every instance, one at a time…"):
        steps = steps_for(service, http, here, speed, used_bytes)
        try:
            rows = try_all(service, steps, said)
        except InstanceError as exc:
            fail_with(exc)
    if not rows:
        console.print("No instances: `mnm add <link>` saves one.")
        return
    table = Table(*HEADERS, box=box.SIMPLE)
    for row in rows:
        table.add_row(*(escape(cell) for cell in row.cells))
    console.print(table)
    path = write_report(here, rows)
    console.print(f"Saved as {escape(path.name)} in this folder.")


@app.command()
def vram(
    seconds: Annotated[int, typer.Option(help="How long to read for.", min=0, max=600)] = 10,
    every: Annotated[float, typer.Option(help="Seconds between readings.", min=0.1)] = 2.0,
) -> None:
    """GPU memory in use, read every `--every` seconds for `--seconds`, and the highest."""
    count = int(seconds // every) + 1
    readings = vram_readings(used_bytes, count, every)
    if not readings:
        fail_with(RuntimeError("nvidia-smi gave no reading: is the GPU visible (`mnm doctor`)?"))
    mib = [r // MIB for r in readings]
    console.print(f"{' '.join(f'{m:,}' for m in mib)} MiB in use; highest {max(mib):,} MiB")
