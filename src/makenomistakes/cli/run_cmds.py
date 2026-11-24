"""`mnm run` (the user's prompt to an instance, kept), `mnm outputs` (the runs and their
files) and `mnm clean` (empty what runs and engines left behind)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from makenomistakes import try_run
from makenomistakes.cleanup import AUTO, Cleaner, Target, parse_age
from makenomistakes.cli._app import app, console, fail_with, print_run_folder
from makenomistakes.cli.complete import instance_names
from makenomistakes.cli.notify_cmd import notified
from makenomistakes.comfy.engine import ComfyEngine
from makenomistakes.config import Settings
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.make import (
    MAX_REPEAT,
    Ask,
    close_parent,
    fixed_seed,
    make,
    open_parent,
    takes_candidates,
)
from makenomistakes.nightly import settings as nightly_settings
from makenomistakes.run_prune import select
from makenomistakes.runs import RunError, RunRecord, RunState, RunStore, size_limit
from makenomistakes.sizes import size_words
from makenomistakes.smoke import TryError

outputs_app = typer.Typer(help="What runs made: list, show, delete, prune.")
app.add_typer(outputs_app, name="outputs")


def _store() -> RunStore:
    return RunStore.for_settings(Settings())


RepeatOption = Annotated[
    int,
    typer.Option(
        "--repeat",
        min=1,
        max=MAX_REPEAT,
        help=f"Make it N times (1 to {MAX_REPEAT}), one run each.",
    ),
]


def print_repeats(made: list[RunRecord], store: RunStore) -> None:
    """The N runs together: their folders, in the order they were made."""
    console.print(f"{len(made)} runs (`mnm outputs ls` lists them; the gallery shows them):")
    for run in made:
        console.print(f"  {escape(run.id)}  {escape(str(store.folder(run)))}", soft_wrap=True)


def _given(path: Path | None, what: str) -> tuple[str, bytes] | None:
    """A file from the current folder only (as `mnm try` takes them: links resolved first)."""
    if path is None:
        return None
    if not path.resolve().is_relative_to(Path.cwd().resolve()):
        fail_with(ValueError(f"The {what} must be in the current folder: {path}"))
    return (path.name, path.read_bytes())


def _lyrics(path: Path | None) -> str | None:
    """The lyrics file's text (a file anywhere: it is read, never kept as a file)."""
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        fail_with(ValueError(f"Cannot read the lyrics in {path}: {exc}"))


def _age(text: str | None) -> timedelta | None:
    if text is None:
        return None
    try:
        return parse_age(text)
    except ValueError as exc:
        fail_with(exc)


FileOption = Annotated[Path | None, typer.Option(exists=True, dir_okay=False, readable=True)]


@app.command(name="run")
@notified("run", key="name")
def run_command(
    name: Annotated[str, typer.Argument(help="The instance.", autocompletion=instance_names)],
    prompt: Annotated[
        str | None, typer.Argument(help="What you want: a question, a picture, a clip, a line.")
    ] = None,
    image: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="A picture to repaint, edit, animate or ask a vision model about.",
        ),
    ] = None,
    mask: FileOption = None,
    audio: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="A clip to transcribe, split into stems, convert or score.",
        ),
    ] = None,
    lyrics: Annotated[
        Path | None,
        typer.Option(
            exists=True, dir_okay=False, readable=True, help="A song's lyrics, from a text file."
        ),
    ] = None,
    repeat: RepeatOption = 1,
) -> None:
    """Send an instance your prompt through the router and keep what comes back as a run
    (the web's gallery, `mnm outputs`); `--repeat N` makes N runs of it, one after another
    (a song's N are candidates under one run, to pick one from)."""
    service = InstanceService(Settings())
    store = RunStore.for_settings(service.settings)
    ask = Ask(
        prompt=prompt,
        picture=_given(image, "picture"),
        mask=_given(mask, "mask"),
        clip=_given(audio, "clip"),
        lyrics=_lyrics(lyrics),
    )
    made: list[RunRecord] = []
    parent: RunRecord | None = None
    try:
        recipe = service.get(name)
        if takes_candidates(recipe, repeat):
            parent = open_parent(store, recipe, ask)
        if repeat > 1 and (seed := fixed_seed(recipe)) is not None:
            console.print(
                f"[yellow]{escape(name)} draws with seed {seed} every time: each repeat is "
                f"the same picture (`mnm set {escape(name)} seed=` draws a new one each).[/yellow]"
            )
        with try_run.http_client(service.settings) as http:
            for turn in range(1, repeat + 1):
                working = f"{name} is working (the router loads it if it is parked)…"
                if repeat > 1:
                    kind = "Candidate" if parent is not None else "Run"
                    working = f"{kind} {turn} of {repeat}: {working}"
                with console.status(working):
                    run = make(service, store, recipe, http, ask, parent=parent)
                made.append(run)
                said = f"{run.seconds or 0:.1f} s: {escape(run.answer)}"
                console.print(f"[green]{escape(name)}[/green] in {said}")
                print_run_folder(run.id, store.folder(run))
    except (TryError, InstanceError, RunError) as exc:
        if parent is not None:
            close_parent(store, parent, made, repeat, error=str(exc))
        if made:
            console.print(f"{len(made)} of {repeat} runs were made before this one failed.")
        fail_with(exc)
    if parent is not None:
        close_parent(store, parent, made, repeat)
        console.print(
            f"{len(made)} candidates under run {escape(parent.id)} (the gallery shows them "
            "together; `mnm outputs show` lists each):"
        )
        for run in made:
            console.print(f"  {escape(run.id)}  {escape(str(store.folder(run)))}", soft_wrap=True)
    elif repeat > 1:
        print_repeats(made, store)


@outputs_app.command("ls")
def outputs_ls(
    limit: Annotated[int, typer.Option(help="How many of the newest runs.", min=1)] = 30,
) -> None:
    """The runs, newest first, with their files and sizes."""
    store = _store()
    runs = store.all()
    table = Table(box=box.SIMPLE_HEAD)
    for column in ("Run", "Kind", "Target", "State", "Files", "Size"):
        table.add_column(column)
    for run in runs[:limit]:
        files = "removed" if run.files_removed else ", ".join(f.name for f in run.outputs)
        table.add_row(
            escape(run.id),
            run.kind.value,
            escape(run.target),
            _state(run),
            escape(files or "-"),
            size_words(run.size),
        )
    console.print(table)
    total = sum(r.size for r in runs)
    console.print(f"{len(runs)} runs, {size_words(total)} in {escape(str(store.outputs_dir))}")
    for problem in store.problems():
        console.print(f"[yellow]{escape(problem)}[/yellow]")


STATE_STYLE = {  # every RunState
    RunState.DONE: "green",
    RunState.FAILED: "red",
    RunState.RUNNING: "yellow",
    RunState.WAITING: "cyan",
}


def _state(run: RunRecord) -> str:
    if run.interrupted:  # its process is gone: deletable, a film resumable
        return "[red]interrupted[/red]"
    style = STATE_STYLE[run.state]
    return f"[{style}]{run.state.value}[/{style}]"


@outputs_app.command("show")
def outputs_show(run_id: Annotated[str, typer.Argument()]) -> None:
    """One run: what was asked, with which settings, and its files."""
    store = _store()
    try:
        run = store.get(run_id)
        folder = store.folder(run)  # a hand-edited record's folder is said
    except RunError as exc:
        fail_with(exc)
    console.print(f"{escape(run.id)}  {run.kind.value} {escape(run.target)}  {_state(run)}")
    if run.prompt:
        console.print(f"Prompt: {escape(run.prompt)}")
    for key, value in run.settings.items():
        lines = value.splitlines()
        shown = f"{lines[0]} … ({len(lines)} lines)" if len(lines) > 1 else value
        console.print(f"  {escape(key)}: {escape(shown)}")
    if run.parent:  # Where it sits in its take's tree
        console.print(f"Under run {escape(run.parent)} ({run.role or 'a child'}).")
    for child in store.children(run.id):
        console.print(f"  {child.role or 'child'}: {escape(child.id)}  {_state(child)}")
    if run.answer:
        console.print(f"Answer: {escape(run.answer)}")
    if run.error:
        console.print(f"[red]{escape(run.error)}[/red]")
    for item in run.files:
        console.print(f"  {item.role}: {escape(str(folder / item.name))} ({size_words(item.size)})")


@outputs_app.command("rm")
def outputs_rm(
    run_ids: Annotated[list[str], typer.Argument(help="Run ids (`mnm outputs ls`).")],
    keep_files: Annotated[
        bool, typer.Option("--keep-files", help="Forget the run, keep its folder.")
    ] = False,
) -> None:
    """Delete runs: their records and their files."""
    store = _store()
    for run_id in run_ids:
        try:
            gone = store.remove(run_id, keep_files=keep_files)
        except RunError as exc:
            fail_with(exc)
        if not gone.damaged:
            console.print(f"Deleted {escape(gone.run_id)} ({size_words(gone.size)}).")
            continue
        where = f" and its folder {escape(str(gone.folder))}" if gone.folder else ""
        console.print(f"Deleted {escape(gone.run_id)}'s damaged record{where}.")


@outputs_app.command("prune")
def outputs_prune(
    older_than: Annotated[
        str | None, typer.Option("--older-than", help="An age: 12h, 30d, 2w.")
    ] = None,
    max_gb: Annotated[
        float | None,
        typer.Option("--max-gb", help="Delete the oldest runs until the rest fit.", min=0),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask.")] = False,
) -> None:
    """Delete old runs (records and files), by age, by total size, or both."""
    if older_than is None and max_gb is None:
        fail_with(ValueError("Say how: --older-than 30d, --max-gb 20, or both."))
    store = _store()
    try:
        until = size_limit(max_gb) if max_gb is not None else None
    except RunError as exc:
        fail_with(exc)
    picked = select(store.read_all(), _age(older_than), until)
    if not picked:
        console.print("Nothing to delete.")
        return
    size = sum(r.size for r in picked)
    if not yes and not typer.confirm(f"Delete {len(picked)} runs ({size_words(size)})?"):
        raise typer.Exit(code=1)
    for run in picked:
        store.delete(run.id)
    console.print(f"Deleted {len(picked)} runs ({size_words(size)}).")


@app.command(name="clean")
def clean(
    targets: Annotated[
        list[Target] | None,
        typer.Argument(
            help="outputs, runs, logs, workflows, imports, comfy (none: show what each holds)."
        ),
    ] = None,
    older_than: Annotated[
        str | None, typer.Option("--older-than", help="Only what is older: 12h, 30d, 2w.")
    ] = None,
    keep_files: Annotated[
        bool, typer.Option("--keep-files", help="runs: forget the history, keep the files.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask.")] = False,
    auto: Annotated[
        bool,
        typer.Option(
            "--auto",
            help="What the nightly cleans: runs and logs older than `mnm nightly set "
            "clean=30d`'s age, without asking.",
        ),
    ] = False,
) -> None:
    """Empty what runs and engines left behind, one target at a time. Models, recipes, sets
    and engine images are never touched (`mnm rm`, `uninstall.sh`)."""
    settings = Settings()
    cleaner = Cleaner(
        RunStore.for_settings(settings),
        settings.expanded_state_dir(),
        ComfyEngine(settings).clear_history,
    )
    if auto:
        if targets or older_than or keep_files:
            fail_with(ValueError("--auto takes its targets and age from `mnm nightly set`."))
        targets, older_than, yes = list(AUTO), _nightly_age(settings), True
    age = _age(older_than)
    survey = {item.target: item for item in cleaner.survey(age)}
    if not targets:
        width = max(len(target.value) for target in Target)
        for item in survey.values():
            console.print(
                f"{item.target.value:<{width}} {item.count:>5} {size_words(item.size):>10}  "
                f"{escape(item.description)}"
            )
        console.print("`mnm clean <target>` empties one (`--older-than 30d` keeps the newer).")
        return
    chosen = [survey[t] for t in targets]
    what = ", ".join(f"{i.target.value} ({i.count}, {size_words(i.size)})" for i in chosen)
    if not yes and not typer.confirm(f"Delete {what}?"):
        raise typer.Exit(code=1)
    for item in cleaner.clean(list(targets), age, keep_files):
        note = f" ({escape(item.note)})" if item.note else ""
        console.print(f"Cleaned {item.target.value}: {item.count}, {size_words(item.size)}.{note}")


def _nightly_age(settings: Settings) -> str:
    """The age `mnm nightly set clean=` saved; refused when the user never turned it on."""
    try:
        saved = nightly_settings.read(settings.expanded_state_dir()).clean
    except nightly_settings.NightlySettingsError as exc:
        fail_with(exc)
    if saved is None:
        fail_with(ValueError("`mnm clean --auto` is off: `mnm nightly set clean=30d` turns it on."))
    return saved
