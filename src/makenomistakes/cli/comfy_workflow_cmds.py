"""`mnm comfyui workflow add|ls|show|rm|fields|pick` and `mnm comfyui run|again`:
import a ComfyUI workflow from a link or a file, see what it needs before anything is installed,
run it and run it again."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from makenomistakes.cli._app import console, fail_with, print_run_folder
from makenomistakes.cli.comfyui_cmds import comfy_app
from makenomistakes.cli.notify_cmd import notified
from makenomistakes.cli.run_cmds import RepeatOption, print_repeats
from makenomistakes.comfy import sources
from makenomistakes.comfy.engine import ComfyEngine, ComfyError, State, unread_nodes
from makenomistakes.comfy.fetch import remove_workflow
from makenomistakes.comfy.form import Kind
from makenomistakes.comfy.models import ModelError, Models
from makenomistakes.comfy.object_info import ObjectInfo
from makenomistakes.comfy.packs import Approval, PackStore, standing
from makenomistakes.comfy.registry import PackError
from makenomistakes.comfy.report import Report, analyze
from makenomistakes.comfy.room import park_for_comfy
from makenomistakes.comfy.run import ComfyRunError, ComfyRunner
from makenomistakes.comfy.sources import Found, SourceError, fetch, from_file
from makenomistakes.comfy.store import WorkflowStore, WorkflowStoreError
from makenomistakes.config import Settings
from makenomistakes.hub.civitai import CivitaiClient
from makenomistakes.instances import InstanceService
from makenomistakes.local_time import local
from makenomistakes.pipeline.workflow_users import WorkflowInUseError, refuse_if_used
from makenomistakes.progress_words import with_ellipsis
from makenomistakes.runs import RunError, RunRecord, RunStore
from makenomistakes.sizes import count_words, size_words
from makenomistakes.try_figures import run_figures

workflow_app = typer.Typer(help="ComfyUI workflows: import from a link or a file, inspect, remove.")
comfy_app.add_typer(workflow_app, name="workflow")


def _store() -> WorkflowStore:
    return WorkflowStore(Settings().expanded_state_dir())


def _fetch(reference: str) -> list[Found]:
    """A file here, or a link (tests replace this)."""
    path = Path(reference).expanduser()
    if path.is_file():
        return from_file(path)
    settings = Settings()
    with sources.default_http() as http:
        civitai = CivitaiClient(settings.civitai_token_value())
        try:
            return fetch(reference, http, civitai)
        finally:
            civitai.close()


def _packs_now(settings: Settings) -> tuple[list[Approval], list[str]]:
    """The approved packs and the ones built into ComfyUI's image; an unreadable ledger is
    said once and read as none built (the report still prints)."""
    try:
        approvals = PackStore(settings.expanded_state_dir()).approvals()
    except PackError:
        approvals = []
    try:
        ledger = ComfyEngine(settings).ledger()
    except ComfyError as exc:
        console.print(f"[yellow]{escape(str(exc))}[/yellow]")
        ledger = None
    return approvals, ledger.packs if ledger else []


def print_report(report: Report, unchecked: State | None, name: str | None = None) -> None:
    """What a workflow needs, in the order the user acts on it. `unchecked` is None once its
    nodes were checked against ComfyUI's; else ComfyUI's state, which says what to do first."""
    for problem in report.problems:
        console.print(f"[red]Refused: {escape(problem)}.[/red]")
    if report.packs:
        console.print("Node packs:")
        approvals, built = _packs_now(Settings())
        for pack in report.packs:
            state = standing(pack.pack, pack.version, approvals, built)
            console.print(
                f"  {escape(pack.pack)} {escape(pack.version or '')} ({state}): "
                f"{escape(', '.join(pack.nodes))}"
            )
        console.print(
            "`mnm comfyui packs add --for <workflow>` adds them; `install` builds them in."
        )
    else:
        console.print("Node packs: none (ComfyUI's own nodes only).")
    if report.missing:
        console.print(
            f"[yellow]Nodes no pack is named for:[/yellow] {escape(', '.join(report.missing))}"
        )
    if report.risky:
        risky = escape(", ".join(report.risky))
        console.print(f"[yellow]Nodes that may run code they are given:[/yellow] {risky}")
    if report.models:
        console.print("Model files:")
        for model in report.models:
            where = model.folder or "folder unknown"
            source = "a download link" if model.url else "no link: mnm searches for it"
            note = " [yellow](a pickle: can run code when loaded)[/yellow]" if model.pickle else ""
            console.print(f"  {escape(model.name)} → {escape(where)}, {source}{note}")
    if unchecked is not None:
        first = escape(unread_nodes(unchecked))
        console.print(f"[dim]Its nodes are not checked yet. {first}[/dim]")
    elif report.runnable and report.models:
        where = f": `mnm comfyui models {escape(name)}` says which are" if name else ""
        console.print(f"[green]Runs here[/green] once its model files are in place{where}.")
    elif report.runnable:  # no model files to place (frames, frames2)
        console.print("[green]Runs here.[/green]")


@workflow_app.command("add")
def add(
    reference: Annotated[
        str, typer.Argument(help="A link (comfy.org, Civitai, HF, GitHub) or a file.")
    ],
    name: Annotated[str | None, typer.Option(help="Its name in mnm (one workflow only).")] = None,
) -> None:
    """Import a workflow (a Civitai zip may hold several). Nothing is installed or downloaded
    beyond the workflow file itself."""
    store = _store()
    try:
        found = _fetch(reference)
    except SourceError as exc:
        fail_with(exc)
    if name is not None and len(found) > 1:
        fail_with(ValueError(f"{reference} holds {len(found)} workflows: leave --name out."))
    info, unchecked = _nodes(ComfyEngine(Settings()))
    for item in found:
        try:
            wanted = name or store.free_name(item.name)
            meta = store.add(wanted, item.data, item.source, title=item.name)
        except WorkflowStoreError as exc:
            console.print(f"[red]{escape(item.name)}: {escape(str(exc))}[/red]")
            continue
        console.print(f"[green]Imported[/green] {escape(meta.name)} from {escape(item.source)}")
        print_report(analyze(store.workflow(meta.name), info), unchecked, meta.name)


@workflow_app.command("ls")
def list_cmd() -> None:
    """The imported workflows."""
    table = Table(box=box.SIMPLE_HEAD)
    for column in ("Workflow", "From", "Imported"):
        table.add_column(column)
    store = _store()
    for meta in store.all():
        table.add_row(
            escape(meta.name), escape(meta.source[:70]), f"{local(meta.imported_at):%Y-%m-%d}"
        )
    console.print(table)
    for problem in store.problems():
        console.print(f"[yellow]Not readable:[/yellow] {escape(problem)}")


@workflow_app.command("show")
def show(name: Annotated[str, typer.Argument()]) -> None:
    """What a workflow needs: node packs, model files, anything refused."""
    store = _store()
    try:
        meta = store.meta(name)
        workflow = store.workflow(name)
    except (WorkflowStoreError, ValueError) as exc:
        fail_with(exc)
    console.print(f"{escape(meta.name)}: {escape(meta.source)}")
    info, unchecked = _nodes(ComfyEngine(Settings()))
    print_report(analyze(workflow, info), unchecked, meta.name)


def _nodes(engine: ComfyEngine) -> tuple[ObjectInfo | None, State | None]:
    """ComfyUI's nodes as last read, or why there are none: its state (an unreadable ledger
    stops the command with its own words, never read as "not installed")."""
    info = engine.object_info()
    if info is not None:
        return info, None
    try:
        return None, engine.checked_state()
    except ComfyError as exc:
        fail_with(exc)


@workflow_app.command("rm")
def remove(
    name: Annotated[str, typer.Argument()],
    files: Annotated[
        bool,
        typer.Option(
            "--files",
            help="Also delete the model files mnm fetched for it that no other workflow or "
            "instance uses.",
        ),
    ] = False,
) -> None:
    """Remove an imported workflow (its runs stay in `mnm outputs`)."""
    settings = Settings()
    models = Models(settings.expanded_models_dir(), settings.expanded_state_dir())
    try:
        runs = RunStore.for_settings(settings).read_all()
        refuse_if_used(name, runs, settings.expanded_state_dir())
        removed = remove_workflow(
            _store(),
            models,
            name,
            InstanceService(settings).all(),
            settings.expanded_models_dir(),
            files,
        )
    except (WorkflowStoreError, ModelError, OSError, WorkflowInUseError, RunError) as exc:
        fail_with(exc)
    if removed.present:
        console.print(f"Removed workflow {escape(name)}.")
    else:
        console.print(f"{escape(name)} was removed before; now its files.")
    if removed.kept:
        console.print(f"[yellow]{escape(removed.kept)}.[/yellow]")
    if removed.cleanup is not None:
        done = removed.cleanup
        line = f"Deleted {count_words(len(done.deleted), 'file')} ({size_words(done.freed_bytes)})."
        if removed.stays:
            them = "it" if removed.stays == 1 else "them"
            line += (
                f" {count_words(removed.stays, 'file')} still there: another workflow or an "
                f"instance uses {them}."
            )
        console.print(line)


def _runner() -> ComfyRunner:
    """The runner for this box; tests replace it."""
    settings = Settings()
    state = settings.expanded_state_dir()
    return ComfyRunner(
        ComfyEngine(settings),
        WorkflowStore(state),
        RunStore.for_settings(settings),
        park_for_comfy(InstanceService(settings)),
    )


@workflow_app.command("fields")
def fields(
    name: Annotated[str, typer.Argument()],
    every: Annotated[bool, typer.Option("--all", help="More settings too.")] = False,
) -> None:
    """The settings a run takes (`key=value` for `mnm comfyui run`), with the workflow's own
    values."""
    try:
        _prompt, found = _runner().prompt(name)
    except ComfyRunError as exc:
        fail_with(exc)
    table = Table(box=box.SIMPLE_HEAD)
    for column in ("Key", "Setting", "Kind", "Its value"):
        table.add_column(column)
    for field in found:
        if field.shown or every:
            value = "random each run" if field.kind is Kind.SEED else str(field.default)[:60]
            label = f"{field.label} (picked)" if field.picked else field.label
            table.add_row(escape(field.key), escape(label), field.kind.value, escape(value))
    console.print(table)


@workflow_app.command("pick")
def pick(
    name: Annotated[str, typer.Argument()],
    keys: Annotated[list[str], typer.Argument(help="Keys from `fields --all`.")],
    unpick: Annotated[bool, typer.Option("--unpick", help="Back under More settings.")] = False,
) -> None:
    """Show settings from More settings on top of the web form (and in `fields`)."""
    runner = _runner()

    def change(now: list[str]) -> list[str]:
        kept = [k for k in now if k not in keys]
        return kept if unpick else kept + keys

    try:
        picked = runner.set_shown(name, change)
    except (ComfyRunError, WorkflowStoreError) as exc:
        fail_with(exc)
    shown = ", ".join(picked) or "none"
    console.print(f"{escape(name)}: settings picked to show: {escape(shown)}")


@comfy_app.command("run")
@notified("comfyui run", key="name")
def run_cmd(
    name: Annotated[str, typer.Argument(help="An imported workflow.")],
    values: Annotated[
        list[str] | None, typer.Argument(help="key=value settings (`mnm comfyui workflow fields`).")
    ] = None,
    file: Annotated[
        list[str] | None,
        typer.Option("--file", help="key=path: a picture, clip or sound from this folder."),
    ] = None,
    repeat: RepeatOption = 1,
) -> None:
    """Run an imported workflow; ComfyUI starts if it is off, mnm's models make room, and the
    outputs are kept as a run (`mnm outputs`). `--repeat N`: N runs, a new seed for each
    after the first."""
    runner = _runner()
    settings = dict(_pairs(values or []))
    uploads = {}
    for key, path in _pairs(file or []):
        given = Path(path)
        if not given.resolve().is_relative_to(Path.cwd().resolve()) or not given.is_file():
            fail_with(ValueError(f"{path} must be a file in the current folder."))
        uploads[key] = (given.name, given.read_bytes())
    made: list[RunRecord] = []
    try:
        run = runner.new(name, settings)
        made.append(runner.run(run, settings, uploads, _progress))
        _print_done(runner, made[-1])
        for turn in range(2, repeat + 1):  # the same settings and uploads, new seeds
            console.print(f"Run {turn} of {repeat}, a new seed:")
            plan = runner.again(made[-1].id, new_seed=True)
            run = runner.new(plan.target, plan.values)
            made.append(runner.run(run, plan.values, plan.uploads, _progress))
            _print_done(runner, made[-1])
    except ComfyRunError as exc:
        if made:
            console.print(f"{len(made)} of {repeat} runs were made before this one failed.")
        fail_with(exc)
    if repeat > 1:
        print_repeats(made, runner.runs)


@comfy_app.command("again")
@notified("comfyui again", key="run_id")
def again_cmd(
    run_id: Annotated[str, typer.Argument(help="A workflow run (`mnm outputs ls`).")],
    new_seed: Annotated[bool, typer.Option("--new-seed", help="Random seeds this time.")] = False,
) -> None:
    """A workflow run's settings and uploads again, as a new run."""
    runner = _runner()
    try:
        plan = runner.again(run_id, new_seed)
        run = runner.new(plan.target, plan.values)
        done = runner.run(run, plan.values, plan.uploads, _progress)
    except ComfyRunError as exc:
        fail_with(exc)
    _print_done(runner, done)


def _print_done(runner: ComfyRunner, done: RunRecord) -> None:
    console.print(f"[green]{escape(done.target)}[/green]: {escape(done.answer)}")
    figures = [f"{value} {label}" for value, label in run_figures(done.numbers)]
    if done.seconds is not None:
        figures.insert(0, f"{done.seconds:.1f} s")
    console.print(escape(" · ".join(figures)))
    print_run_folder(done.id, runner.runs.folder(done))


def _pairs(items: list[str]) -> list[tuple[str, str]]:
    found = []
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            fail_with(ValueError(f"'{item}' is not key=value."))
        found.append((key, value))
    return found


def _progress(message: str) -> None:
    console.print(escape(with_ellipsis(message)))
