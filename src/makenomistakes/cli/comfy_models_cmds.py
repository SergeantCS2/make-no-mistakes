"""`mnm comfyui models <workflow>`: where each model file a workflow names is,
and fetching the missing ones through mnm's downloader (one copy, linked where ComfyUI
looks). Asked before anything downloads, with the sizes."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from makenomistakes.cli._app import console, fail_with
from makenomistakes.cli.comfyui_cmds import comfy_app
from makenomistakes.comfy.fetch import fetch_all, find_links, to_fetch
from makenomistakes.comfy.models import ModelError, Models, Status, Where, parse_source
from makenomistakes.comfy.report import analyze
from makenomistakes.comfy.store import WorkflowMeta, WorkflowStore, WorkflowStoreError
from makenomistakes.config import Settings
from makenomistakes.hub.civitai import CivitaiClient
from makenomistakes.instances import InstanceService
from makenomistakes.progress_words import with_ellipsis
from makenomistakes.recipes import Recipe
from makenomistakes.sizes import count_words, size_words


def _models(settings: Settings) -> Models:
    """The model files service for this box; tests replace it."""
    return Models(
        settings.expanded_models_dir(),
        settings.expanded_state_dir(),
        civitai=lambda: CivitaiClient(settings.civitai_token_value()),
    )


@comfy_app.command("models")
def models_cmd(
    name: Annotated[str, typer.Argument(help="An imported workflow.")],
    given: Annotated[
        list[str] | None,
        typer.Option("--set", help="FILE=LINK: where a file comes from (Hugging Face or Civitai)."),
    ] = None,
    find: Annotated[
        bool, typer.Option("--find", help="Look for files without a link in known repositories.")
    ] = False,
    fetch: Annotated[bool, typer.Option("--fetch", help="Download what is missing.")] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask before downloading.")] = False,
) -> None:
    """Each model file the workflow names: in place, on disk (an instance's), fetchable, or
    unknown; `--fetch` downloads and links them."""
    settings = Settings()
    store = WorkflowStore(settings.expanded_state_dir())
    models = _models(settings)
    try:
        pairs = [pair.partition("=") for pair in given or []]
        for file, sep, link in pairs:
            if not sep:
                raise ModelError(f"'{file}' is not FILE=LINK.")
            parse_source(link)  # refused now, not at the download

        def set_links(meta: WorkflowMeta) -> None:
            for file, _sep, link in pairs:
                meta.link(file, link)

        meta = store.update(name, set_links)
        report = analyze(store.workflow(name), None)
        recipes = InstanceService(settings).all()
        statuses = models.status(report.models, recipes, meta.links)
    except (WorkflowStoreError, ModelError, ValueError, RuntimeError) as exc:
        fail_with(exc)
    if find:
        found = find_links(models, statuses)
        for line in found.said:
            console.print(escape(line))
        try:
            meta = store.update(name, found.apply)
        except WorkflowStoreError as exc:  # removed while Find ran
            fail_with(exc)
        statuses = models.status(report.models, recipes, meta.links)
        for status in (s for s in statuses if s.where is Where.UNKNOWN):
            for suggestion in meta.found.get(status.need.name, []):
                console.print(f"  {escape(suggestion.label)}")
                pair = f"{status.need.name}={suggestion.url}"
                console.print(f"    use it: --set {escape(repr(pair))}")
    _print(models, statuses)
    if not fetch:
        return
    _fetch(models, statuses, settings, recipes, name, yes)


def _fetch(
    models: Models,
    statuses: list[Status],
    settings: Settings,
    recipes: list[Recipe],
    name: str,
    yes: bool,
) -> None:
    """Download and link what is missing; with nothing missing, the files in place are
    still claimed for the workflow (`rm --files` weighs them)."""
    todo = to_fetch(statuses)
    if todo:
        sizes = [_size(models, s) for s in todo]
        amount = size_words(sum(size or 0 for size in sizes))
        if unknown := sum(1 for size in sizes if not size):  # never a total that looks whole
            amount += f", and {count_words(unknown, 'file')} of a size the server did not say"
        question = f"Download and link {count_words(len(todo), 'file')} ({amount})?"
        if yes:  # the total before the downloads, asked or not
            names = ", ".join(s.link for s in todo)
            console.print(f"{escape(amount)} to download ({escape(names)}).")
        elif not typer.confirm(question):
            raise typer.Exit(code=1)
    try:
        fetch_all(models, statuses, settings, recipes, _progress, owner=name)
    except (ModelError, OSError) as exc:
        fail_with(exc)
    if todo:
        console.print("[green]Linked[/green] where ComfyUI looks; run the workflow again.")
    else:
        console.print("Nothing to fetch.")


def _size(models: Models, status: Status) -> int | None:
    """A file's download size (Hugging Face or Civitai), when the server says."""
    if status.source is None:
        return None
    return models.size(status.source, PurePosixPath(status.link).name)


def _progress(message: str) -> None:
    console.print(escape(with_ellipsis(message)))


def _print(models: Models, statuses: list[Status]) -> None:
    table = Table(box=box.SIMPLE_HEAD)
    for column in ("File", "Folder", "Where", "From", "Size"):
        table.add_column(column)
    for status in statuses:
        size = _size(models, status)
        table.add_row(
            escape(status.link),
            escape(status.need.folder or "checkpoints"),
            status.where.value,
            escape(status.source.label if status.source else (status.target or "-")),
            size_words(size) if size else "",
        )
    console.print(table)
    for status in (s for s in statuses if s.refused):  # the file to `--set`, and why
        pair = f"{status.need.name}=<link>"
        console.print(f"{escape(status.link)}: {escape(status.refused)}")
        console.print(f"  give it one: --set {escape(repr(pair))}")
