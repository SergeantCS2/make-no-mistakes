"""`mnm url`: what a client pastes. With an instance's name, its cards for one client (GUIDE
13); without one, a workspace app's whole file or card, every model by desk role:
`mnm url --for opencode|librechat|open-webui|openhands|goose|odysseus [--write [--to PATH]
[--force]]`.
`--from here|docker|lan` (`client_view.py`) names where the client sits; on a client install
(`MNM_ROUTER_URL`) every URL is the box's, and only an app's file is printed."""

from __future__ import annotations

from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from makenomistakes.apps.card import AppCard
from makenomistakes.apps.cards import card_for, local_manifest
from makenomistakes.apps.connect import write_card_file, written_line
from makenomistakes.apps.files import AppFileError, WriteResult
from makenomistakes.apps.names import LABELS, App
from makenomistakes.apps.offer import NO_MCP_KEY, lan_notes
from makenomistakes.cli._app import app, console, fail_with, print_block
from makenomistakes.cli.complete import instance_names
from makenomistakes.cli.instance_cmds import print_urls
from makenomistakes.client_view import From, ReachError, Seen, seen_from
from makenomistakes.config import Settings
from makenomistakes.consumers import Consumer, engine_origin
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.manifest import seen_by, view

FOR_VALUES = sorted({c.value for c in Consumer} | {a.value for a in App})
# One choice list for --for: the clients' cards and the workspace apps (an enum, so a wrong
# value is a usage error, exit 2, as every other option's).
ForWhom = StrEnum("ForWhom", [(v.replace("-", "_").upper(), v) for v in FOR_VALUES])  # type: ignore[misc]
DEFAULT_FOR = ForWhom(Consumer.ODYSSEUS.value)
WHOLE_LIST = (
    "{label}'s file lists every model, by desk role first: `mnm url --for {app}` (no instance "
    "name)."
)


@app.command()
def url(
    name: Annotated[
        str | None,
        typer.Argument(
            help="An instance; leave it out for a workspace app's whole file.",
            autocompletion=instance_names,
        ),
    ] = None,
    whom: Annotated[
        ForWhom,
        typer.Option(
            "--for",
            help="Who pastes it: a chat client, a coding agent, another app (GUIDE 13), or a "
            "workspace app (opencode, librechat, openhands, goose, odysseus: no name needed).",
        ),
    ] = DEFAULT_FOR,
    direct: Annotated[
        bool,
        typer.Option("--direct", help="The engine's own address on 127.0.0.1, without the router."),
    ] = False,
    write_file: Annotated[
        bool,
        typer.Option(
            "--write", help="Write the app's file (its own place, or --to); merges into yours."
        ),
    ] = False,
    to: Annotated[
        Path | None, typer.Option("--to", help="Where --write writes (default: the app's own).")
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Replace a file mnm cannot merge into (a copy is kept)."),
    ] = False,
    where: Annotated[
        From | None,
        typer.Option(
            "--from",
            help="Where the client sits: here (this machine, a Windows app), docker (a "
            "container here) or lan (another machine: the router's LAN address, the key "
            "required). Default: where that client usually sits.",
        ),
    ] = None,
) -> None:
    """Print the URLs (and config or commands) to paste into a harness or a workspace app."""
    service = InstanceService(Settings())
    consumer = str(whom.value)
    try:
        seen = view(where, service.settings)
    except ReachError as exc:
        fail_with(exc)
    if name is None:
        if consumer not in App.__members__.values():
            fail_with(InstanceError(f"Name an instance: `mnm url <name> --for {consumer}`."))
        with seen_from(seen):
            app_card_and_write(
                service, App(consumer), write_file or to is not None, to, force, seen
            )
        return
    if write_file or to is not None or force:
        fail_with(InstanceError("--write is for a workspace app's file: leave the name out."))
    if consumer not in Consumer.__members__.values():
        fail_with(InstanceError(WHOLE_LIST.format(label=LABELS[App(consumer)], app=consumer)))
    if service.settings.router_url:
        fail_with(ReachError(CLIENT_INSTANCE.format(url=service.settings.router_url)))
    try:
        recipe = service.get(name)
    except InstanceError as exc:
        fail_with(exc)
    if direct:
        if seen is not None and seen.where is not From.HERE:
            fail_with(ReachError("--direct is the engine on this machine only: no --from."))
        console.print(
            f"{engine_origin(recipe)}  (the engine itself, this machine only: no loading on "
            "demand and none of the router's rewrites; for debugging, never for clients)"
        )
        return
    with seen_from(seen):
        print_urls(recipe, Consumer(consumer))


CLIENT_INSTANCE = (
    "MNM_ROUTER_URL is set ({url}): this is a client install, which has no instances of its "
    "own. `mnm url --for <app>` prints a workspace app's file by desk role, `mnm router "
    "manifest` every model; an instance's own cards: `mnm url <name> --for … --from lan` on "
    "the box."
)


def app_card_and_write(
    service: InstanceService,
    which: App,
    writing: bool,
    to: Path | None,
    force: bool,
    seen: Seen | None = None,
) -> None:
    try:
        manifest = seen_by(local_manifest(service), seen)
    except InstanceError as exc:
        fail_with(exc)
    card = card_for(which, manifest, service.settings)
    if seen is not None and seen.where is From.LAN:
        notes = lan_notes(which, mcp_takes_key=which not in NO_MCP_KEY)
        card = replace(card, lines=[*card.lines, *notes])
    print_card(card, show_files=not writing)
    if not writing:
        return
    try:
        done = write_card_file(card, service.settings, force, to)
    except AppFileError as exc:
        fail_with(exc)
    _print_written(done)


def print_card(card: AppCard, show_files: bool = True) -> None:
    console.print(f"[bold]{escape(card.label)}[/bold]")
    for line in card.lines:
        console.print(f"  [dim]{escape(line)}[/dim]")
    for file in card.files if show_files else []:
        console.print(f"\n[bold]{escape(file.label)}[/bold] (`--write` writes it)")
        print_block(file.text())
    for label, text in card.blocks:
        console.print(f"\n[bold]{escape(label)}[/bold]")
        print_block(text)


def _print_written(done: WriteResult) -> None:
    console.print(escape(written_line(done)), soft_wrap=True)  # a path a terminal can copy whole
    if done.diff:
        print_block(done.diff)
