"""`mnm apps`: the workspace apps mnm installs (OpenCode, LibreChat,
OpenHands), pinned, on 127.0.0.1, with their config written from the router's manifest; and
the ones it only connects (Goose, Odysseus)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.markup import escape
from rich.table import Table

from makenomistakes import report
from makenomistakes.apps import exposure
from makenomistakes.apps.cards import card_for, local_manifest
from makenomistakes.apps.connect import apply_odysseus, odysseus_folder, preview
from makenomistakes.apps.context import AppContext
from makenomistakes.apps.defs import CONNECTABLE, DEFS, card_only
from makenomistakes.apps.hub import AppsService, AppStatus
from makenomistakes.apps.ledger import AppsError
from makenomistakes.apps.names import LABELS, App
from makenomistakes.apps.open_webui_accounts import first_account
from makenomistakes.apps.users import SubprocessInputRunner, add_librechat_user, ask_password
from makenomistakes.cli._app import app, console, fail_with, print_block
from makenomistakes.cli.notify_cmd import notified
from makenomistakes.cli.url_cmd import app_card_and_write, print_card
from makenomistakes.config import Settings
from makenomistakes.hardware import SubprocessRunner
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.setup_prompt import TerminalPrompter
from makenomistakes.store import StoreError

apps_app = typer.Typer(
    help="Workspace apps: OpenCode, LibreChat and OpenHands installed (pinned, on 127.0.0.1); "
    "Goose and Odysseus connected.",
    no_args_is_help=True,
)
app.add_typer(apps_app, name="apps")
user_app = typer.Typer(help="Accounts in an app mnm installed (LibreChat).", no_args_is_help=True)
apps_app.add_typer(user_app, name="user")


def _known(app: App) -> App:
    """An app the `mnm apps` verbs take; Hermes Agent, a card only, refused with where its
    card is (never a KeyError, never a refusal pointing at another refusal)."""
    if (why := card_only(app)) is not None:
        fail_with(AppsError(why))
    return app


AppArg = Annotated[
    App,
    typer.Argument(help="opencode, librechat, open-webui or openhands.", callback=_known),
]


def _say(line: str) -> None:
    console.print(f"  [dim]{escape(line)}[/dim]", soft_wrap=True)


def _service(force: bool = False) -> AppsService:
    service = InstanceService(Settings())
    ctx = AppContext(
        service.settings,
        SubprocessRunner(),
        lambda: local_manifest(service),
        say=_say,
        force=force,
    )
    return AppsService(ctx)


def _fail(exc: Exception) -> NoReturn:
    """mnm's own refusals as they are; a file or folder that could not be written, worded."""
    if isinstance(exc, InstanceError | StoreError):
        fail_with(exc)
    if isinstance(exc, OSError):
        fail_with(AppsError(f"{exc.filename or 'A file'}: {exc.strerror or exc}"))
    raise exc


@apps_app.command("ls")
def ls() -> None:
    """Every app: installed or not, running, its version against this mnm's pin, its URL."""
    rows = _service().statuses()
    table = Table(title="mnm apps")
    for column in ("App", "State", "Version"):
        table.add_column(column)
    table.add_column("URL", no_wrap=True)
    for row in rows:
        where = row.definition.url or f"mnm apps connect {row.app.value}"
        table.add_row(row.label, _state(row), _version(row), where)
    console.print(table)
    console.print(
        "[dim]`mnm apps status <app>` says what each one is and where its password is.[/dim]"
    )


def _state(row: AppStatus) -> str:
    style = {"running": "green", "stopped": "yellow", "unknown": "red"}.get(row.state, "dim")
    return f"[{style}]{row.state}[/{style}]"


def _version(row: AppStatus) -> str:
    if row.version is None:
        return f"[dim]pin {row.pinned}[/dim]" if row.pinned else ""
    if row.behind:
        return f"{row.version} [yellow](pin {row.pinned}: `mnm apps update`)[/yellow]"
    return row.version


@apps_app.command()
def status(name: AppArg) -> None:
    """One app: its state, version, URL and where its password or key is."""
    row = _service().status(name)
    console.print(f"[bold]{row.label}[/bold]: {_state(row)}  {_version(row)}")
    _say(f"{row.definition.blurb} ({row.definition.licence})")
    if row.error:
        console.print(f"  [red]{escape(row.error)}[/red]")
    if row.definition.url and row.installed:
        console.print(f"  {row.definition.url}", soft_wrap=True)
    for line in row.notes:
        _say(line)
    if row.installed and row.installed.previous:
        _say(f"before it: {row.installed.previous.version} (`mnm apps rollback {name.value}`)")


@apps_app.command()
@notified("apps install", key="name")
def install(
    name: AppArg,
    force: Annotated[
        bool, typer.Option("--force", help="Replace a config file of yours mnm cannot merge.")
    ] = False,
) -> None:
    """Install the app at this mnm's pin, write its config from the router's manifest, start it."""
    definition = DEFS[name]
    console.print(f"[bold]{LABELS[name]}[/bold]: {escape(definition.blurb)} ({definition.licence})")
    try:
        done = _service(force).install(name)
    except (AppsError, InstanceError, StoreError, OSError) as exc:
        _fail(exc)
    console.print(
        f"{LABELS[name]} {done.pin.version} is installed: {definition.url}", soft_wrap=True
    )
    status(name)


@apps_app.command()
def start(name: AppArg) -> None:
    """Start an installed app (its config as it is)."""
    _run(lambda s: s.start(name), f"{LABELS[name]} started.")


@apps_app.command()
def stop(name: AppArg) -> None:
    """Stop an app; it stays installed (and stopped across a reboot)."""
    _run(lambda s: s.stop(name), f"{LABELS[name]} stopped.")


@apps_app.command()
def logs(
    name: AppArg, lines: Annotated[int, typer.Option("-n", help="How many lines.")] = 200
) -> None:
    """The app's newest log lines."""
    try:
        print_block(_service().logs(name, lines))
    except (AppsError, InstanceError, OSError) as exc:
        _fail(exc)


@apps_app.command()
@notified("apps update", key="name")
def update(name: AppArg) -> None:
    """Move to the pin this mnm carries; the installed version is kept for rollback."""
    try:
        done = _service().update(name)
    except (AppsError, InstanceError, StoreError, OSError) as exc:
        _fail(exc)
    if done is None:
        console.print(f"{LABELS[name]} is already at this mnm's pin.")
        return
    previous = done.previous.version if done.previous else "?"
    console.print(f"{LABELS[name]} {done.pin.version} is in place; {previous} kept for rollback.")


@apps_app.command()
def rollback(name: AppArg) -> None:
    """Back to the version the last update replaced."""
    try:
        done = _service().rollback(name)
    except (AppsError, InstanceError, StoreError, OSError) as exc:
        _fail(exc)
    console.print(f"{LABELS[name]} {done.pin.version} is back in place.")


@apps_app.command()
def uninstall(
    name: AppArg,
    keep_data: Annotated[
        bool, typer.Option("--keep-data", help="Keep its accounts, chats and settings.")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", help="Do not ask before deleting its data.")
    ] = False,
) -> None:
    """Stop and remove the app; asks before deleting its data (under MNM_APPS_DIR)."""
    service = _service()
    folder = service.ctx.folder(name)
    if not keep_data and not yes and folder.exists():
        keep_data = not typer.confirm(
            f"Delete {LABELS[name]}'s data in {folder} too?", default=False
        )
    try:
        for line in service.uninstall(name, keep_data):
            _say(line)
    except (AppsError, InstanceError, OSError) as exc:
        _fail(exc)
    console.print(f"{LABELS[name]} is uninstalled.")


@apps_app.command()
def expose(
    name: AppArg,
    off: Annotated[bool, typer.Option("--off", help="Back on 127.0.0.1 only.")] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Open it without asking (plain http).")
    ] = False,
) -> None:
    """Open an installed app to your network (its own login and the router's key required),
    or close it again with --off."""
    service = _service()
    try:
        if not off and not yes and service.may_expose(name):  # plain http, asked first
            console.print(escape(exposure.clear_text(name)), soft_wrap=True)
            if not typer.confirm(f"Open {LABELS[name]} to your network anyway?", default=False):
                console.print(f"{LABELS[name]} stays on 127.0.0.1.")
                raise typer.Exit(1)
        lines = service.expose(name, not off)
    except (AppsError, InstanceError, OSError) as exc:
        _fail(exc)
    for line in lines:
        console.print(escape(line), soft_wrap=True)


@user_app.command("add")
def user_add(
    name: AppArg,
    email: Annotated[str, typer.Argument(help="The account's email (its login).")],
    full_name: Annotated[str | None, typer.Option("--name", help="Shown in LibreChat.")] = None,
    username: Annotated[
        str | None, typer.Option(help="Default: the email's part before @.")
    ] = None,
) -> None:
    """An account in mnm's LibreChat (registration is off). Asks for the password."""
    if name is App.OPEN_WEBUI:
        fail_with(AppsError(first_account()))
    if name is not App.LIBRECHAT:
        fail_with(AppsError(f"{LABELS[name]} has no accounts mnm makes: only LibreChat does."))
    password = ask_password(TerminalPrompter(console), lambda line: console.print(escape(line)))
    if password is None:
        raise typer.Exit(2)
    try:
        console.print(
            escape(
                add_librechat_user(email, full_name, username, password, SubprocessInputRunner())
            )
        )
    except AppsError as exc:
        fail_with(exc)


@apps_app.command()
def connect(
    name: Annotated[App, typer.Argument(help="goose or odysseus.", callback=_known)],
    write_file: Annotated[
        bool, typer.Option("--write/--no-write", help="Write the app's file (Goose).")
    ] = True,
    force: Annotated[
        bool, typer.Option("--force", help="Replace a file mnm cannot merge.")
    ] = False,
    folder: Annotated[
        Path | None, typer.Option("--dir", help="Odysseus's folder (default ~/odysseus).")
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Write and restart without asking.")] = False,
) -> None:
    """Link an app mnm does not install: Goose's provider file, or Odysseus's .env (a diff
    first, then `docker compose up -d` on yes) and the forms to fill in its Settings."""
    if name not in CONNECTABLE:
        fail_with(
            AppsError(
                f"{LABELS[name]} is installed, not connected: `mnm apps install {name.value}`."
            )
        )
    service = InstanceService(Settings())
    if name is App.GOOSE:
        app_card_and_write(service, name, write_file, None, force)
        return
    _connect_odysseus(service, odysseus_folder(folder), force, yes)


def _connect_odysseus(service: InstanceService, folder: Path, force: bool, yes: bool) -> None:
    settings = service.settings
    card = card_for(App.ODYSSEUS, local_manifest(service), settings)
    print_card(card, show_files=False)
    known = report.secrets(settings).known
    try:
        seen = preview(card, folder, known)
    except AppsError as exc:
        fail_with(exc)
    if not seen.diff:
        console.print(f"{seen.path} already says this: nothing to write.")
        return
    console.print(f"\n[bold]{escape(str(seen.path))}[/bold] would change:")
    print_block(seen.diff)
    if not yes and not typer.confirm("Write it and restart Odysseus?", default=False):
        console.print("Nothing written.")
        return
    try:
        for line in apply_odysseus(seen, folder, settings, SubprocessRunner(), force):
            console.print(escape(line), soft_wrap=True)
    except AppsError as exc:
        fail_with(exc)


def _run(work: Callable[[AppsService], object], said: str) -> None:
    try:
        work(_service())
    except (AppsError, InstanceError, StoreError, OSError) as exc:
        _fail(exc)
    console.print(said)
