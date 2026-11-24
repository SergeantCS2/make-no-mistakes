"""`mnm machine`: the box mnm runs on, as its fingerprint says it. `show` (the default)
reads the box now (cheap reads), records it and says it in full, with what changed since the
box before and which instances were planned on another; `history` lists every box this state
folder has seen; `forget <id>` drops one that is not this box; `adopt` plans every
instance again for this box (`machine_adopt`), asking before any download."""

from __future__ import annotations

from typing import Annotated

import click
import typer
from rich.markup import escape
from rich.table import Table

from makenomistakes.adopt_live import live_deps
from makenomistakes.cli._app import app, console, fail_with
from makenomistakes.config import Settings
from makenomistakes.hardware.probe import probe_memory
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.local_time import local
from makenomistakes.machine import observe_snapshot, saved_recipes
from makenomistakes.machine_adopt import AdoptError, Adoption, AdoptOptions, adopt
from makenomistakes.machine_facts import MachineFactsError, load_file
from makenomistakes.machine_facts import forget as forget_box
from makenomistakes.machine_show import history_lines, show_lines
from makenomistakes.measurements import MeasurementStore
from makenomistakes.readings_filter import belongs_to, per_box

machine_app = typer.Typer(
    help="The box mnm runs on: its fingerprint, every box seen, what changed since plans were "
    "made.",
    invoke_without_command=True,
)
app.add_typer(machine_app, name="machine")


@machine_app.callback()
def machine(ctx: typer.Context) -> None:
    """This box (the same as `mnm machine show`)."""
    if ctx.invoked_subcommand is None:
        show(as_json=False)


@machine_app.command()
def show(
    as_json: Annotated[
        bool, typer.Option("--json", help="The whole machine file as JSON (uuids included).")
    ] = False,
    reveal: Annotated[
        bool,
        typer.Option(
            "--reveal", help="Show the cards' uuids (private: never paste them anywhere)."
        ),
    ] = False,
) -> None:
    """This box in full, what changed since the box before, and the instances planned on
    another box."""
    settings = Settings()
    observed = observe_snapshot(settings, probe_memory())
    if observed.unread is not None:
        console.print(f"[yellow]This box could not be read: {escape(observed.unread)}.[/yellow]")
    elif observed.error is not None:
        console.print(f"[yellow]This box was not recorded: {escape(observed.error)}.[/yellow]")
    try:
        found = load_file(settings.expanded_state_dir())
    except MachineFactsError as exc:
        fail_with(exc)
    if as_json:
        print(found.model_dump_json(indent=2) if found is not None else "{}")
        return
    if found is None:
        console.print("No box recorded yet.")
        return
    counts = per_box(MeasurementStore(settings.expanded_state_dir()), found)
    for line in show_lines(found, saved_recipes(settings), reveal, counts):
        console.print(escape(line))


@machine_app.command()
def history() -> None:
    """Every box this state folder has seen, newest first, one line each."""
    settings = Settings()
    try:
        found = load_file(settings.expanded_state_dir())
    except MachineFactsError as exc:
        fail_with(exc)
    counts = per_box(MeasurementStore(settings.expanded_state_dir()), found)
    for line in history_lines(found, saved_recipes(settings), counts):
        console.print(escape(line))


@machine_app.command()
def forget(
    machine_id: Annotated[str, typer.Argument(metavar="ID", help="A box's id (`history`).")],
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask first.")] = False,
) -> None:
    """Drop a box that is not this one, with what was measured on it (its RAM bandwidth, its
    expert rate and every reading taken on it). Instances planned on it stay, and still
    count as planned elsewhere."""
    machine_id = machine_id.strip().rstrip(":")  # copied from `history`'s "<id>: first …"
    settings = Settings()
    store = MeasurementStore(settings.expanded_state_dir())
    try:
        found = load_file(settings.expanded_state_dir())
    except MachineFactsError as exc:
        fail_with(exc)
    entry = found.entry(machine_id) if found is not None else None
    current = found.current.fingerprint.id if found is not None and found.current else None
    owned = belongs_to(machine_id, found)
    count = per_box(store, found).get(machine_id, 0)
    if entry is not None and machine_id != current and not yes:  # a refusal is never asked
        last = f"{local(entry.last_seen):%Y-%m-%d}"
        typer.confirm(
            f"Forget {machine_id} ({entry.fingerprint.words()}, last seen {last}), what was "
            f"measured on it and its {count} reading{'s' * (count != 1)}?",
            abort=True,
        )
    try:
        gone = forget_box(settings.expanded_state_dir(), machine_id)
    except MachineFactsError as exc:
        fail_with(exc)
    removed = store.drop(owned)  # only once the box is gone: a refused forget keeps them
    console.print(
        f"Forgot {escape(machine_id)}: {escape(gone.fingerprint.words())}; "
        f"{removed} reading{'s' * (removed != 1)} removed."
    )


@machine_app.command(name="adopt")
def adopt_cmd(
    only: Annotated[
        list[str] | None,
        typer.Argument(metavar="[NAME]...", help="Only these instances (`--only` too)."),
    ] = None,
    only_opt: Annotated[
        list[str] | None, typer.Option("--only", help="Only this instance (repeatable).")
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Download what the plans need.")] = False,
    keep_quants: Annotated[
        bool, typer.Option("--keep-quants", help="Keep each instance's quant (no new files).")
    ] = False,
    no_tune: Annotated[bool, typer.Option("--no-tune", help="Queue no tunes.")] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the box and the plans; save nothing.")
    ] = False,
) -> None:
    """Plan every instance again for this box after its hardware changed: the RAM measured,
    the quant free to change, every download listed with its size and asked once, tunes
    queued. Never deletes an instance and never writes the secrets file."""
    service = InstanceService(Settings())
    names = tuple(dict.fromkeys([*(only or []), *(only_opt or [])]))
    options = AdoptOptions(
        yes=yes, keep_quants=keep_quants, tune=not no_tune, only=names, dry_run=dry_run
    )
    deps = live_deps(service, _say, _ask)
    try:
        done = adopt(deps, options)
    except (AdoptError, InstanceError) as exc:
        fail_with(exc)
    if done.rows:
        console.print(_table(done))


def _say(line: str) -> None:
    console.print(escape(line), highlight=False)


def _ask(question: str) -> bool:
    try:
        return typer.confirm(question, default=False)
    except click.Abort:  # no answer (no terminal): nothing is downloaded
        console.print("No answer: nothing downloaded (`--yes` takes the downloads).")
        return False


def _table(done: Adoption) -> Table:
    table = Table(title="Dry run: what adopt would do" if done.dry_run else "Adopted")
    for column in ("Instance", "Plan", "Files to fetch", "Tune"):
        table.add_column(column)
    for row in done.rows:
        files = ", ".join(f"{f.path.rsplit('/', 1)[-1]} ({_size(f.size)})" for f in row.fetch)
        fetched = f"; {row.fetched}" if files and row.fetched else ""
        table.add_row(
            escape(row.name),
            escape(row.kept or row.plan),
            escape((files or "-") + fetched),
            escape(row.tune or "-"),
        )
    return table


def _size(size: int | None) -> str:
    return f"{size / 2**30:.1f} GiB" if size else "size unknown"
