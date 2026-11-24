"""`mnm upgrade <name>`: a newer family member planned with the instance's
settings, shown as a diff; `--yes` adds it beside the old one (the new takes the name, the
old is kept as `<name>-old`); `--replace` removes `<name>-old` and its files."""

from __future__ import annotations

import sys
from typing import Annotated

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from makenomistakes.cli._app import (
    GIB,
    GpuOption,
    RamOption,
    VramOption,
    app,
    budget_override,
    console,
    fail_with,
    print_progress,
)
from makenomistakes.cli.complete import instance_names
from makenomistakes.cli.notify_cmd import notified
from makenomistakes.cli.tune_cmds import queue_after_add
from makenomistakes.config import Settings
from makenomistakes.download import cache_status
from makenomistakes.families import Offer
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.plan import PlanError, open_hub
from makenomistakes.plan_budget import planning_budget
from makenomistakes.successors import NoSuccessor, cached, hub_lookup
from makenomistakes.swap import SwapService
from makenomistakes.upgrade import Upgrade, plan_upgrade
from makenomistakes.upgrade_apply import (
    add_beside,
    download,
    old_name,
    refusal,
    replace,
    set_words,
)


@app.command()
@notified("upgrade", key="name")
def upgrade(
    name: Annotated[
        str, typer.Argument(help="The instance to upgrade.", autocompletion=instance_names)
    ],
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            help="Add the new model beside the old one: it takes the name, the old one is kept "
            "as <name>-old (parked first, after its replies, when loaded).",
        ),
    ] = False,
    replace_old: Annotated[
        bool,
        typer.Option("--replace", help="Remove <name>-old and the files no other instance uses."),
    ] = False,
    vram_gb: VramOption = None,
    ram_gb: RamOption = 14.0,
    gpu: GpuOption = "RTX 4070",
) -> None:
    """Plan the newest family member of an instance's model with its settings, show what
    changes, and add it beside the old one with --yes."""
    settings = Settings()
    service = InstanceService(settings)
    if replace_old:
        _replace(service, name, yes)
        return
    try:
        budget = budget_override(vram_gb, ram_gb, gpu) or planning_budget(service)
        with open_hub(settings) as hub:
            found = plan_upgrade(service, name, budget, cached(hub_lookup(hub.model_info)))
    except (PlanError, InstanceError) as exc:
        fail_with(exc)
    if isinstance(found, NoSuccessor):
        console.print(escape(found.words))
        return
    _show(service, found)
    refused = refusal(service, found)
    if not yes:
        if refused is not None:
            console.print(f"[yellow]{escape(refused)}[/yellow]")
            return
        running = service.status(found.old).running
        park = f" ({name} is loaded: it is parked first, after its replies)" if running else ""
        console.print(
            f"[bold]mnm upgrade {escape(name)} --yes[/bold] adds it as {escape(name)} and "
            f"keeps this one as {escape(found.old_name)}{escape(park)}."
        )
        return
    _add(service, found)


def _show(service: InstanceService, found: Upgrade) -> None:
    """The successor, where it was read, the diff, the download, what carries over."""
    old, to = found.old, found.successor
    console.print(
        f"[bold]{escape(old.name)}[/bold] ({escape(old.repo_id)}, {escape(old.quant)}) → "
        f"[bold]{escape(to.target)}[/bold]"
    )
    console.print(f"[dim]Found in {escape(to.via)}.[/dim]")
    if to.note:
        console.print(escape(to.note))
    if to.offer is Offer.FASTER:
        console.print(
            "[yellow]A speed offer, not a better model: the one you have stays the more "
            "accurate.[/yellow]"
        )
    table = Table(box=box.SIMPLE_HEAD, title="planned for this machine now (estimates)")
    table.add_column("")
    now = found.before
    table.add_column(f"now: {escape(now.label)}" if now else "now", overflow="fold")
    table.add_column(f"after: {escape(found.after.label)}", overflow="fold")
    after = found.after
    for label, was, will in (
        ("Quant", now.quant if now else old.quant, after.quant),
        ("Settings", now.settings if now else "-", after.settings),
        ("Memory", now.memory if now else "-", after.memory),
        ("Fit", now.fit if now else "-", after.fit),
        ("Speed", now.speed if now else "-", after.speed),
    ):
        table.add_row(label, escape(was), escape(will))
    console.print(table)
    cache = cache_status(found.new, service.models_dir)
    fetch = max(found.download_bytes - cache.cached_bytes, 0)
    console.print(
        f"Download: {fetch / GIB:.1f} GiB"
        + (" (some files already here)." if cache.cached else ".")
    )
    for note in found.notes:
        console.print(f"[yellow]{escape(note[0].upper() + note[1:])}.[/yellow]")
    if found.kept:
        console.print(f"Kept: {escape('; '.join(found.kept))}.")
    if found.dropped:
        console.print(f"[yellow]Dropped: {escape('; '.join(found.dropped))}.[/yellow]")
    for line in set_words(service, old.name, to.label, found.old_name, done=False):
        console.print(escape(line))


def _add(service: InstanceService, found: Upgrade) -> None:
    name = found.old.name
    try:
        added = add_beside(service, found, SwapService(service).park)
    except InstanceError as exc:
        fail_with(exc)
    parked = " (parked after its replies)" if added.parked else ""
    console.print(
        f"{escape(name)} is now {escape(found.successor.label)}; the old model is "
        f"{escape(added.old.name)}{parked}, loadable by that name."
    )
    for line in [*added.sets, *added.notes]:
        console.print(escape(line))
    try:
        recipe = download(service, added, print_progress)
    except InstanceError as exc:
        fail_with(exc)
    queue_after_add(service, recipe)
    console.print(
        f"Saved. Next: [bold]mnm start {escape(name)}[/bold] (or a request through the router). "
        f"[bold]mnm upgrade {escape(name)} --replace[/bold] removes {escape(added.old.name)} "
        "and its files once the new one has proved itself."
    )


def _replace(service: InstanceService, name: str, yes: bool) -> None:
    target = old_name(name)
    if service.find(target) is None:  # said before any question
        try:
            replace(service, name, SwapService(service).park)
        except InstanceError as exc:
            fail_with(exc)
    if not yes:
        if not sys.stdin.isatty():
            fail_with(InstanceError(f"Removing {target} deletes its files: add --yes to confirm."))
        if not typer.confirm(f"Remove {target} and the files no other instance uses?"):
            console.print(f"Kept {escape(target)}.")
            return
    try:
        old, cleanup = replace(service, name, SwapService(service).park)
    except InstanceError as exc:
        fail_with(exc)
    if cleanup is None:
        console.print(f"{escape(old.name)} removed.")
        return
    count = len(cleanup.deleted)
    console.print(
        f"{escape(old.name)} removed; {count} {'file' if count == 1 else 'files'} deleted, "
        f"{cleanup.freed_bytes / GIB:.1f} GiB freed."
    )
    if cleanup.shared:
        console.print(
            f"[dim]Kept {len(cleanup.shared)} shared with other instances "
            f"(`mnm rm --files`' rules).[/dim]"
        )
