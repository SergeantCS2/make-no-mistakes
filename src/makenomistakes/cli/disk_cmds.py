"""`mnm disk`: what the models folder holds and the disk policy; `mnm disk
keep --models 85%|400G|off` sets the policy's limit, `mnm disk check` runs it now."""

from __future__ import annotations

from typing import Annotated

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from makenomistakes.cli._app import app, console, fail_with
from makenomistakes.cli.notify_cmd import tell
from makenomistakes.config import Settings
from makenomistakes.disk import disk_report, instance_rows
from makenomistakes.disk_policy import DiskPolicy, Outcome
from makenomistakes.disk_settings import DiskSettings, DiskSettingsError, warning_now, when_words
from makenomistakes.disk_types import DiskFile, DiskReport, InstanceDisk, Level, gb, parse_keep
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService

UNLISTED_SHOWN = 10
disk_app = typer.Typer(
    help="What the models folder holds, and the disk policy that keeps it under a limit.",
    invoke_without_command=True,
)
app.add_typer(disk_app, name="disk")


@disk_app.callback()
def disk(
    ctx: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="The report as JSON.")] = False,
) -> None:
    """The models folder by instance, by source and unlisted; the other folders; the policy."""
    if ctx.invoked_subcommand is not None:
        return
    service = InstanceService(Settings())
    try:
        report = disk_report(service)
    except InstanceError as exc:
        fail_with(exc)
    if as_json:
        print(report.model_dump_json(indent=2))
        return
    print_report(report)


@disk_app.command()
def keep(
    models: Annotated[
        str | None,
        typer.Option(
            "--models",
            help="The limit: a share of the models folder's disk (85%, the default), a size "
            "of the folder (400G, 1.5T; G is 10^9 bytes, GiB 2^30), or off.",
        ),
    ] = None,
) -> None:
    """Show the disk policy's limit, or set it (`--models`) and run the policy at once."""
    settings = Settings()
    files = DiskSettings(settings.expanded_state_dir())
    try:
        if models is None:
            console.print(f"Disk policy: {escape(files.read().keep.words)}.")
            return
        limit = parse_keep(models)

        def saved(current: DiskFile) -> DiskFile:  # a new limit: its warnings tell again
            return current.model_copy(update={"keep": limit, "notified": []})

        files.update(saved)
    except (ValueError, DiskSettingsError) as exc:
        fail_with(exc)
    said = f"keeps {limit.words}" if limit.on else "is off: nothing is ever deleted"
    console.print(f"Saved: the disk policy {escape(said)} ({files.path}).")
    if limit.on:
        print_outcome(_policy(InstanceService(settings)).check())


@disk_app.command()
def check() -> None:
    """Run the disk policy now: measure, warn, and at the limit evict (what `mnm add`, every
    start and the doctor run on their own)."""
    print_outcome(_policy(InstanceService(Settings())).check())


def _policy(service: InstanceService) -> DiskPolicy:
    """The policy as a command runs it: notices sent before the process ends, its journal
    lines printed by `print_outcome` instead of on stderr."""
    return DiskPolicy(service, notify=tell, say=lambda _line: None)


def print_outcome(outcome: Outcome) -> None:
    gauge = outcome.gauge
    if gauge is None:
        console.print(f"Disk policy: {escape(outcome.keep.words)}; nothing measured.")
    elif outcome.warning:
        said = outcome.warning[0].upper() + outcome.warning[1:]
        console.print(f"[yellow]{escape(said)}.[/yellow]")
    else:
        console.print(f"Disk policy: {escape(outcome.keep.words)}; {escape(gauge.subject)}.")
    for line in outcome.lines:
        console.print(f"[yellow]{escape(line)}[/yellow]")
    if outcome.refused:
        console.print(f"[red]{escape(outcome.refused)}[/red]")


def print_warning(settings: Settings) -> None:
    """The policy's line under `mnm add`'s plan and `mnm ls`, from the first warning on."""
    line = warning_now(settings.expanded_state_dir(), settings.expanded_models_dir())
    if line is not None:
        console.print(f"[yellow]Disk:[/yellow] {escape(line)}")


def print_disk(service: InstanceService, rows: bool) -> None:
    """Under `mnm ls`: the policy's warning, and with `--disk` each instance's files."""
    if rows:
        try:
            found, notes = instance_rows(service)
        except InstanceError as exc:
            fail_with(exc)
        console.print(instances_table(found))
        for note in notes:
            console.print(f"[yellow]Note:[/yellow] {escape(note)}")
    print_warning(service.settings)


def print_report(report: DiskReport) -> None:
    held = gb(report.held_bytes)
    console.print(f"Models folder {escape(report.models_dir)}: holds {held}; {disk_words(report)}.")
    console.print(instances_table(report.instances))
    sources = Table(title="by folder", box=box.SIMPLE_HEAD)
    for column in ("Folder", "Size", "Files"):
        sources.add_column(column, overflow="fold")
    for found in report.sources:
        sources.add_row(escape(found.label), gb(found.size_bytes), str(found.files))
    for found in report.workflows:
        sources.add_row(f"  {escape(found.label)}", gb(found.size_bytes), str(found.files))
    for found in report.other:
        sources.add_row(
            f"{escape(found.label)} (not mnm's)", gb(found.size_bytes), str(found.files)
        )
    console.print(sources)
    _print_unlisted(report)
    folders = Table(title="other folders", box=box.SIMPLE_HEAD)
    for column in ("Folder", "Path", "Size"):
        folders.add_column(column, overflow="fold")
    for found in report.folders:
        size = gb(found.size_bytes) if found.exists else "not there"
        folders.add_row(escape(found.label), escape(found.path), size)
    console.print(folders)
    _print_policy(report)
    for note in report.notes:
        console.print(f"[yellow]Note:[/yellow] {escape(note)}")


def disk_words(report: DiskReport) -> str:
    """`its disk: 39.2 GB used, 2.3 GB free (94%)`, as `df` counts it (the root's reserve left
    out of both)."""
    total, free = report.disk_total_bytes, report.disk_free_bytes
    if not total:
        return "its disk could not be read"
    share = 100 * (total - free) / total
    return f"its disk: {gb(total - free)} used, {gb(free)} free ({share:.0f}%)"


def instances_table(rows: list[InstanceDisk]) -> Table:
    """By instance: its own files, what it shares (counted under each user, once on disk)."""
    table = Table(title="by instance", box=box.SIMPLE_HEAD)
    for column in ("Name", "Own", "Shared", "Saved prompts", "Files", "State", "Last used"):
        table.add_column(column, overflow="fold")
    for row in rows:
        users = sorted({u for f in row.shared for u in f.users})
        shared = f"{gb(row.shared_bytes)} with {', '.join(users)}" if row.shared else "-"
        used = f"{when_words(row.last_used)} ({row.last_used_from})"
        prompts = gb(row.prompt_bytes) if row.prompt_bytes else "-"
        table.add_row(
            escape(row.name),
            gb(row.own_bytes),
            escape(shared),
            prompts,
            str(row.files),
            row.state,
            used,
        )
    return table


def _print_unlisted(report: DiskReport) -> None:
    if not report.unlisted:
        console.print("Unlisted: none (every file in the download folders is listed).")
        return
    console.print(
        f"Unlisted: {len(report.unlisted)} files, {gb(report.unlisted_bytes)} that no instance, "
        "workflow or LoRA lists (what `mnm rm` without `--files` leaves, a download cut short); "
        "the disk policy never deletes them."
    )
    for found in report.unlisted[:UNLISTED_SHOWN]:
        console.print(f"  {gb(found.size_bytes):>9}  {escape(found.path)} [dim]({found.why})[/dim]")
    if len(report.unlisted) > UNLISTED_SHOWN:
        console.print(f"  … and {len(report.unlisted) - UNLISTED_SHOWN} more (`mnm disk --json`)")


def _print_policy(report: DiskReport) -> None:
    policy = report.policy
    gauge = policy.gauge
    if not policy.keep.on:
        console.print("Disk policy: off (`mnm disk keep --models 85%` turns it on).")
    elif gauge is not None:
        style = {Level.OK: "green", Level.WARN: "yellow", Level.OVER: "red"}[policy.level]
        first, second = gauge.warnings
        console.print(
            f"Disk policy: {escape(policy.keep.words)}: [{style}]{escape(gauge.subject)}[/{style}]"
            f" (warnings from {first:g}% and {second:g}%; files go at {gauge.limit:g}%, until "
            f"under {gauge.target:g}%)."
        )
    if policy.keep.on:
        order = ", ".join(policy.next) or "none"
        console.print(f"  Next to go, least recently used first: {escape(order)}.")
        for why in policy.held:
            console.print(f"  [dim]Stays: {escape(why)}[/dim]")
    for record in policy.evictions[-10:]:
        console.print(
            f"  Evicted {when_words(record.at)}: {escape(record.name)}, {gb(record.freed_bytes)} "
            f"({record.files} file{'s' if record.files != 1 else ''}) at {record.where}"
        )
