"""`mnm backup --auto`, `mnm backup ls`, `mnm backup schedule` and `mnm restore --test`:
the backups folder kept by itself, each backup proved to restore (`backup_auto.py`)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from makenomistakes import backup_timer
from makenomistakes.backup import many
from makenomistakes.backup_auto import (
    AutoRun,
    Said,
    auto_backup,
    check_backup,
    check_words,
    read_last,
    run_words,
)
from makenomistakes.backup_rotation import age, keep, listing, size_words
from makenomistakes.cli._app import console, fail_with
from makenomistakes.cli.backup_cmds import backup_app
from makenomistakes.cli.notify_cmd import tell
from makenomistakes.config import Settings
from makenomistakes.hardware.probe import SubprocessRunner
from makenomistakes.instances import InstanceService
from makenomistakes.local_time import local
from makenomistakes.measurements import gpu_name
from makenomistakes.nightly import timer as nightly_timer
from makenomistakes.restore_check import RestoreCheck


def _card(settings: Settings) -> str | None:
    return gpu_name(InstanceService(settings).memory)


def run_auto() -> None:
    """`mnm backup --auto`: what the timer and the web's Back up now run."""
    settings = Settings()
    done = auto_backup(settings, _card(settings), tell)
    say_auto(done)
    if not done.ok:
        raise typer.Exit(code=1)


def say_auto(done: AutoRun) -> None:
    _print(run_words(done))


def say_check(check: RestoreCheck) -> None:
    _print(check_words(check))


def _print(said: Said) -> None:
    for tone, text in said:
        console.print(f"[{tone.value}]{escape(text)}[/]" if tone.value else escape(text))


def run_test(file: Path | None) -> None:
    """`mnm restore --test [FILE]`."""
    settings = Settings()
    check = check_backup(settings, file, _card(settings), tell)
    if check.file != Path():
        console.print(f"Tested {escape(str(check.file))}:")
    say_check(check)
    if not check.ok:
        raise typer.Exit(code=1)


@backup_app.command("ls")
def backup_ls() -> None:
    """The backups in MNM_BACKUPS_DIR with their sizes and ages, what the rotation keeps each
    as, and the last restore test."""
    settings = Settings()
    folder = settings.expanded_backups_dir()
    found = listing(folder)
    if not found:
        console.print(
            f"No backups in {escape(str(folder))} yet: `mnm backup --auto` writes one, "
            f"`mnm backup schedule --at {backup_timer.DEFAULT_AT}` every night."
        )
    else:
        kept = keep(found)
        now = datetime.now()
        console.print(f"In {escape(str(folder))}:")
        table = Table(box=box.SIMPLE_HEAD)
        for column in ("Backup", "Size", "Age", "Kept as"):
            table.add_column(column)
        for item in found:
            table.add_row(
                escape(item.name),
                size_words(item.size),
                age(item.taken, now),
                kept[item.path].value if item.path in kept else "removed at the next backup",
            )
        console.print(table)
        total = sum(item.size for item in found)
        console.print(f"{many(len(found), 'backup', 'backups')}, {size_words(total)} in all.")
    last = read_last(settings.expanded_state_dir())
    if last.test is not None:
        when = f"{local(last.test.at):%Y-%m-%d %H:%M}"
        console.print(
            f"Last restore test ({when}, {escape(last.test.file or 'no file')}): "
            f"{escape(last.test.line)}"
        )
    if last.backup is not None and not last.backup.ok:
        console.print(f"[red]The last backup failed: {escape(last.backup.line)}[/red]")


@backup_app.command("schedule")
def backup_schedule(
    at: Annotated[
        str | None,
        typer.Option("--at", help="Every day at this time on the box's clock (03:30)."),
    ] = None,
    off: Annotated[bool, typer.Option("--off", help="Remove the timer.")] = False,
) -> None:
    """A systemd user timer that runs `mnm backup --auto` every night; no flag shows it."""
    runner = SubprocessRunner()
    if at is not None and off:
        fail_with(backup_timer.ScheduleError("`--at` or `--off`, not both."))
    try:
        if off:
            ran = backup_timer.unschedule(runner)
            for line in ran:
                console.print(f"[dim]{escape(line)}[/dim]")
            console.print("The nightly backup is off." if ran else "No backup timer was installed.")
            return
        if at is not None:
            hour = backup_timer.parse_at(at)
            if (nightly := nightly_timer.installed_at()) is not None:
                raise backup_timer.ScheduleError(
                    f"The nightly runs the backup at {nightly}; `mnm nightly uninstall` first "
                    "(the nightly's timer owns the backup's schedule)."
                )
            for line in backup_timer.schedule(runner, Settings(), hour):
                console.print(f"[dim]{escape(line)}[/dim]")
    except backup_timer.ScheduleError as exc:
        fail_with(exc)
    console.print(escape(backup_timer.timer_state(runner).line))
