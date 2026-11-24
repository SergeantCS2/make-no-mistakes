"""`mnm nightly`: the self-check a timer runs every night: `run`, `show`,
`ls`, `set`, and the timer's `install`, `status`, `uninstall`."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.markup import escape

from makenomistakes.backup_auto import Said
from makenomistakes.cli._app import app, console, fail_with
from makenomistakes.config import Settings
from makenomistakes.hardware.probe import SubprocessRunner
from makenomistakes.nightly import settings as nightly_settings
from makenomistakes.nightly import timer
from makenomistakes.nightly.report import NightlyStore
from makenomistakes.nightly.run import Skip
from makenomistakes.nightly.wiring import NightlyBusyError, run_nightly
from makenomistakes.nightly.words import listing_line, page
from makenomistakes.user_timer import ScheduleError, parse_at

nightly_app = typer.Typer(
    help="The nightly self-check: the doctor, every app's doctor, `try --all --speed`, a "
    "backup, the disk; one report a night and a notice only when something regressed.",
)
app.add_typer(nightly_app, name="nightly")


def _print(said: Said) -> None:
    for tone, text in said:
        console.print(f"[{tone.value}]{escape(text)}[/]" if tone.value else escape(text))


def _line(text: str) -> None:
    console.print(escape(text), highlight=False)


@nightly_app.command("run")
def run(
    now: Annotated[
        bool,
        typer.Option(
            "--now",
            help="Do not wait for the router's idle minute (it still waits for a "
            "tune, a start or stop, or a ComfyUI job to end).",
        ),
    ] = False,
    skip: Annotated[
        list[Skip] | None,
        typer.Option(
            "--skip",
            help="Leave a step out, once per step: try, apps, backup, disk, clean, doctor, "
            "freshness; deep runs the doctor without --deep.",
        ),
    ] = None,
) -> None:
    """One nightly run, as the timer runs it: the report under <state>/nightly/<date>.json."""
    try:
        report = run_nightly(Settings(), _line, now=now, skip=frozenset(skip or ()))
    except NightlyBusyError as exc:
        fail_with(exc)
    console.print()
    _print(page(report))


@nightly_app.command("show")
def show(
    date: Annotated[
        str | None, typer.Argument(help="A report's name (2026-10-26, 2026-10-26-2); the last.")
    ] = None,
) -> None:
    """A night's report: the verdicts with what mnm did or offers, then each step."""
    store = NightlyStore(Settings().expanded_state_dir())
    ids = store.ids()
    if not ids:
        console.print(
            "No nightly has run yet: `mnm nightly run --now` runs one, "
            f"`mnm nightly install` every night at {timer.UNIT.default_at}."
        )
        return
    chosen = date or ids[-1]
    if chosen not in ids:
        fail_with(LookupError(f"No nightly report {chosen!r}: `mnm nightly ls` lists them."))
    found = store.listed(chosen)
    if found.report is None:
        fail_with(ValueError(f"The nightly report {chosen} {found.problem}."))
    _print(page(found.report))


@nightly_app.command("ls")
def ls() -> None:
    """Every night's report, newest last, with one word: quiet, regression, failed, waited."""
    store = NightlyStore(Settings().expanded_state_dir())
    found = store.every()
    if not found:
        console.print("No nightly has run yet: `mnm nightly run --now` runs one.")
        return
    _print([listing_line(item) for item in found])


@nightly_app.command("set")
def set_setting(
    assignment: Annotated[
        str | None, typer.Argument(help="clean=30d (or 2w, 12h), clean=off.")
    ] = None,
) -> None:
    """The nightly's settings: `clean=<age>` turns on `mnm clean --auto` in the night."""
    state = Settings().expanded_state_dir()
    try:
        current = nightly_settings.read(state)
        if assignment is not None:
            current = nightly_settings.changed(current, assignment)
            nightly_settings.save(state, current)
    except nightly_settings.NightlySettingsError as exc:
        fail_with(exc)
    said = "Saved: " if assignment is not None else ""
    console.print(f"{said}clean: {escape(current.clean_words)}.")


@nightly_app.command("install")
def install(
    at: Annotated[
        str, typer.Option("--at", help="Every day at this time on the box's clock.")
    ] = timer.UNIT.default_at,
) -> None:
    """A systemd user timer that runs `mnm nightly run` every night; it runs the backup too,
    so the backup's own timer goes."""
    runner = SubprocessRunner()
    try:
        for line in timer.install(runner, Settings(), parse_at(at)):
            console.print(f"[dim]{escape(line)}[/dim]")
    except ScheduleError as exc:
        fail_with(exc)
    console.print(escape(timer.state(runner).line))


@nightly_app.command("status")
def status() -> None:
    """The timer, and the last night's word."""
    console.print(escape(timer.state(SubprocessRunner()).line))
    latest = NightlyStore(Settings().expanded_state_dir()).latest()
    if latest is not None:
        console.print(f"Last: {escape(latest.id)}, {latest.outcome.value} (`mnm nightly show`).")


@nightly_app.command("uninstall")
def uninstall() -> None:
    """Remove the timer; the reports stay."""
    ran = timer.uninstall(SubprocessRunner())
    for line in ran:
        console.print(f"[dim]{escape(line)}[/dim]")
    console.print("The nightly timer is removed." if ran else "No nightly timer was installed.")
