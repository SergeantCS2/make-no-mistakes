"""`mnm update` and `mnm service`: keep the checkout and the web UI current without thinking."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.markup import escape

from makenomistakes import backup_timer
from makenomistakes import service as svc
from makenomistakes.cli._app import app, console, fail_with, print_raw
from makenomistakes.config import Settings
from makenomistakes.consumers import shared_base_url
from makenomistakes.doctor_types import Status
from makenomistakes.hardware import platform as platforms
from makenomistakes.hardware.probe import SubprocessRunner
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.machine_checks import outputs_check
from makenomistakes.nightly import timer as nightly_timer
from makenomistakes.service_ready import wait_until_up, words
from makenomistakes.service_watch import logs_remedy, unit_restarts

service_app = typer.Typer(help="Run the web UI as a background service (systemd --user).")
app.add_typer(service_app, name="service")


@app.command()
def update() -> None:
    """Pull the latest code, install dependencies, restart the web UI and router services."""
    runner = SubprocessRunner()
    try:
        root = svc.repo_root()
        state = svc.systemd_state(runner)
        report = svc.update(runner, root, svc.uv_binary(), state.installed)
    except svc.ServiceError as exc:
        fail_with(exc)
    if report.after == report.before:
        console.print(f"Already up to date ({report.after}).")
    else:
        console.print(f"[green]Updated {report.before} → {report.after}[/green]")
        for line in report.changes:
            console.print(f"  {escape(line)}")
    if report.service_restarted:
        web_port = Settings().port  # the web UI's own, not an instance's
        console.print(
            f"Web UI and router restarted: http://127.0.0.1:{web_port} and {shared_base_url()}"
        )
    for note in report.notes:
        console.print(f"[yellow]{escape(note)}[/yellow]")
    _running_note()
    _outputs_note()
    console.print(
        "[dim]Engine builds (llama.cpp, stable-diffusion.cpp, whisper.cpp, audio.cpp) update "
        "separately: "
        "`mnm engines update`.[/dim]"
    )


def _outputs_note() -> None:
    """The doctor's outputs-folder warning: a box updated from an older mnm has no such
    folder until `mnm setup` makes it, and every run would fail on it."""
    check = outputs_check(Settings())
    if check.status is not Status.OK:
        what = check.remedy or check.detail  # the remedy says what is wrong, whole
        console.print(f"[yellow]{escape(check.name)}: {escape(what)}[/yellow]")


def _running_note() -> None:
    try:
        running = InstanceService(Settings()).running_names()
    except InstanceError as exc:
        console.print(f"[yellow]{escape(str(exc))}[/yellow]")
        return
    if running:
        names = ", ".join(escape(n) for n in running)
        console.print(
            f"Running instances keep running: {names}. Restart one (`mnm stop` then `mnm start`) "
            "only when its settings changed or the update notes say so."
        )


@service_app.command("install")
def service_install() -> None:
    """Install and start the web UI as a user service that survives logout and reboot."""
    runner = SubprocessRunner()
    try:
        ran = svc.install(runner, Settings(), told=lambda line: console.print(escape(line)))
    except svc.ServiceError as exc:
        fail_with(exc)
    for line in ran:
        console.print(f"[dim]{escape(line)}[/dim]")
    settings = Settings()
    console.print("Waiting for the web UI and the router to answer …")
    ready = wait_until_up(settings, runner)
    for good, line in words(ready, settings):
        console.print(f"[{'green' if good else 'yellow'}]{escape(line)}[/]")
    console.print(
        f"Both start with {platforms.current().starts_with}; `mnm update` restarts them; "
        "`mnm service status` shows them."
    )
    if not ready.all_up:
        raise typer.Exit(code=1)


@service_app.command("status")
def service_status() -> None:
    """Are the web UI and router services installed, enabled and running?"""
    runner = SubprocessRunner()
    states = {unit: svc.systemd_state(runner, unit=unit) for unit in svc.UNITS}
    first = states[svc.UNIT_NAME]
    if not first.available:
        console.print(f"[yellow]systemd --user is not available ({escape(first.detail)}).[/yellow]")
        console.print(escape(svc.systemd_note(first)))
        raise typer.Exit(code=1)
    for unit, state in states.items():
        console.print(
            f"{unit}: installed: {state.installed}  enabled: {state.enabled}  "
            f"active: {state.active}"
        )
        restarts = unit_restarts(runner, unit)  # systemd's own count, in the hour
        if (said := restarts.words()) is not None:
            tone = "yellow" if restarts.over else "dim"
            console.print(f"  [{tone}]{escape(said)}[/{tone}]")
            if restarts.over:
                console.print(f"  {escape(logs_remedy(unit))}")
    console.print(escape(nightly_timer.state(runner).line))  # The nightly
    console.print(escape(backup_timer.timer_state(runner).line))  # The nightly backup
    if not all(state.installed for state in states.values()):
        console.print("Run `mnm service install` to start the web UI and the router automatically.")
    elif down := [unit for unit, state in states.items() if not state.active]:
        console.print(
            f"Not running: `systemctl --user restart {' '.join(down)}` or `mnm service logs`."
        )


@service_app.command("logs")
def service_logs(
    lines: Annotated[int, typer.Option("--lines", "-n", help="How many lines.")] = 100,
) -> None:
    """The web UI and router services' recent log lines, interleaved (journalctl --user)."""
    print_raw(svc.journal(SubprocessRunner(), lines))


@service_app.command("uninstall")
def service_uninstall() -> None:
    """Stop and remove the web UI service (the instances and models stay)."""
    ran = svc.uninstall(SubprocessRunner())
    for line in ran:
        console.print(f"[dim]{escape(line)}[/dim]")
    console.print("Removed. Start the UI by hand with `mnm serve` when you need it.")
