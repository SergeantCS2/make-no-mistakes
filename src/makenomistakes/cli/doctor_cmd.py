"""`mnm doctor`."""

from __future__ import annotations

import json
from typing import Annotated, NoReturn

import typer
from rich.markup import escape
from rich.table import Table

from makenomistakes.apps.doctor_run import app_doctor
from makenomistakes.apps.names import LABELS, App
from makenomistakes.cli._app import app, console, fail_with
from makenomistakes.cli.doctor_from import print_script, run_from
from makenomistakes.client_view import From
from makenomistakes.config import Settings
from makenomistakes.doctor import Check, Status, worst
from makenomistakes.doctor_deep import deep_tail, json_report
from makenomistakes.hardware import SubprocessRunner, probe
from makenomistakes.hardware.box_facts import read_box
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.machine_checks import machine_checks
from makenomistakes.remote_script import Shell
from makenomistakes.store import StoreError

_STATUS_STYLE = {Status.OK: "green", Status.WARN: "yellow", Status.FAIL: "red"}


@app.command()
def doctor(
    deep: Annotated[
        bool,
        typer.Option(
            help="Also run a GPU container smoke test (pulls ~100 MB) and measure the RAM's "
            "bandwidth (a few seconds, 2 GiB of RAM)."
        ),
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    app_name: Annotated[
        App | None,
        typer.Option(
            "--app",
            help="Check the router from a workspace app's side instead: its model ids, "
            "a tool call, embeddings, MCP, the context floor and the key.",
        ),
    ] = None,
    container: Annotated[
        str | None,
        typer.Option(help="With --app: the app's container, when mnm did not name it."),
    ] = None,
    where: Annotated[
        From | None,
        typer.Option(
            "--from",
            help="Check the router as a client sees it: from here, from a container "
            "here (docker), or at its LAN address (lan; on a client install, MNM_ROUTER_URL).",
        ),
    ] = None,
    remote_script: Annotated[
        bool,
        typer.Option(
            "--remote-script",
            help="Print the same checks as a script for another machine (sh + curl, or "
            "--shell powershell); it reads MNM_API_KEY from its own environment.",
        ),
    ] = False,
    shell: Annotated[
        Shell, typer.Option(help="With --remote-script: sh or powershell.")
    ] = Shell.SH,
) -> None:
    """Check GPU, RAM, WSL networking, Docker, storage and token."""
    settings = Settings()
    picked = [
        flag
        for flag, on in (("--remote-script", remote_script), ("--from", where), ("--app", app_name))
        if on
    ]
    if len(picked) > 1:  # each answers on its own and exits (apps-setup-cli 14)
        raise typer.BadParameter(f"{' and '.join(picked)} each run alone: pick one.")
    if remote_script:
        print_script(settings, shell)
    if where is not None:
        run_from(settings, where, as_json, _print_checks)
    if app_name is not None:
        _app(settings, app_name, container, as_json)
    report = probe(settings.expanded_models_dir(), docker_gpu_smoke=deep, host_ram=True)
    box = read_box(SubprocessRunner(), report.is_wsl)
    checks = machine_checks(settings, report, box, deep)
    if deep:  # the RAM's bandwidth, saved for the planner, and Experts in RAM
        checks += deep_tail(settings, report)
    if as_json:
        payload = {"report": json_report(report), "checks": [c.model_dump() for c in checks]}
        console.print_json(json.dumps(payload))
    else:
        _print_checks(checks)
    raise typer.Exit(code=1 if worst(checks) is Status.FAIL else 0)


def _app(settings: Settings, which: App, container: str | None, as_json: bool) -> NoReturn:
    """`mnm doctor --app`: where it asks from first, then the checks; exit 1 on a
    FAIL, 2 when mnm's own files cannot be read (said as every command says it)."""

    def where(said: str) -> None:
        if not as_json:
            console.print(f"{LABELS[which]}, {escape(said)}:")

    try:
        found = app_doctor(which, InstanceService(settings), container, before=where)
    except (InstanceError, StoreError) as exc:
        fail_with(exc)
    if as_json:
        payload = {"where": found.where, "checks": [c.model_dump() for c in found.checks]}
        console.print_json(json.dumps(payload))
    else:
        _print_checks(found.checks, f"mnm doctor --app {which.value}")
    raise typer.Exit(code=1 if worst(found.checks) is Status.FAIL else 0)


def _print_checks(checks: list[Check], title: str = "mnm doctor") -> None:
    table = Table(title=title, show_lines=False)
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")  # a path whole, never cut with "…"
    for check in checks:
        style = _STATUS_STYLE[check.status]
        detail = escape(check.detail)
        if check.remedy is not None:
            detail += f"\n[dim]→ {escape(check.remedy)}[/dim]"
        table.add_row(check.name, f"[{style}]{check.status.value.upper()}[/{style}]", detail)
    console.print(table)
