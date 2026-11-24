"""`mnm setup`: check every prerequisite and install what is missing."""

from __future__ import annotations

import sys
from typing import Annotated

import typer
from rich.markup import escape

from makenomistakes import config, service
from makenomistakes.apps.setup_ops import LiveApps
from makenomistakes.cli._app import app, console
from makenomistakes.cli.starter_cmd import add_starter_item
from makenomistakes.config import SECRETS_FILE, Settings
from makenomistakes.fit import Budget
from makenomistakes.hardware import HardwareReport, SubprocessRunner, probe
from makenomistakes.hardware import platform as platforms
from makenomistakes.hardware.box_facts import read_box
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.plan_budget import planning_budget
from makenomistakes.setup import (
    InteractiveRunner,
    Launcher,
    build_steps,
    looks_like_hf_token,
    run_setup,
    write_hf_token,
    write_setting,
)
from makenomistakes.setup_guide import SECTIONS, Guide, reboot_words
from makenomistakes.setup_guide import run as run_guide
from makenomistakes.setup_network import (
    ROUTER_HOST,
    decides,
    file_host,
    moved_words,
    unattended_host,
)
from makenomistakes.setup_prompt import Defaults, TerminalPrompter
from makenomistakes.setup_server import plain_lines
from makenomistakes.setup_types import Outcome, StepResult
from makenomistakes.setup_view import View, steps_table

NO_TERMINAL = (
    "No terminal, so nothing was installed: these are the checks only. Run `mnm setup` "
    "at a terminal, or `mnm setup --yes` to install unattended on the defaults "
    "(install.sh: MNM_SETUP_YES=1)."
)


@app.command()
def setup(
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be installed.")
    ] = False,
    pull: Annotated[
        bool, typer.Option(help="Also pull the llama.cpp engine image (~2 GB).")
    ] = True,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help=(
                "Unattended: every question takes its default. Without a terminal it is "
                "what lets setup install anything (else it prints the checks only)."
            ),
        ),
    ] = False,
    install_driver: Annotated[
        bool,
        typer.Option(
            "--install-driver",
            help="Native Linux: install the NVIDIA driver without asking (it needs a reboot).",
        ),
    ] = False,
    plain: Annotated[
        bool, typer.Option("--plain", help="The checks table only, no guided sections.")
    ] = False,
) -> None:
    """Set this machine up, guided at a terminal: Docker, the NVIDIA runtime (and on native
    Linux the driver), where models live, the router's network and key, tokens, the
    background services and a first model. Re-running is safe: each step checks first."""
    settings = Settings()
    report = probe(settings.expanded_models_dir())
    attended = _attended()
    if not plain and (yes or attended):
        summary_failed = _guided(settings, report, dry_run, pull, yes, install_driver)
        raise typer.Exit(code=1 if summary_failed else 0)
    # Nobody to ask and nobody said yes: the checks only (else a piped install.sh installs
    # Docker, the toolkit and the folders unasked).
    checks_only = not (yes or attended or dry_run)
    steps = build_steps(
        report,
        settings,
        runner=SubprocessRunner(),
        pull_image=pull,
        launcher=Launcher.for_checkout(),
    )
    if not (dry_run or checks_only):
        console.print(
            "Commands that need sudo are printed before they run; "
            "you may be asked for your password."
        )
    results = run_setup(
        steps,
        InteractiveRunner(),
        dry_run or checks_only,
        echo=lambda line: console.print(f"[dim]$ {escape(line)}[/dim]"),
        confirm=(lambda _question: True) if install_driver else None,
    )
    _print(results)
    if not checks_only:  # before install.sh installs the services from the secrets file
        _keep_router(settings, report, dry_run)
    for line in plain_lines():  # A server board's BIOS checklist
        console.print(escape(line))
    worst = {r.outcome for r in results}
    if checks_only and Outcome.PLANNED in worst:  # said first: a MANUAL row's exit follows
        console.print(f"[yellow]{NO_TERMINAL}[/yellow]")
    if worst & {Outcome.FAILED, Outcome.SKIPPED}:
        raise typer.Exit(code=1)
    if Outcome.MANUAL in worst:
        console.print(
            "[yellow]One step needs you (see the note), then run `mnm setup` again.[/yellow]"
        )
        raise typer.Exit(code=1)
    if Outcome.PLANNED in worst and checks_only:
        raise typer.Exit(code=1)
    if Outcome.PLANNED in worst:
        console.print("Dry run: nothing was changed. Run `mnm setup` to apply.")
        return
    if dry_run:  # everything is in place: a dry run still writes nothing, the token included
        console.print("Dry run: every step is in place; nothing to change.")
        return
    _offer_hf_token(settings)
    console.print("[green]Ready.[/green] Next: `mnm doctor --deep`, then `mnm add <model>`.")


def _keep_router(settings: Settings, report: HardwareReport, dry_run: bool) -> None:
    """The plain setup asks nothing, so the router gets the guided setup's unattended answer
    (else a script's `MNM_SETUP_YES=1` install left a native box's router on the network with
    no key); the address is saved for the services and every later run."""
    if not decides(platforms.current(), report):
        return
    host = unattended_host(settings, report)
    if host != settings.router_host:
        console.print(escape(moved_words(dry_run)))
    if not dry_run and file_host(config.SECRETS_FILE) != host:
        write_setting(config.SECRETS_FILE, ROUTER_HOST, host)


def _attended() -> bool:
    """Someone at a terminal to answer (install.sh points stdin at /dev/null without one)."""
    return sys.stdin.isatty()


def _offer_hf_token(settings: Settings) -> None:
    """Optional: a Hugging Face token for gated/private repos and faster downloads."""
    if settings.hf_token_value() is not None or not sys.stdin.isatty():
        return
    console.print(
        "\nA Hugging Face token is optional: it unlocks gated/private repos (Llama, Gemma) "
        "and lifts the anonymous rate limit. Create one at huggingface.co/settings/tokens."
    )
    token = typer.prompt(
        "Hugging Face token (Enter to skip)", default="", hide_input=True, show_default=False
    ).strip()
    if not token:
        console.print(f"[dim]Skipped. Add HF_TOKEN=hf_... to {SECRETS_FILE} any time.[/dim]")
        return
    if not looks_like_hf_token(token):
        console.print(
            "[yellow]That does not look like a token (they start with hf_); not saved.[/yellow]"
        )
        return
    write_hf_token(SECRETS_FILE, token)
    console.print(f"[green]Token saved[/green] to {SECRETS_FILE} (mode 600).")


def _print(results: list[StepResult]) -> None:
    console.print(steps_table(results))
    if any(r.reboot for r in results):
        what = reboot_words([r for r in results if r.reboot])
        console.print(
            f"[yellow]Reboot now ({escape(what)} loads at boot), then run `mnm setup` "
            "again.[/yellow]"
        )


def _guided(
    settings: Settings,
    report: HardwareReport,
    dry_run: bool,
    pull: bool,
    unattended: bool,
    install_driver: bool,
) -> bool:
    """The guided flow; True when something failed or needs the user."""
    # A dry run asks nothing: an answer it cannot keep (a key it would show) would mislead.
    prompt = Defaults() if unattended or dry_run else TerminalPrompter(console)
    guide = Guide(
        settings=settings,
        report=report,
        platform=platforms.current(),
        runner=SubprocessRunner(),
        fix_runner=InteractiveRunner(),
        prompt=prompt,
        view=View(console, SECTIONS),
        secrets_file=SECRETS_FILE,
        install_services=lambda chosen: service.install(SubprocessRunner(), chosen),
        add_model=add_starter_item,
        apps=LiveApps(),
        card_now=_card_now,
        cpu=read_box(SubprocessRunner(), report.is_wsl).cpu,  # Section 1's CPU line
        launcher=Launcher.for_checkout(),
        dry_run=dry_run,
        pull_image=pull,
        install_driver=install_driver,
    )
    run_guide(guide)
    return guide.failed or guide.summary.failed


def _card_now(settings: Settings) -> Budget | None:
    """The budget `mnm add` would plan against now (the guide asks whether it is busy)."""
    try:
        return planning_budget(InstanceService(settings))
    except InstanceError:  # a saved recipe that cannot be read: the adds say so themselves
        return None
