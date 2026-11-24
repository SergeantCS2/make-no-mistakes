"""`mnm setup`, guided: eight sections, each asking only what it needs and
saying what it did, then a summary of what is left.

1. This machine: what was found (platform, each card and its link, RAM and its measured
   bandwidth, CPU, disk), what large models the box can hold (`setup_hardware.py`),
   and on a native box with several cards which ones mnm uses (`MNM_GPU`).
2. Where models live: the folder, and the space there.
3. Network: who may reach the router, its key, a firewall in the way.
4. Prerequisites: the plain setup's steps (Docker, the NVIDIA runtime, a native box's
   driver), shown first, run once agreed.
5. Accounts: the Hugging Face and Civitai tokens, both optional.
6. Background services: the web UI and the router as systemd user units.
7. First models: a starter set that fits the card.
8. Workspace apps (`setup_apps.py`): OpenCode, LibreChat, OpenHands installed here or
   connected from another machine, Goose's file, Odysseus connected when ~/odysseus exists.

Every answer has a default, so `--yes` runs the same flow unattended (the CLI gives a run
without a terminal the plain table instead); the driver is installed then only with
`--install-driver`, and no token, key, model or app is added. Settings go to the secrets file
(`write_setting`), which the services and every terminal read. Nothing is written in a dry
run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from makenomistakes.busy_card import held_elsewhere
from makenomistakes.config import Settings
from makenomistakes.consumers import shared_base_url
from makenomistakes.fit import Budget
from makenomistakes.hardware import CommandRunner, HardwareReport
from makenomistakes.hardware.cpu_topology import CpuTopology
from makenomistakes.hardware.platform import Platform, nvidia_pci_devices
from makenomistakes.setup import (
    Launcher,
    build_steps,
    current_user,
    looks_like_hf_token,
    run_setup,
    write_setting,
)
from makenomistakes.setup_apps import workspace_apps
from makenomistakes.setup_docker import Docker
from makenomistakes.setup_folders import models_folder
from makenomistakes.setup_hardware import cards_vram, machine
from makenomistakes.setup_network import network
from makenomistakes.setup_prompt import Prompter
from makenomistakes.setup_types import Outcome, Step, StepResult
from makenomistakes.setup_view import Mark, Summary, View
from makenomistakes.starter import Item, Starter, download_gb, fitting, floor_words, parse_picks

if TYPE_CHECKING:
    from makenomistakes.apps.setup_ops import AppsSetup

GIB = 1024**3
SECTIONS = 8


@dataclass
class Guide:
    """Everything the flow touches, injected: tests drive it with fakes."""

    settings: Settings
    report: HardwareReport
    platform: Platform
    runner: CommandRunner  # checks, answered quietly
    fix_runner: CommandRunner  # fixes, with the terminal attached (sudo prompts)
    prompt: Prompter
    view: View
    secrets_file: Path
    install_services: Callable[[Settings], list[str]]
    add_model: Callable[[Item], bool]
    # Section 8's installs and connects (`apps/setup_ops.LiveApps`); None offers none.
    apps: AppsSetup | None = None
    launcher: Launcher | None = None
    pci_devices: Callable[[], list[str]] = nvidia_pci_devices
    # The budget a plan would get now (mnm's own instances count as free): `busy_card` says
    # whether something else holds the card. None: it cannot be read.
    card_now: Callable[[Settings], Budget | None] = lambda _settings: None
    cpu: CpuTopology | None = None  # Section 1's CPU line (the CLI reads lscpu)
    dry_run: bool = False
    pull_image: bool = True
    install_driver: bool = False  # unattended consent for the driver
    summary: Summary = field(default_factory=Summary)
    reboot: bool = False
    reboot_for: str = "the NVIDIA driver"  # What loads at the boot (an AMD box's kernel…)
    relogin: bool = False
    failed: bool = False
    services_installed: bool = False  # this run installed them (a changed folder reinstalls)

    def relogin_pending(self) -> bool:
        """Docker answers this shell only after a re-login (its group was just added)."""
        return Docker(
            self.runner, lambda _p: "", current_user(), platform=self.platform
        ).relogin_pending()

    def save(self, name: str, value: str) -> None:
        if not self.dry_run:
            write_setting(self.secrets_file, name, value)


def run(guide: Guide) -> Summary:
    if guide.dry_run:
        subtitle = "Dry run: what setup would do here; nothing is asked or changed."
    elif guide.prompt.interactive:
        subtitle = "Guided setup: every step checks first and asks before it changes anything."
    else:
        subtitle = "Setup, unattended: every question takes its default."
    guide.view.banner(subtitle)
    machine(guide)
    models_folder(guide)
    network(guide)
    prerequisites(guide)
    accounts(guide)
    services(guide)
    first_models(guide)
    workspace_apps(guide)
    closing(guide)
    guide.view.summary(guide.summary)
    return guide.summary


# --- 1: setup_hardware.py ------------------------------------------------------------


# --- 2: setup_folders.py ----------------------------------------------------------


# --- 3: setup_network.py ------------------------------------------------------------


# --- 4 --------------------------------------------------------------------------------


def prerequisites(guide: Guide) -> None:
    guide.view.section(
        4,
        "Prerequisites",
        "Checked first; anything missing is installed from the vendors' apt repositories "
        "(commands are printed before they run, and sudo may ask for your password).",
    )
    plan = run_setup(_steps(guide), guide.runner, dry_run=True)
    guide.view.steps(plan, "What is there, and what would be installed", commands=False)
    missing = [r for r in plan if r.outcome is Outcome.PLANNED]
    # What waits on a reboot or a re-login is known before anything runs (a driver installed
    # earlier but not loaded yet, a group added by an earlier run): the services wait too.
    guide.reboot = any(r.reboot for r in plan)
    guide.reboot_for = reboot_words([r for r in plan if r.reboot])
    if guide.dry_run:
        guide.summary.add(Mark.SKIP, "Prerequisites", f"dry run: {len(missing)} to install")
        return
    guide.relogin = guide.relogin_pending()
    if not missing and not any(r.outcome is Outcome.MANUAL for r in plan):
        guide.summary.add(Mark.DONE, "Prerequisites", "everything in place")
        _waits(guide)
        return
    if missing and not guide.prompt.yes(f"Install the {len(missing)} missing pieces now?", True):
        guide.summary.add(Mark.SKIP, "Prerequisites", "not installed (run `mnm setup` again)")
        guide.summary.todo.extend(
            f"{r.name}: {r.note}" for r in plan if r.outcome is Outcome.MANUAL and r.note
        )
        _waits(guide)
        guide.failed = True
        return
    results = run_setup(
        _steps(guide),  # built again: the checks above may have changed nothing, or something
        guide.fix_runner,
        dry_run=False,
        echo=guide.view.command,
        confirm=lambda question: (
            guide.prompt.yes(question, True) if guide.prompt.interactive else guide.install_driver
        ),
    )
    guide.view.steps(results, "Prerequisites")
    _record(guide, results)


def _steps(guide: Guide) -> list[Step]:
    return build_steps(
        guide.report,
        guide.settings,
        runner=guide.runner,
        pull_image=guide.pull_image,
        launcher=guide.launcher,
        platform=guide.platform,
        pci_devices=guide.pci_devices,
    )


def _record(guide: Guide, results: list[StepResult]) -> None:
    bad = [r for r in results if r.outcome in (Outcome.FAILED, Outcome.SKIPPED)]
    manual = [r for r in results if r.outcome is Outcome.MANUAL]
    if any(r.reboot for r in results):
        guide.reboot, guide.reboot_for = True, reboot_words([r for r in results if r.reboot])
    guide.relogin = guide.relogin_pending()
    for r in manual:
        guide.summary.todo.append(f"{r.name}: {r.note}")
    if bad:
        guide.failed = True
        names = ", ".join(r.name for r in bad)
        guide.summary.add(Mark.FAIL, "Prerequisites", f"failed: {names} (see the table)")
    elif manual:
        guide.summary.add(Mark.NOTE, "Prerequisites", f"{len(manual)} step(s) need you")
    else:
        guide.summary.add(Mark.DONE, "Prerequisites", "installed")
    _waits(guide)


def reboot_words(results: list[StepResult]) -> str:
    """What a reboot loads: the NVIDIA driver as before, or an AMD box's steps."""
    if not results or any(r.name.startswith("NVIDIA") for r in results):
        return "the NVIDIA driver"
    return " and ".join(dict.fromkeys(_REBOOT_WORDS.get(r.name, r.name) for r in results))


_REBOOT_WORDS = {
    "Kernel": "the new kernel",
    "AMD driver": "the card's firmware",
    "APU memory (GTT)": "the card's share of the RAM",
}


def _waits(guide: Guide) -> None:
    """A pending reboot or re-login goes first in the to-do list."""
    if guide.reboot:
        guide.summary.todo.insert(
            0, f"Reboot ({guide.reboot_for} loads at boot), then `mnm setup`."
        )
    elif guide.relogin:
        guide.summary.todo.insert(0, f"For Docker without sudo: {guide.platform.relogin}.")


# --- 5 --------------------------------------------------------------------------------


def accounts(guide: Guide) -> None:
    guide.view.section(
        5,
        "Accounts (optional)",
        "Tokens unlock gated Hugging Face repos and Civitai downloads that need a login. "
        "Enter skips; add them to the secrets file any time.",
    )
    if guide.dry_run or not guide.prompt.interactive:
        guide.summary.add(Mark.SKIP, "Accounts", "skipped")
        return
    saved: list[str] = []
    if guide.settings.hf_token_value() is None:
        token = guide.prompt.text("Hugging Face token (huggingface.co/settings/tokens)", "", True)
        if token and looks_like_hf_token(token):
            guide.save("HF_TOKEN", token)
            saved.append("Hugging Face")
        elif token:
            guide.view.say("That does not look like a token (they start with hf_); not saved.")
    else:
        saved.append("Hugging Face (already set)")
    if guide.settings.civitai_token_value() is None:
        token = guide.prompt.text("Civitai API key (civitai.com, Account settings)", "", True)
        if token:
            guide.save("CIVITAI_TOKEN", token)
            saved.append("Civitai")
    else:
        saved.append("Civitai (already set)")
    guide.summary.add(
        Mark.DONE if saved else Mark.SKIP, "Accounts", ", ".join(saved) or "no tokens (public only)"
    )


# --- 6 --------------------------------------------------------------------------------


def services(guide: Guide) -> None:
    guide.view.section(
        6,
        "Background services",
        "The web UI (port 7800) and the router (7801) as systemd user units: they start with "
        "the machine and `mnm update` restarts them.",
    )
    if guide.dry_run:
        guide.summary.add(Mark.SKIP, "Services", "dry run")
        return
    if guide.reboot or guide.relogin:
        when = "reboot" if guide.reboot else "re-login"
        guide.summary.add(Mark.NOTE, "Services", f"after the {when}: `mnm service install`")
        guide.summary.todo.append("Then `mnm service install` (the web UI and the router).")
        return
    if not guide.prompt.yes("Install and start them now?", True):
        guide.summary.add(Mark.SKIP, "Services", "not installed (`mnm service install`)")
        return
    try:
        ran = guide.install_services(guide.settings)
    except Exception as exc:
        guide.view.say(str(exc), style="yellow")
        guide.summary.add(Mark.NOTE, "Services", "not started (see the note above)")
        guide.summary.todo.append(f"Services: {exc}")
        return
    for line in ran:
        guide.view.command(line)
    guide.services_installed = True
    guide.summary.add(
        Mark.DONE,
        "Services",
        f"the web UI on port {guide.settings.port}, the router on {guide.settings.router_port}",
    )


# --- 7 --------------------------------------------------------------------------------


def first_models(guide: Guide) -> None:
    guide.view.section(
        7,
        "First models",
        "Each is an ordinary `mnm add`: planned for this card, saved, downloaded. Add more "
        "any time (`mnm starter`, or `mnm add <link>`).",
    )
    if guide.dry_run or not guide.prompt.interactive:
        guide.summary.add(Mark.SKIP, "First models", "none added (`mnm starter`)")
        return
    gpus = guide.report.gpus
    gpu = gpus[0] if isinstance(gpus, list) and gpus else None
    if gpu is None:
        guide.summary.add(Mark.SKIP, "First models", "after the driver: `mnm starter`")
        guide.summary.todo.append("Pick your first models: `mnm starter`.")
        return
    memory = guide.report.memory
    offered = fitting(
        gpu.vram_total_bytes,
        memory.total_bytes if memory is not None else None,
        cards_vram(guide),  # Every card mnm uses, for the large starters
    )
    picked = choose_starters(guide, offered)
    if not picked:
        guide.summary.add(Mark.SKIP, "First models", "none (`mnm starter` any time)")
        return
    busy = card_still_busy(guide)
    added = [item.name for s in picked for item in s.items if guide.add_model(item)]
    wanted = [item.name for s in picked for item in s.items]
    left = [name for name in wanted if name not in added]
    detail = ", ".join(added) + (f"; not added: {', '.join(left)}" if left else "") or "none"
    if busy is not None:
        detail += f" (planned with {busy / GIB:.1f} GiB of the card held elsewhere)"
        guide.summary.todo.append(
            "The models were planned while another program held the card: `mnm replan "
            "--all` plans them for the whole card when more of it is free (and `mnm starter` "
            "adds what did not fit)."
        )
    guide.summary.add(Mark.NOTE if left or busy is not None else Mark.DONE, "First models", detail)


LOOKS = 3  # times the guide offers to look again before it plans around a busy card


def card_still_busy(guide: Guide) -> int | None:
    """Before the plans: when something else holds the card (a program holding 8 GiB would
    size every plan around it for good), say so and offer to look again. The bytes still
    held when the plans go ahead, else None."""
    for asked in range(LOOKS + 1):
        budget = guide.card_now(guide.settings)
        held = held_elsewhere(budget) if budget is not None else None
        if budget is None or held is None:
            return None
        if asked == LOOKS:  # looked as often as offered: say so, and plan around it
            guide.view.say(f"{held}: planning around it.", style="yellow")
            return budget.busy_bytes
        guide.view.say(
            f"{held} right now. Models planned now are sized around it and keep those "
            "settings until `mnm replan`.",
            style="yellow",
        )
        if not guide.prompt.yes("Look again now?", True):
            return budget.busy_bytes
    return None


def choose_starters(guide: Guide, offered: list[Starter]) -> list[Starter]:
    for number, starter in enumerate(offered, 1):
        names = ", ".join(item.name for item in starter.items)
        guide.view.say(
            f" {number}. {starter.title:<13} {names:<12} ~{starter.download_gb:>4.1f} GB, "
            f"{floor_words(starter):<26}  {starter.what}"
        )
        if starter.note:
            guide.view.say(f"    {starter.note}", style="yellow")
    for _ in range(3):
        answer = guide.prompt.text("Which? Numbers or names (Enter: 1, chat; `none` to skip)", "")
        picked = parse_picks(answer, offered)
        if picked is None:
            guide.view.say("Pick from the list above, such as `1 5` or `chat memory`.")
            continue
        if not picked:
            return []
        total = download_gb(picked)
        if guide.prompt.yes(f"Download about {total:.1f} GB now?", True):
            return picked
        return []
    return []


# --- summary ----------------------------------------------------------------------------


def closing(guide: Guide) -> None:
    """The addresses to paste and the commands after this, at the end of the summary."""
    if guide.dry_run:
        guide.summary.todo.append("Dry run: nothing was changed. Run `mnm setup` to apply.")
        return
    guide.summary.todo.extend(
        [
            "Check everything: `mnm doctor` (and `mnm doctor --deep` once Docker sees the GPU).",
            f"Apps on this machine: {shared_base_url()} (the instance name is the model, or a "
            "desk role); `mnm apps ls` lists the workspace apps, `mnm url <name>` prints every "
            "card, `mnm apps connect odysseus` links Odysseus.",
            "Start a model: `mnm start <name>` (or just send it a request: the router loads it).",
            'Make things: an instance\'s Run tab in the web UI, or `mnm run <name> "…"`; a '
            "short film: `mnm pipeline run`.",
            "Optional: ComfyUI workflows need `mnm comfyui install` (or System → ComfyUI): "
            "about 9 GB, built here, nothing until you ask.",
        ]
    )
