"""Section 8 of the guided setup: the workspace apps, each optional. The user picks any
of OpenCode, LibreChat, Open WebUI, OpenHands, Goose and (when ~/odysseus exists) Odysseus; for
each, "here" installs it through `mnm apps install`'s own code (or writes Goose's file, or shows
Odysseus's `.env` diff and writes it only on yes), and "another machine" prints the connect card
(the router's LAN address, the key required; nothing here changes). The projects folder
(`MNM_PROJECTS_DIR`) is asked once, before the first app that works in it; LibreChat's first
account right after it installs, its password never kept; Open WebUI's first account is made on
its own page (the first sign-up is its admin), which setup says to do now. Unattended (`--yes`)
and in a dry run nothing is installed or asked; a failed app is said and left for `mnm apps`,
never the end of setup."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from makenomistakes.apps.defs import DEFS
from makenomistakes.apps.names import LABELS, App
from makenomistakes.apps.offer import USES_PROJECTS, Offered, offered, parse_picks
from makenomistakes.apps.open_webui_accounts import first_account
from makenomistakes.apps.projects import DEFAULT_PROJECTS, projects_problem
from makenomistakes.apps.users import ask_password
from makenomistakes.instance_types import InstanceError
from makenomistakes.setup_folders import folder_problem
from makenomistakes.setup_view import Mark
from makenomistakes.store import StoreError

if TYPE_CHECKING:
    from makenomistakes.apps.setup_ops import AppsSetup
    from makenomistakes.setup_guide import Guide

NUMBER = 8
TITLE = "Workspace apps (optional)"
ASKS = 3  # a bad answer is asked again this often
LATER = "`mnm apps install <app>`, `mnm apps connect <app>`, or the web's Get started"


@dataclass
class _Run:
    guide: Guide
    apps: AppsSetup
    projects_asked: bool = False
    done: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def say(self, text: str, style: str | None = None) -> None:
        self.guide.view.say(text, style=style)

    def progress(self, text: str) -> None:
        self.guide.view.say(f"  {text}", style="dim")


def workspace_apps(guide: Guide) -> None:
    guide.view.section(
        NUMBER,
        TITLE,
        "Apps that work with the models: installed here at the versions mnm was tested with, "
        "on 127.0.0.1 only (a Windows browser reaches them through localhost), or connected "
        "from another machine. Enter skips; " + LATER + " any time.",
    )
    projects = guide.settings.expanded_projects_dir()
    if (problem := projects_problem(projects, guide.settings)) is not None:
        guide.view.say(f"MNM_PROJECTS_DIR: {problem}", style="yellow")  # said even unasked
        guide.summary.todo.append(f"MNM_PROJECTS_DIR: {problem}")
    if guide.dry_run:
        guide.summary.add(Mark.SKIP, "Workspace apps", "dry run")
        return
    if not guide.prompt.interactive or guide.apps is None:
        guide.summary.add(Mark.SKIP, "Workspace apps", f"none ({LATER})")
        return
    if guide.reboot or guide.relogin:
        when = "reboot" if guide.reboot else "re-login"
        guide.summary.add(Mark.NOTE, "Workspace apps", f"after the {when}: {LATER}")
        guide.summary.todo.append(f"Workspace apps, if you want them: {LATER}.")
        return
    run = _Run(guide, guide.apps)
    folder = run.apps.odysseus()
    offers = offered(folder is not None)
    for offer in offers:
        run.say(f" {offer.number}. {offer.label}")
        run.say(f"    {offer.paragraph(folder or _home_odysseus())}", style="dim")
    if folder is None:
        run.say(
            f"    (Odysseus: nothing at {_home_odysseus()}; mnm never installs it. "
            "`mnm apps connect odysseus` links it once it is there.)",
            style="dim",
        )
    picked = _pick(run, offers)
    if not picked:
        guide.summary.add(Mark.SKIP, "Workspace apps", f"none ({LATER})")
        return
    for offer in picked:
        try:
            run.done.append(_one(run, offer.app))
        except (InstanceError, StoreError, OSError) as exc:  # AppsError is an InstanceError
            run.say(f"{offer.label}: {exc}", style="yellow")
            run.failed.append(offer.label)
            guide.summary.todo.append(f"{offer.label}: {exc}")
    detail = "; ".join(run.done)
    if run.failed:
        detail += ("; " if detail else "") + f"not done: {', '.join(run.failed)} (see above)"
    guide.summary.add(Mark.NOTE if run.failed else Mark.DONE, "Workspace apps", detail)


def _home_odysseus() -> Path:
    return Path("~/odysseus")


def _pick(run: _Run, offers: list[Offered]) -> list[Offered]:
    for _ in range(ASKS):
        answer = run.guide.prompt.text("Which apps? Numbers or names (Enter: none)", "")
        picked = parse_picks(answer, offers)
        if picked is not None:
            return picked
        run.say("Pick from the list above, such as `1 2` or `opencode librechat`.")
    return []


def _one(run: _Run, app: App) -> str:
    """One app: installed, written or its card shown; the summary's words for it."""
    label = LABELS[app]
    settings = run.guide.settings
    if app is App.ODYSSEUS:
        return _odysseus(run)
    here = (
        "write its provider file on this machine (Goose's CLI here)"
        if app is App.GOOSE
        else f"install it on this machine (127.0.0.1:{DEFS[app].port})"
    )
    where = run.guide.prompt.choose(
        f"{label}: where does it run?",
        {"here": here, "remote": "on another machine: print its connect card"},
        "here",
    )
    if where == "remote":
        for line in run.apps.remote([app], settings)[app]:
            run.say(line)
        return f"{label}: connect card shown"
    if app is App.GOOSE:
        for line in run.apps.connect_goose(settings):
            run.say(line)
        return "Goose: provider file written"
    if app in USES_PROJECTS:
        _projects_folder(run)
    run.say(f"Installing {label} (its pinned build; a pull can take minutes)…")
    said = run.apps.install(app, run.guide.settings, run.progress)
    run.say(said, style="green")
    if app is App.LIBRECHAT:
        _first_account(run)
    if app is App.OPEN_WEBUI:  # no script makes its accounts: the first sign-up is the admin
        run.say(first_account(), style="bold")
        run.guide.summary.todo.append(f"Open WebUI: {first_account()}")
    return f"{label} installed"


def _projects_folder(run: _Run) -> None:
    """MNM_PROJECTS_DIR, asked once: where OpenCode works and what OpenHands mounts."""
    if run.projects_asked:
        return
    run.projects_asked = True
    guide = run.guide
    current = guide.settings.expanded_projects_dir()
    offered_dir = current  # a refused one (`~`) is not offered again
    if projects_problem(current, guide.settings) is not None:
        offered_dir = DEFAULT_PROJECTS.expanduser()
    chosen = offered_dir
    for _ in range(ASKS):
        answer = guide.prompt.text(
            "Projects folder (your repositories: OpenCode and OpenHands work there)",
            str(offered_dir),
        )
        problem, path = folder_problem(answer)
        if problem is None and path is not None:
            problem = projects_problem(path, guide.settings)  # never ~ or above it
        if problem is None and path is not None:
            chosen = path
            break
        run.say(problem or "", style="yellow")
    if (refused := projects_problem(chosen, guide.settings)) is not None:  # apps-setup 13
        run.say(f"No projects folder is made: {refused}", style="yellow")
        return
    chosen.mkdir(parents=True, exist_ok=True)
    if chosen == current:
        return
    guide.settings = guide.settings.model_copy(update={"projects_dir": chosen})
    guide.save("MNM_PROJECTS_DIR", str(chosen))
    if guide.services_installed:  # their units carry the folder: written again with it
        for line in guide.install_services(guide.settings):
            guide.view.command(line)


def _first_account(run: _Run) -> None:
    """LibreChat's registration is off: its first account, made by its own script."""
    later = "LibreChat's first account: `mnm apps user add librechat EMAIL`."
    email = run.guide.prompt.text("LibreChat's first account: email (Enter: later)", "")
    if not email.strip():
        run.guide.summary.todo.append(later)
        return
    password = ask_password(run.guide.prompt, lambda line: run.say(line, style="yellow"))
    if password is None:
        run.guide.summary.todo.append(later)
        return
    try:
        run.say(run.apps.add_user(email.strip(), password), style="green")
    except InstanceError as exc:
        run.say(str(exc), style="yellow")
        run.guide.summary.todo.append(later)


def _odysseus(run: _Run) -> str:
    """Its `.env` lines as a diff, written and its compose restarted only on yes; the forms
    its Settings take (endpoints, the MCP Tool Server) shown to enter by hand."""
    settings = run.guide.settings
    link = run.apps.link_odysseus(settings)
    for line in link.lines:
        run.say(line)
    for label, text in link.blocks:
        run.say(label, style="bold")
        run.say(text.rstrip(), style="dim")
    if link.seen is None:
        run.say(link.problem or "", style="yellow")
        return "Odysseus: card shown (nothing to write)"
    if not link.seen.diff:
        run.say(f"{link.seen.path} already says this: nothing to write.")
        return "Odysseus: already connected"
    run.say(f"{link.seen.path} would change:", style="bold")
    run.say(link.seen.diff.rstrip(), style="dim")
    if not run.guide.prompt.yes("Write it and restart Odysseus?", False):
        return "Odysseus: nothing written (`mnm apps connect odysseus`)"
    for line in run.apps.apply_odysseus(settings, link):
        run.say(line)
    return "Odysseus connected"
