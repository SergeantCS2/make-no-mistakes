"""Which folder may be the projects folder (`MNM_PROJECTS_DIR`). OpenHands mounts it
read-write at `/projects` as the user's uid, and OpenCode's server runs in it: `~` would hand
an agent the secrets file, `~/.ssh`, `~/.bashrc` and the systemd user units. Refused: the home
folder itself and any folder above it, `/mnt/c/Users` and above it (a Windows profile under
WSL), and any folder holding mnm's state, secrets, models or apps folder or this checkout.
Checked where the folder is used: the setup's question, `mnm apps install opencode|openhands`
(and every start that writes their unit or compose file) and the web's Get started panel;
never at every `mnm` command."""

from __future__ import annotations

import os
from pathlib import Path

from makenomistakes import config
from makenomistakes.apps.ledger import AppsError
from makenomistakes.config import Settings

WINDOWS_USERS = Path("/mnt/c/Users")
DEFAULT_PROJECTS = Path("~/src")  # `Settings.projects_dir`'s default
HOME_WORDS = (
    "would give OpenHands your whole home folder, your secrets file and SSH keys included; "
    "pick a folder for projects, such as ~/src."
)


def _real(path: Path) -> Path:
    return Path(os.path.realpath(path.expanduser()))


def _within(inner: Path, outer: Path) -> bool:
    """`inner` is `outer` or below it."""
    return inner == outer or outer in inner.parents


def _checkout() -> Path | None:
    from makenomistakes.service import ServiceError, repo_root  # noqa: PLC0415 (a leaf module)

    try:
        return repo_root()
    except ServiceError:
        return None


def projects_problem(path: Path, settings: Settings, home: Path | None = None) -> str | None:
    """Why `path` may not be the projects folder, or None."""
    real = _real(path)
    shown = str(path)
    home_real = _real(home or Path.home())
    if _within(home_real, real):
        where = "`~`" if real == home_real else f"{shown} (above your home folder)"
        return f"{where} {HOME_WORDS}"
    if _within(_real(WINDOWS_USERS), real):
        return (
            f"{shown} would give OpenHands every Windows profile (/mnt/c/Users), yours "
            "included; pick a folder for projects, such as ~/src."
        )
    kept = {
        "mnm's state folder": settings.expanded_state_dir(),
        "the secrets file's folder": config.SECRETS_FILE.parent,
        "the models folder": settings.expanded_models_dir(),
        "the apps folder": settings.expanded_apps_dir(),
    }
    if (checkout := _checkout()) is not None:
        kept["mnm's checkout"] = checkout
    for name, folder in kept.items():
        if _within(_real(folder), real):
            return (
                f"{shown} holds {name} ({folder}), which OpenHands and OpenCode would then "
                "reach; pick a folder for projects, such as ~/src."
            )
    return None


def checked_projects_dir(settings: Settings) -> Path:
    """The projects folder, or AppsError with the reason it is refused."""
    path = settings.expanded_projects_dir()
    if (problem := projects_problem(path, settings)) is not None:
        raise AppsError(f"MNM_PROJECTS_DIR: {problem} Set MNM_PROJECTS_DIR in the secrets file.")
    return path
