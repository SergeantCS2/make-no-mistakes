"""The apps folder (`MNM_APPS_DIR`, /data/apps by default) made ready before an install
(else LibreChat's and OpenHands' installs die on `PermissionError: /data/apps`, a folder
nothing else makes). Created when its parent is writable; otherwise, or when it exists and is
not writable, refused with the commands that make it. The CLI, the
web's install job and the setup's section 8 all reach it through `AppsService.install`."""

from __future__ import annotations

import getpass
import os
import shlex
from pathlib import Path

from makenomistakes.apps.ledger import AppsError
from makenomistakes.apps.names import App

# The apps whose data and config live under the apps folder (OpenCode's binary lives in the
# state folder; Goose, Odysseus and Hermes are only connected).
IN_APPS_FOLDER = frozenset({App.LIBRECHAT, App.OPENHANDS, App.OPEN_WEBUI})


def ready(folder: Path, user: str | None = None) -> None:
    """`folder` exists and this user writes to it, else AppsError with the fix."""
    if folder.is_dir():
        if os.access(folder, os.W_OK | os.X_OK):
            return
        raise AppsError(f"{folder} exists but is not writable by you. {_fix(folder, user)}")
    if folder.exists():
        raise AppsError(
            f"{folder} is a file, not a folder: move it, or set MNM_APPS_DIR to a folder of "
            "yours in the secrets file."
        )
    parent = _existing_parent(folder)
    if os.access(parent, os.W_OK | os.X_OK):
        try:
            folder.mkdir(parents=True, exist_ok=True)
            return
        except OSError as exc:
            raise AppsError(f"{folder} could not be made ({exc}). {_fix(folder, user)}") from exc
    raise AppsError(
        f"The apps folder {folder} does not exist and {parent} is not writable by you, so mnm "
        f"cannot make it. {_fix(folder, user)}"
    )


def _fix(folder: Path, user: str | None) -> str:
    who = user or getpass.getuser()
    where = shlex.quote(str(folder))
    return (
        f"Run `sudo mkdir -p {where} && sudo chown {who}:{who} {where}` (or set MNM_APPS_DIR "
        "to a folder of yours in the secrets file), then install again."
    )


def _existing_parent(folder: Path) -> Path:
    probe = folder.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe
