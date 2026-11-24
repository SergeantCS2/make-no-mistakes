"""What `mnm apps` installed: `<state>/apps/<app>.json`, the pin in use and the one
before it (for `rollback`), schema-versioned and changed under the apps folder's lock, like
the engine and ComfyUI ledgers.

A file this mnm cannot read, or one a newer mnm wrote, is an AppsError naming it, never "not
installed": reading it as absent would install over a running app and lose the pin to roll
back to.
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from makenomistakes.apps.defs import Pin
from makenomistakes.apps.names import App
from makenomistakes.instance_types import InstanceError
from makenomistakes.locking import shared_lock

SCHEMA_VERSION = 1


class AppsError(InstanceError):
    """An app mnm will not install, start or change, worded for the user."""


class AppLedger(BaseModel):
    schema_version: int = SCHEMA_VERSION
    app: App
    pin: Pin
    previous: Pin | None = None
    # `mnm apps expose` put it on every interface (its own login and the router's key
    # on); default off, so a ledger an older mnm rewrites binds 127.0.0.1 again.
    exposed: bool = False
    installed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    changed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class _Schema(BaseModel):
    schema_version: int = 1


def apps_state(state_dir: Path) -> Path:
    return state_dir / "apps"


def locked(state_dir: Path) -> AbstractContextManager[None]:
    """The one lock every app change takes (`files.Ledger` takes the same one)."""
    return shared_lock(apps_state(state_dir))()


def ledger_path(state_dir: Path, app: App) -> Path:
    return apps_state(state_dir) / f"{app.value}.json"


def read(state_dir: Path, app: App) -> AppLedger | None:
    """The app's ledger; None when it is not installed. Unreadable or newer: AppsError."""
    path = ledger_path(state_dir, app)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise AppsError(f"{path} cannot be read: {exc}") from exc
    try:
        version = _Schema.model_validate_json(text).schema_version
    except ValidationError:
        version = SCHEMA_VERSION
    if version > SCHEMA_VERSION:
        raise AppsError(f"{path} is from a newer mnm: run `mnm update`.")
    try:
        return AppLedger.model_validate_json(text)
    except ValidationError as exc:
        raise AppsError(
            f"{path} cannot be read ({exc.errors()[0]['msg']}): move it aside or delete it, "
            f"then `mnm apps install {app.value}` installs the app again and writes it anew "
            "(every `mnm apps` verb reads it first)."
        ) from None


def save(state_dir: Path, ledger: AppLedger) -> None:
    """Written beside and moved into place, under the caller's `locked`."""
    path = ledger_path(state_dir, ledger.app)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(ledger.model_dump_json(indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def forget(state_dir: Path, app: App) -> None:
    ledger_path(state_dir, app).unlink(missing_ok=True)
