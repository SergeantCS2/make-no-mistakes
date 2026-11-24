"""What the guided setup's Workspace apps section and the web's Get started do with an app,
over the same code `mnm apps` runs: `AppsService.install` (the pinned build, its
config from the router's manifest, started and answering), `connect.write_card_file` (Goose's
provider file), `connect.preview`/`apply_odysseus` (Odysseus's `.env` diff, written and its
compose restarted only on yes) and `users.add_librechat_user` (the first account, its password
on the script's stdin, never kept). `AppsSetup` can be swapped; `LiveApps` is the
real one, built for the settings the guide has gathered so far."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from makenomistakes import report
from makenomistakes.apps.cards import card_for, local_manifest
from makenomistakes.apps.connect import (
    Preview,
    apply_odysseus,
    odysseus_folder,
    preview,
    write_card_file,
    written_line,
)
from makenomistakes.apps.context import AppContext
from makenomistakes.apps.defs import DEFS
from makenomistakes.apps.files import AppFileError
from makenomistakes.apps.hub import AppsService
from makenomistakes.apps.ledger import AppsError
from makenomistakes.apps.names import LABELS, App
from makenomistakes.apps.offer import remote_lines
from makenomistakes.apps.users import SubprocessInputRunner, add_librechat_user
from makenomistakes.config import Settings
from makenomistakes.hardware import SubprocessRunner
from makenomistakes.instances import InstanceService

Say = Callable[[str], None]


@dataclass(frozen=True)
class OdysseusLink:
    """What connecting Odysseus would do: its card's lines and forms, and the `.env` change
    (None with `problem`: nothing to write, and why)."""

    folder: Path
    lines: list[str]
    blocks: list[tuple[str, str]] = field(default_factory=list)
    seen: Preview | None = None
    problem: str | None = None


class AppsSetup(Protocol):
    def odysseus(self) -> Path | None:
        """Odysseus's folder when it exists (only then is it offered)."""
        ...

    def install(self, app: App, settings: Settings, say: Say) -> str: ...
    def connect_goose(self, settings: Settings) -> list[str]: ...
    def link_odysseus(self, settings: Settings) -> OdysseusLink: ...
    def apply_odysseus(self, settings: Settings, link: OdysseusLink) -> list[str]: ...
    def add_user(self, email: str, password: str) -> str: ...
    def remote(self, apps: list[App], settings: Settings) -> dict[App, list[str]]: ...


class LiveApps:
    """Docker, systemd and the files for real."""

    def __init__(self, folder: Path | None = None) -> None:
        self.folder = odysseus_folder(folder)

    def odysseus(self) -> Path | None:
        return self.folder if self.folder.is_dir() else None

    def install(self, app: App, settings: Settings, say: Say) -> str:
        """`mnm apps install <app>`: AppsError (or InstanceError) says why it did not."""
        service = InstanceService(settings)
        ctx = AppContext(settings, SubprocessRunner(), lambda: local_manifest(service), say=say)
        done = AppsService(ctx).install(app)
        return f"{LABELS[app]} {done.pin.version} is installed: {DEFS[app].url}"

    def connect_goose(self, settings: Settings) -> list[str]:
        card = card_for(App.GOOSE, local_manifest(InstanceService(settings)), settings)
        try:
            done = write_card_file(card, settings)
        except AppFileError as exc:
            raise AppsError(str(exc)) from exc
        return [*card.lines, written_line(done)]

    def link_odysseus(self, settings: Settings) -> OdysseusLink:
        card = card_for(App.ODYSSEUS, local_manifest(InstanceService(settings)), settings)
        try:
            seen = preview(card, self.folder, report.secrets(settings).known)
        except AppsError as exc:
            return OdysseusLink(self.folder, card.lines, card.blocks, problem=str(exc))
        return OdysseusLink(self.folder, card.lines, card.blocks, seen)

    def apply_odysseus(self, settings: Settings, link: OdysseusLink) -> list[str]:
        if link.seen is None:
            raise AppsError(link.problem or "Odysseus's .env gets nothing from mnm now.")
        return apply_odysseus(link.seen, link.folder, settings, SubprocessRunner())

    def add_user(self, email: str, password: str) -> str:
        return add_librechat_user(email, None, None, password, SubprocessInputRunner())

    def remote(self, apps: list[App], settings: Settings) -> dict[App, list[str]]:
        """Each app's card for another machine, from one manifest."""
        manifest = local_manifest(InstanceService(settings))
        return {app: remote_lines(app, manifest) for app in apps}
