"""What every `mnm apps` step is given: the settings, the Docker/systemd runner, where
to tell progress, the manifest its config files come from, and the swappable parts (the
download, the HTTP check, the clock). Plus the two things every app does: write a config file
by `files.write`'s rules (merged into the user's, a `.before-mnm` copy, `written.json`)
and wait for its page to answer."""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from makenomistakes import config, report
from makenomistakes.apps.binary import Download, http_download
from makenomistakes.apps.card import AppFile
from makenomistakes.apps.files import AppFileError, Ledger, WriteResult, write
from makenomistakes.apps.ledger import AppsError
from makenomistakes.apps.names import App
from makenomistakes.config import Settings
from makenomistakes.hardware.probe import LineRunner
from makenomistakes.manifest import Manifest

Say = Callable[[str], None]
# GET a URL with optional basic auth: its status (0 when nothing answered).
Check = Callable[[str, tuple[str, str] | None], int]
# PATCH a JSON body with headers: the status (0: nothing answered) and the body.
Patch = Callable[[str, dict[str, str], dict[str, object]], tuple[int, str]]
READY_SECONDS = 300.0  # LibreChat's first start (Mongo, Meili, its own build) takes a minute


def http_status(url: str, auth: tuple[str, str] | None) -> int:
    try:
        return httpx.get(url, auth=auth, timeout=5.0).status_code
    except httpx.HTTPError:
        return 0


def http_patch(url: str, headers: dict[str, str], body: dict[str, object]) -> tuple[int, str]:
    try:
        answer = httpx.patch(url, headers=headers, json=body, timeout=30.0)
    except httpx.HTTPError as exc:
        return 0, str(exc)
    return answer.status_code, answer.text


def generated(length: int = 32) -> str:
    """A fresh secret: `length` random bytes as hex."""
    return secrets.token_hex(length)


@dataclass
class AppContext:
    settings: Settings
    runner: LineRunner
    manifest: Callable[[], Manifest]
    say: Say = print
    force: bool = False  # replace a config file of the user's mnm cannot merge into
    download: Download = http_download
    check: Check = http_status
    patch: Patch = http_patch
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    unit_dir: Path | None = None
    written: list[WriteResult] = field(default_factory=list)

    @property
    def state_dir(self) -> Path:
        return self.settings.expanded_state_dir()

    @property
    def apps_dir(self) -> Path:
        return self.settings.expanded_apps_dir()

    def folder(self, app: App) -> Path:
        return self.apps_dir / app.value

    def write(self, app: App, file: AppFile, path: Path, extra: list[tuple[str, str]]) -> None:
        """One config file by `files.write`'s rules; said in one line (a merge's diff hides
        every known secret and `extra`, the values mnm just generated)."""
        known = report.secrets(self.settings).known + extra
        try:
            done = write(file, path, app, Ledger(self.state_dir), self.force, known)
        except (AppFileError, OSError) as exc:
            if isinstance(exc, AppFileError):
                raise AppsError(str(exc)) from exc
            raise AppsError(f"{path}: {exc}") from exc
        self.written.append(done)
        self.say(_said(done))

    def redacted(self, text: str, extra: list[tuple[str, str]]) -> str:
        """An app's own answer, safe to print: every secret mnm knows, `extra` (the values
        just sent) and whatever only looks like one, through `report.redact`."""
        known = report.secrets(self.settings, config.SECRETS_FILE).known + extra
        return report.redact(text, [(n, v) for n, v in known if v])

    def wait_ready(self, url: str, label: str, auth: tuple[str, str] | None = None) -> None:
        """Until `url` answers 200, saying so every 15 s; AppsError after READY_SECONDS."""
        started = self.clock()
        told = started
        while True:
            status = self.check(url, auth)
            if status == 200:
                self.say(f"{label} answers at {url}.")
                return
            now = self.clock()
            if now - started > READY_SECONDS:
                raise AppsError(
                    f"{label} did not answer at {url} within {READY_SECONDS:.0f} s (last "
                    f"answer: {status or 'none'}): `mnm apps logs` shows why."
                )
            if now - told >= 15:
                self.say(f"waiting for {label} ({now - started:.0f} s)…")
                told = now
            self.sleep(2.0)


def _said(done: WriteResult) -> str:
    words = {
        "wrote": "wrote",
        "replaced": "rewrote (mnm's own)",
        "merged": f"merged mnm's part into (yours kept as {done.backup})",
        "unchanged": "unchanged:",
        "overwrote": f"replaced (--force; yours kept as {done.backup})",
    }
    return f"{words[done.action]} {done.path}"
