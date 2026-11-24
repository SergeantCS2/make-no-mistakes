"""A Docker app mnm runs as its own compose project: LibreChat and OpenHands. The
compose file is rendered from the pin (`compose.py`) and written with the app's config files
by `files.write` into `MNM_APPS_DIR/<app>/`; `docker compose -p <project> -f compose.yaml
--env-file .env` pulls (told line by line), starts, stops and reads the logs; the app's page
is waited for before it is called installed. Each app says what it writes and what it does
once it answers (`ComposeSpec`)."""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from makenomistakes.apps import compose
from makenomistakes.apps.card import AppFile
from makenomistakes.apps.context import AppContext
from makenomistakes.apps.defs import DEFS, ImagesPin, Pin
from makenomistakes.apps.ledger import AppsError
from makenomistakes.apps.names import CONTAINERS, LABELS, App

PULL_SECONDS = 3600.0  # LibreChat's five images are ~3 GB
UP_SECONDS = 600.0
LAYER = re.compile(r"^\s*[0-9a-f]{12} ")  # `<layer id> Downloading 5.2MB` and the like


@dataclass(frozen=True)
class ComposeSpec:
    app: App
    folders: tuple[str, ...]  # made under the app's folder before the first start
    # The config files and the secrets they hold (hidden in a diff), from the context.
    files: Callable[[AppContext, Path], tuple[list[tuple[AppFile, Path]], list[tuple[str, str]]]]
    compose: Callable[[AppContext, ImagesPin, Path], compose.Compose]
    after_start: Callable[[AppContext, Path], None] | None = None
    notes: Callable[[AppContext], list[str]] = lambda _ctx: []
    data_note: str = ""  # what `--keep-data` keeps, in words


class ComposeApp:
    def __init__(self, ctx: AppContext, spec: ComposeSpec) -> None:
        self.ctx = ctx
        self.spec = spec
        self.app = spec.app
        self.definition = DEFS[spec.app]
        self.folder = ctx.folder(spec.app)

    @property
    def project(self) -> str:
        return self.definition.project or f"mnm-{self.app.value}"

    def _compose(self, *args: str) -> list[str]:
        return [
            "docker",
            "compose",
            "-p",
            self.project,
            "-f",
            str(self.folder / compose.COMPOSE_FILE),
            "--env-file",
            str(self.folder / compose.ENV_FILE),
            *args,
        ]

    def _stream(self, argv: list[str], timeout: float, what: str) -> None:
        """A compose command's lines said as they come, less a pull's per-layer progress (a
        LibreChat pull prints 2,200 of them)."""
        tail: list[str] = []

        def line(text: str | None) -> None:
            if text is None:
                self.ctx.say(f"still {what}…")
                return
            tail.append(text)
            del tail[:-20]
            if not LAYER.match(text):  # one line per image layer and byte count: kept for errors
                self.ctx.say(text)

        code = self.ctx.runner.stream(argv, line, timeout=timeout)
        if code != 0:
            last = "\n".join(tail[-8:])
            raise AppsError(f"`{' '.join(argv[:2])} … {what}` failed (exit {code}):\n{last}")

    # --- the steps -------------------------------------------------------------------

    def install(self, pin: Pin) -> None:
        images = _images(pin)
        self._docker_answers()
        for name in self.spec.folders:
            (self.folder / name).mkdir(parents=True, exist_ok=True)
        self.write_config()
        self._write_compose(images)
        self._up(fresh=True)

    def write_config(self) -> None:
        files, secrets = self.spec.files(self.ctx, self.folder)
        for file, path in files:
            self.ctx.write(self.app, file, path, secrets)

    def switch(self, pin: Pin) -> None:
        self._docker_answers()
        self._write_compose(_images(pin))
        self._up(fresh=True)

    def start(self, pin: Pin) -> None:
        self._docker_answers()
        self._write_compose(_images(pin))
        self._up(fresh=False)

    def stop(self) -> None:
        code, out, err = self.ctx.runner.run(self._compose("stop"), timeout=180.0)
        if code != 0:
            raise AppsError(f"docker compose stop failed: {(err or out).strip()}")

    def running(self) -> bool | None:
        name = CONTAINERS.get(self.app)
        if name is None:
            return None
        code, out, err = self.ctx.runner.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name]
        )
        if code == 0:
            return out.strip() == "true"
        if "no such object" in (err or out).lower() or "no such container" in (err or out).lower():
            return False
        return None  # Docker could not say

    def logs(self, tail: int) -> str:
        code, out, err = self.ctx.runner.run(
            self._compose("logs", "--no-color", "--tail", str(tail)), timeout=60.0
        )
        return out + err if code == 0 else err or out

    def remove(self, keep_data: bool) -> list[str]:
        done: list[str] = []
        if (self.folder / compose.COMPOSE_FILE).exists():
            # `-v`: the anonymous volumes the images declare (Mongo's configdb, Agent Canvas's
            # ~/.openhands); the data is in bind mounts and named volumes there are none
            code, out, err = self.ctx.runner.run(self._compose("down", "-v"), timeout=300.0)
            if code != 0:
                raise AppsError(f"docker compose down failed: {(err or out).strip()}")
            done.append(f"stopped and removed the {self.project} containers")
        if keep_data:
            done.append(f"kept {self.folder}: {self.spec.data_note}")
        elif self.folder.is_symlink():
            done.append(f"kept {self.folder}: a link, which mnm never deletes through")
        elif self.folder.exists():
            shutil.rmtree(self.folder)
            done.append(f"deleted {self.folder} ({self.spec.data_note})")
        return done

    def prune(self, dropped: Pin) -> list[str]:
        """The images of a pin neither in use nor kept for rollback."""
        gone = []
        for ref in _images(dropped).images.values():
            if self.ctx.runner.run(["docker", "image", "rm", ref], timeout=120.0)[0] == 0:
                gone.append(ref)
        return gone

    def notes(self) -> list[str]:
        return self.spec.notes(self.ctx)

    # --- the pieces ------------------------------------------------------------------

    def _docker_answers(self) -> None:
        code, out, err = self.ctx.runner.run(["docker", "compose", "version"], timeout=30.0)
        if code != 0:
            raise AppsError(
                f"`docker compose` does not answer ({(err or out).strip()}): `mnm doctor` "
                "says what Docker needs; `mnm setup` installs its compose plugin."
            )

    def _write_compose(self, pin: ImagesPin) -> None:
        """mnm's own file, never merged: one the user changed is refused without --force."""
        value = compose.value(self.spec.compose(self.ctx, pin, self.folder))
        file = AppFile(f"{self.app.value}/{compose.COMPOSE_FILE}", None, "yaml", value)
        self.ctx.write(self.app, file, self.folder / compose.COMPOSE_FILE, [])

    def _up(self, fresh: bool) -> None:
        """Pull (a new pin only), start, wait; the app's own step after a new pin."""
        if fresh:
            self.ctx.say(f"pulling the pinned images of {LABELS[self.app]}…")
            self._stream(self._compose("pull"), PULL_SECONDS, "pulling")
        # A new pin or new config files: every container again, since the apps read their
        # mounted `.env` and config at start (an unchanged service would keep the old ones).
        again = ("--force-recreate",) if fresh else ()
        self._stream(self._compose("up", "-d", "--remove-orphans", *again), UP_SECONDS, "starting")
        self.ctx.wait_ready(self.definition.health_url or "", LABELS[self.app])
        if fresh and self.spec.after_start is not None:
            self.spec.after_start(self.ctx, self.folder)


def _images(pin: Pin) -> ImagesPin:
    if not isinstance(pin, ImagesPin):
        raise AppsError("A compose app's pin names images; this one does not.")
    return pin
