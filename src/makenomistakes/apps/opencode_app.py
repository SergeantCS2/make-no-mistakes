"""OpenCode installed by mnm: native, the one exception to "everything in
Docker" (a workspace that acts on the user's repositories with the user's toolchain, not an
engine). The pinned release binary (`binary.py`), its `opencode.json` by its renderer
(`apps/opencode.py`, written by `files.write` to OpenCode's global config), the projects
folder, and the user unit `mnm-opencode` (`unit.py`) running `opencode serve` on
127.0.0.1:4096, its web UI on the same port behind HTTP basic auth (user `opencode`, the
password mnm generated into the secrets file as `MNM_OPENCODE_PASSWORD`: `/`,
`/doc` and `/global/health` answer 401 without it).

Without systemd's user manager (a WSL distro without `[boot] systemd=true`) the files are all
written and the command to run by hand is said; nothing is started."""

from __future__ import annotations

import shlex

from makenomistakes import config
from makenomistakes.apps import binary, unit
from makenomistakes.apps.cards import card_for
from makenomistakes.apps.context import AppContext, generated
from makenomistakes.apps.defs import DEFS, BinaryPin, Pin
from makenomistakes.apps.exposure import bind_host
from makenomistakes.apps.ledger import AppsError, apps_state
from makenomistakes.apps.names import App
from makenomistakes.apps.projects import checked_projects_dir
from makenomistakes.consumers import origin
from makenomistakes.router_bind import LOOPBACK
from makenomistakes.setup import write_setting

PASSWORD = "MNM_OPENCODE_PASSWORD"
USERNAME = "opencode"
DEF = DEFS[App.OPENCODE]
PORT = DEF.port or 4096


class OpenCodeApp:
    app = App.OPENCODE

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self.state = apps_state(ctx.state_dir)
        self.env_file = self.state / "opencode" / unit.ENV_NAME

    # --- the steps -------------------------------------------------------------------

    def install(self, pin: Pin) -> None:
        found = _binary_pin(pin)
        checked_projects_dir(self.ctx.settings)  # refused before anything downloads
        self.ctx.say(f"OpenCode {found.version}: the release binary, checked by its sha256…")
        exe = binary.fetch(found, self.state, self.ctx.download)
        self.ctx.say(f"{exe} (OpenCode {found.version})")
        self.write_config()
        self.start(found)

    def write_config(self) -> None:
        """`opencode.json` from the manifest: the desk's roles, every chat model, the MCP."""
        card = card_for(App.OPENCODE, self.ctx.manifest(), self.ctx.settings)
        file = card.files[0]
        if file.default is None:  # OpenCode's own file always has a place
            raise AppsError("OpenCode's config file has no place of its own.")
        self.ctx.write(App.OPENCODE, file, file.default, [])

    def switch(self, pin: Pin) -> None:
        found = _binary_pin(pin)
        binary.fetch(found, self.state, self.ctx.download)
        self.start(found)

    def start(self, pin: Pin) -> None:
        """The unit written again first: its address follows `mnm apps expose`."""
        self._unit(_binary_pin(pin))
        self._env()
        if not unit.systemd_answers(self.ctx.runner):
            self.ctx.say(self._by_hand(_binary_pin(pin)))
            return
        for args in (("daemon-reload",), ("enable", unit.UNIT), ("restart", unit.UNIT)):
            code, said = unit.systemctl(self.ctx.runner, *args)
            if code != 0:
                raise AppsError(f"systemctl --user {' '.join(args)} failed: {said}")
        self.ctx.wait_ready(DEF.health_url or "", "OpenCode", (USERNAME, self.password()))

    def stop(self) -> None:
        if unit.systemd_answers(self.ctx.runner):
            unit.systemctl(self.ctx.runner, "stop", unit.UNIT)

    def running(self) -> bool | None:
        return unit.active(self.ctx.runner)

    def logs(self, tail: int) -> str:
        argv = ["journalctl", "--user", "-u", unit.UNIT, "-n", str(tail), "--no-pager"]
        code, out, err = self.ctx.runner.run(argv, timeout=30.0)
        return out if code == 0 else err or out

    def remove(self, keep_data: bool) -> list[str]:
        done: list[str] = []
        path = unit.unit_path(self.ctx.unit_dir)
        if path.exists():
            if unit.systemd_answers(self.ctx.runner):
                unit.systemctl(self.ctx.runner, "disable", "--now", unit.UNIT)
            path.unlink()
            if unit.systemd_answers(self.ctx.runner):
                unit.systemctl(self.ctx.runner, "daemon-reload")
            done.append(f"removed the unit {path}")
        folder = self.state / "opencode"
        if folder.exists():
            for child in folder.iterdir():
                if child.is_dir():
                    binary.remove_version(self.state, child.name)
            self.env_file.unlink(missing_ok=True)
            done.append(f"removed OpenCode's binaries under {folder}")
        done.append(
            "kept: OpenCode's own sessions and settings (~/.local/share/opencode, "
            "~/.config/opencode) and the projects folder: they are yours"
        )
        del keep_data  # OpenCode keeps no data under MNM_APPS_DIR
        return done

    def prune(self, dropped: Pin) -> list[str]:
        folder = binary.folder(self.state, dropped.version)
        if not folder.exists():
            return []
        binary.remove_version(self.state, dropped.version)
        return [str(folder)]

    def notes(self) -> list[str]:
        return [
            f"user `{USERNAME}`, the password is {PASSWORD} in the secrets file "
            f"({config.SECRETS_FILE}); never printed",
            "a browser opens the URL above (OpenCode's web UI); `opencode attach "
            f"{origin(LOOPBACK, PORT)}` joins from a terminal",
        ]

    # --- the pieces ------------------------------------------------------------------

    def password(self) -> str:
        found = self.ctx.settings.app_secret(PASSWORD)
        if found:
            return found
        made = generated(24)
        write_setting(config.SECRETS_FILE, PASSWORD, made)
        self.ctx.say(f"generated OpenCode's password into the secrets file as {PASSWORD}")
        return made

    def _env(self) -> None:
        text = unit.env_text(self.password(), self.ctx.settings.api_key_value())
        unit.write_private(self.env_file, text)

    def _unit(self, pin: BinaryPin) -> None:
        projects = checked_projects_dir(self.ctx.settings)
        projects.mkdir(parents=True, exist_ok=True)
        exe = binary.executable(self.state, pin.version)
        path = unit.unit_path(self.ctx.unit_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        host = bind_host(self.ctx, App.OPENCODE)
        text = unit.unit_text(exe, PORT, projects, self.env_file, host)
        path.write_text(text, encoding="utf-8")
        self.ctx.say(f"wrote the unit {path} (in {projects})")

    def _by_hand(self, pin: BinaryPin) -> str:
        exe = shlex.quote(str(binary.executable(self.state, pin.version)))
        return (
            "systemd's user manager does not answer, so OpenCode was not started: enable it "
            "(`mnm service install` says how), then `mnm apps start opencode`; or by hand: "
            f"`set -a; . {shlex.quote(str(self.env_file))}; set +a; cd "
            f"{shlex.quote(str(self.ctx.settings.expanded_projects_dir()))} && {exe} serve "
            f"--port {PORT} --hostname {bind_host(self.ctx, App.OPENCODE)}`"
        )


def _binary_pin(pin: Pin) -> BinaryPin:
    if not isinstance(pin, BinaryPin):
        raise AppsError("OpenCode's pin is a release binary; this one is not.")
    return pin
