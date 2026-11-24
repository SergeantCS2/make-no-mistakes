"""`mnm apps expose <app> [--off]`: an installed app on every interface
instead of 127.0.0.1, only behind its own lock and the router's key.

The rule: an app is opened to the LAN only when its
own login is on, as the router's key is required off loopback, and the router's key is set
(the app reaches every model through the router, and a LAN client of the app then reaches the
router's models through it):

- OpenCode: its server password (`MNM_OPENCODE_PASSWORD`, HTTP basic auth on the server and
  its web UI); `--hostname 0.0.0.0` in its unit;
- LibreChat: registration off in its `.env` and at least one account (counted in its own
  Mongo); `0.0.0.0:3080:3080`;
- Open WebUI: its sign-in (`WEBUI_AUTH=true`, sign-up off, both in the compose file mnm
  writes) and its first account made (its `/api/config`, `open_webui_accounts.py`);
  `0.0.0.0:3000:8080`;
- OpenHands: its session key (`MNM_OPENHANDS_KEY`, `OH_SESSION_API_KEYS_0`), with
  `AGENT_CANVAS_ALLOW_LAN_SESSION_KEY` left off, so a LAN browser is asked for it;
  `0.0.0.0:8000:8000`.

The choice is the ledger's `exposed` (default off: an older mnm that rewrites the ledger
drops it and binds 127.0.0.1 again, the safe way round). The drivers read `bind_host` each
time they write the unit or the compose file; `mnm apps uninstall` removes both, and the
ledger with them. ufw is never changed: on a native box with ufw on, the command is said.

Every exposed app speaks plain http, so its login
crosses the LAN in clear text with every request (OpenCode's basic auth is a shell on this
machine for whoever reads it); `clear_text` says so, `mnm apps expose` asks before opening
(`--yes` skips) and the doctor's *Exposed* line repeats it. Docker publishes LibreChat's,
OpenHands' and Open WebUI's ports past ufw's INPUT chain, so for them ufw is no lock at all:
only OpenCode, a native process, gets the ufw line."""

from __future__ import annotations

from collections.abc import Callable

from dotenv import dotenv_values

from makenomistakes.apps import ledger as ledgers
from makenomistakes.apps.compose import ENV_FILE
from makenomistakes.apps.context import AppContext
from makenomistakes.apps.defs import DEFS
from makenomistakes.apps.ledger import AppsError
from makenomistakes.apps.names import LABELS, App
from makenomistakes.apps.open_webui_accounts import has_account
from makenomistakes.client_view import KEY_STEP
from makenomistakes.hardware import platform as platforms
from makenomistakes.hardware.probe import UFW_CONF, parse_ufw_enabled, read_text_or_empty
from makenomistakes.router_bind import EVERYWHERE, LOOPBACK

EXPOSABLE = (App.OPENCODE, App.LIBRECHAT, App.OPENHANDS, App.OPEN_WEBUI)
IN_DOCKER = frozenset({App.LIBRECHAT, App.OPENHANDS, App.OPEN_WEBUI})  # Docker's, past ufw
_LOGINS = {
    App.OPENCODE: "OpenCode's password (HTTP basic auth: whoever reads it runs commands on "
    "this machine as you)",
    App.LIBRECHAT: "LibreChat's sign-in and its session cookie",
    App.OPEN_WEBUI: "Open WebUI's sign-in and its session token",
    App.OPENHANDS: "OpenHands' session key (whoever reads it drives the agent in your "
    "projects folder)",
}
TAILSCALE = "Tailscale Serve (https, GUIDE 13.4) is the safe way to reach it from elsewhere."
MONGO = "mnm-librechat-mongodb-1"  # compose project mnm-librechat, service mongodb
COUNT_USERS = ["mongosh", "--quiet", "LibreChat", "--eval", "db.users.countDocuments()"]


def bind_host(ctx: AppContext, app: App) -> str:
    """Where the app's port is published: every interface once exposed, else 127.0.0.1. An
    unreadable ledger binds loopback (the drivers' caller refuses it anyway)."""
    try:
        found = ledgers.read(ctx.state_dir, app)
    except AppsError:
        return LOOPBACK
    return EVERYWHERE if found is not None and found.exposed else LOOPBACK


def refusal(ctx: AppContext, app: App) -> str | None:
    """Why the app may not be opened to the LAN now; None when it may."""
    label = LABELS[app]
    if app not in EXPOSABLE:
        return f"{label} is not an app mnm runs: nothing of it to open."
    if ctx.settings.api_key_value() is None:
        return (
            f"{label} reaches every model through the router, so the router's key comes "
            f"first: none is set. {KEY_STEP}"
        )
    return _LOCKS[app](ctx)


def _opencode(ctx: AppContext) -> str | None:
    if not ctx.settings.app_secret("MNM_OPENCODE_PASSWORD"):
        return (
            "OpenCode has no server password (MNM_OPENCODE_PASSWORD): `mnm apps "
            "install opencode` generates it."
        )
    return None


def _openhands(ctx: AppContext) -> str | None:
    if not ctx.settings.app_secret("MNM_OPENHANDS_KEY"):
        return (
            "OpenHands has no session key (MNM_OPENHANDS_KEY): `mnm apps install "
            "openhands` generates it."
        )
    return None


def _open_webui(ctx: AppContext) -> str | None:
    """Its login on and sign-up off are mnm's own lines (`open_webui.FIXED`), which the
    compose file carries at every start; an account must exist to sign in with."""
    found = has_account(ctx)
    if isinstance(found, str):
        return f"{found}: `mnm apps start open-webui` first."
    if not found:
        return (
            "Open WebUI has no account yet, so the first stranger to open it would become its "
            "admin: create yours at http://localhost:3000 first."
        )
    return None


def _librechat(ctx: AppContext) -> str | None:
    env = ctx.folder(App.LIBRECHAT) / ENV_FILE
    try:
        values = dotenv_values(env)
    except OSError as exc:
        return f"LibreChat's {env} cannot be read ({exc}): `mnm apps install librechat`."
    opened = [
        name
        for name in ("ALLOW_REGISTRATION", "ALLOW_SOCIAL_LOGIN", "ALLOW_SOCIAL_REGISTRATION")
        if (values.get(name) or "false").strip().lower() != "false"
    ]
    if opened:
        return (
            f"LibreChat's {env} lets strangers in ({', '.join(opened)} not false): mnm writes "
            "them false; `mnm apps install librechat` writes them again."
        )
    code, out, err = ctx.runner.run(["docker", "exec", MONGO, *COUNT_USERS], timeout=60.0)
    if code != 0:
        said = (err or out).strip().splitlines()[-1:] or [f"exit {code}"]
        return (
            f"LibreChat's accounts could not be counted ({said[0]}): `mnm apps start "
            "librechat` first."
        )
    count = out.strip().splitlines()[-1:] or [""]
    if not count[0].isdigit() or int(count[0]) == 0:
        return (
            "LibreChat has no account yet, so nobody could sign in: `mnm apps user add "
            "librechat EMAIL` first."
        )
    return None


# Each app's own lock, checked before it is opened: why not, or None.
_LOCKS: dict[App, Callable[[AppContext], str | None]] = {
    App.OPENCODE: _opencode,
    App.OPENHANDS: _openhands,
    App.LIBRECHAT: _librechat,
    App.OPEN_WEBUI: _open_webui,
}


def clear_text(app: App) -> str:
    """The sentence `mnm apps expose` asks with and the doctor repeats."""
    login = _LOGINS.get(app, f"{LABELS[app]}'s login")
    return (
        f"{LABELS[app]} speaks plain http: {login} crosses your network in clear text with "
        f"every request, readable by anyone who can watch it. {TAILSCALE}"
    )


def ufw_enabled() -> bool:
    return parse_ufw_enabled(read_text_or_empty(UFW_CONF))


def notes(app: App, opened: bool, ufw_on: Callable[[], bool] = ufw_enabled) -> list[str]:
    """What the user reads after the switch: where it answers, and the firewall."""
    port = DEFS[app].port
    label = LABELS[app]
    lines = (
        [
            f"{label} is set to listen on every interface, port {port} (from its next "
            "start: now, unless a line above says it was not started): another machine opens "
            f"http://<this machine's address>:{port}{DEFS[app].ui_path or '/'} and signs in "
            "with the app's own login."
        ]
        if opened
        else [f"{label} answers on 127.0.0.1:{port} only again."]
    )
    if opened:
        lines.append(clear_text(app))
    if opened and app is App.LIBRECHAT:
        lines.append(
            "LibreChat's sign-in cookie is marked secure: over plain http from another machine "
            "a browser may not keep it; Tailscale Serve (https, GUIDE 13.4) does."
        )
    where = platforms.current()
    if where.is_wsl and opened:
        lines.append(
            "Under WSL's NAT networking only Windows reaches this VM: another machine needs "
            "mirrored networking or Tailscale (GUIDE 13.3, 13.4)."
        )
    if opened and app in IN_DOCKER:
        lines.append(
            f"Docker publishes past ufw: the LAN reaches port {port} whatever ufw allows; "
            f"`mnm apps expose {app.value} --off` closes it."
        )
    elif not where.is_wsl and app not in IN_DOCKER and ufw_on():
        lines.append(
            f"ufw is on: `sudo ufw allow {port}/tcp` lets the LAN in (mnm leaves ufw alone)"
            if opened
            else f"ufw: `sudo ufw delete allow {port}/tcp` if you added that rule"
        )
    return lines
