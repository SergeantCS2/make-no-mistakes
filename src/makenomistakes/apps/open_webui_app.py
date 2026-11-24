"""Open WebUI installed by mnm: its compose project (`compose.open_webui`), its settings from
the router's manifest (`apps/open_webui.py`) in the compose file's environment, and its
`.env` (0600) with what must not sit in a file anyone may read: `WEBUI_SECRET_KEY`
(generated once and kept: it signs the sessions and encrypts what Open WebUI stores, its MCP
OAuth tokens and API keys among them; a new one signs everyone out) and every setting that
carries the router's key.

Its accounts differ from LibreChat's: no script makes them. The first
account to sign up on its page becomes the admin, sign-up is off (`ENABLE_SIGNUP=false`) for
everyone after it, and the admin adds others in its Admin Panel. So `mnm apps user add
open-webui` is refused with that sentence, and the install says "create your account now"
while the page still offers it (`/api/config`'s `onboarding`: no user yet, asked inside its
container, `open_webui_accounts.py`)."""

from __future__ import annotations

from pathlib import Path

from dotenv import dotenv_values

from makenomistakes.apps import compose
from makenomistakes.apps.card import AppFile
from makenomistakes.apps.compose_app import ComposeSpec
from makenomistakes.apps.context import AppContext, generated
from makenomistakes.apps.defs import DEFS, ImagesPin
from makenomistakes.apps.exposure import bind_host
from makenomistakes.apps.names import App
from makenomistakes.apps.open_webui import KEYED, environment, sizes
from makenomistakes.apps.open_webui_accounts import URL, first_account, has_account

PORT = DEFS[App.OPEN_WEBUI].port or 3000
SECRET = "WEBUI_SECRET_KEY"
# In the `.env`: the key it signs with, and whatever carries the router's key.
PRIVATE = frozenset({SECRET, "TOOL_SERVER_CONNECTIONS", *KEYED})


def _settings(ctx: AppContext) -> dict[str, str]:
    manifest = ctx.manifest()
    return environment(manifest, ctx.settings.api_key_value(), sizes(ctx.settings), URL)


def env_value(ctx: AppContext, folder: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """The `.env` mnm writes and the secret values in it."""
    try:
        before = {k: v for k, v in dotenv_values(folder / compose.ENV_FILE).items() if v}
    except OSError:
        before = {}
    secret = before.get(SECRET) or generated(32)
    private = {k: v for k, v in _settings(ctx).items() if k in PRIVATE}
    hidden = [(SECRET, secret)]
    if (key := ctx.settings.api_key_value()) is not None:
        hidden.append(("MNM_API_KEY", key))
    return {SECRET: secret, **private}, hidden


def files(
    ctx: AppContext, folder: Path
) -> tuple[list[tuple[AppFile, Path]], list[tuple[str, str]]]:
    env, secrets = env_value(ctx, folder)
    file = AppFile(f"open-webui/{compose.ENV_FILE}", None, "env", env)
    return [(file, folder / compose.ENV_FILE)], secrets


def compose_for(ctx: AppContext, pin: ImagesPin, folder: Path) -> compose.Compose:
    public = {k: v for k, v in _settings(ctx).items() if k not in PRIVATE}
    host = bind_host(ctx, App.OPEN_WEBUI)
    return compose.open_webui(pin, folder, PORT, compose.owner(), public, host)


def after_start(ctx: AppContext, _folder: Path) -> None:
    if has_account(ctx) is False:
        ctx.say(first_account())


def notes(ctx: AppContext) -> list[str]:
    return [
        "accounts: the first to sign up on its page is the admin, sign-up is off after it; "
        "the admin adds others in its Admin Panel (`mnm apps user add` makes none here)",
        "its settings come from mnm at every start (ENABLE_PERSISTENT_CONFIG=false): a change "
        "in its Admin Settings lasts until the next restart; `mnm apps install open-webui` "
        "again after a desk change",
    ]


SPEC = ComposeSpec(
    App.OPEN_WEBUI,
    ("data",),
    files,
    compose_for,
    after_start=after_start,
    notes=notes,
    data_note="its accounts, chats, documents and uploads",
)
