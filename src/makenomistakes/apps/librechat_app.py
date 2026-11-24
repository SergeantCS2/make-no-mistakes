"""LibreChat installed by mnm: its compose project (`compose.librechat`), its
`librechat.yaml` by its renderer (`apps/librechat.py`: mnm as the custom endpoint, speech,
mnm's MCP tools, the `allowedAddresses`), and its `.env` with the secrets it needs generated
once and kept (a later install or update reuses them: a new CREDS_KEY would make every key a
user saved in LibreChat unreadable).

From LibreChat's docs, read 2026-10-22 (librechat.ai/docs/configuration/dotenv: `CREDS_KEY` 64
hex, `CREDS_IV` 32 hex, `JWT_SECRET`/`JWT_REFRESH_SECRET`, `MEILI_MASTER_KEY` of 16 bytes or
more; `ALLOW_REGISTRATION`, `ALLOW_EMAIL_LOGIN`, `ALLOW_SOCIAL_LOGIN`; …/rag_api: the RAG lines
`apps/librechat.rag_lines` gives) and its v0.8.7 `.env.example`. Registration is off: the
first account is `mnm apps user add librechat EMAIL` (`users.py`).

When the router asks for a key, its value goes into this `.env` (mode 0600, beside LibreChat's
own secrets) as `MNM_API_KEY` and `RAG_OPENAI_API_KEY`: `librechat.yaml` reads
`${MNM_API_KEY}` from it, and LibreChat sends it with every model, speech and MCP request."""

from __future__ import annotations

from pathlib import Path

from dotenv import dotenv_values

from makenomistakes.apps import compose
from makenomistakes.apps.card import AppFile
from makenomistakes.apps.cards import card_for
from makenomistakes.apps.compose_app import ComposeSpec
from makenomistakes.apps.context import AppContext, generated
from makenomistakes.apps.defs import DEFS, ImagesPin
from makenomistakes.apps.exposure import bind_host
from makenomistakes.apps.librechat import rag_lines
from makenomistakes.apps.names import App

PORT = DEFS[App.LIBRECHAT].port or 3080
# Each secret and its length in random bytes (shown as twice as many hex digits).
SECRETS = {
    "CREDS_KEY": 32,
    "CREDS_IV": 16,
    "JWT_SECRET": 32,
    "JWT_REFRESH_SECRET": 32,
    "MEILI_MASTER_KEY": 24,
    "POSTGRES_PASSWORD": 24,
}
SETTINGS = {
    "PORT": str(PORT),
    "DOMAIN_CLIENT": f"http://localhost:{PORT}",
    "DOMAIN_SERVER": f"http://localhost:{PORT}",
    "NO_INDEX": "true",
    "ALLOW_EMAIL_LOGIN": "true",
    "ALLOW_REGISTRATION": "false",
    "ALLOW_SOCIAL_LOGIN": "false",
    "ALLOW_SOCIAL_REGISTRATION": "false",
    "ALLOW_PASSWORD_RESET": "false",
    "ALLOW_UNVERIFIED_EMAIL_LOGIN": "true",
    "SEARCH": "true",
    "MEILI_NO_ANALYTICS": "true",
    # mnm's endpoint first, so a new chat opens on it: without it a new account lands on "My
    # Agents" with no agent and every first message fails (v0.8.7).
    "ENDPOINTS": "custom,agents",
}


def env_value(ctx: AppContext, folder: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """The `.env` mnm writes and the secret values in it."""
    try:
        before = {k: v for k, v in dotenv_values(folder / compose.ENV_FILE).items() if v}
    except OSError:
        before = {}
    secrets = {name: before.get(name) or generated(size) for name, size in SECRETS.items()}
    value: dict[str, str] = {**SETTINGS, **secrets}
    rag = rag_lines(ctx.manifest())
    key = ctx.settings.api_key_value()
    if rag is not None:
        value |= {**rag, "RAG_OPENAI_API_KEY": key or rag["RAG_OPENAI_API_KEY"]}
    if key:
        value["MNM_API_KEY"] = key
    hidden = [(name, text) for name, text in secrets.items()]
    return value, hidden + ([("MNM_API_KEY", key)] if key else [])


def rag_env(env: dict[str, str]) -> dict[str, str]:
    """The RAG API's own lines, out of LibreChat's `.env`:
    its embeddings (`RAG_*`, `EMBEDDINGS_*`), the JWT secret it checks LibreChat's requests
    with and the vector database's password; never the CREDS, Meili's key or the router's."""
    return {
        name: text
        for name, text in env.items()
        if name.startswith(compose.RAG_PREFIXES) or name in compose.RAG_SECRETS
    }


def files(
    ctx: AppContext, folder: Path
) -> tuple[list[tuple[AppFile, Path]], list[tuple[str, str]]]:
    env, secrets = env_value(ctx, folder)
    yaml_file = card_for(App.LIBRECHAT, ctx.manifest(), ctx.settings).files[0]
    rag = AppFile(f"librechat/{compose.RAG_ENV_FILE}", None, "env", rag_env(env))
    return (
        [
            (AppFile(f"librechat/{compose.ENV_FILE}", None, "env", env), folder / compose.ENV_FILE),
            (rag, folder / compose.RAG_ENV_FILE),
            (yaml_file, folder / "librechat.yaml"),
        ],
        secrets,
    )


def compose_for(ctx: AppContext, pin: ImagesPin, folder: Path) -> compose.Compose:
    host = bind_host(ctx, App.LIBRECHAT)
    return compose.librechat(pin, folder, PORT, compose.owner(), host)


def notes(_ctx: AppContext) -> list[str]:
    return [
        "registration is off: `mnm apps user add librechat EMAIL` creates an account (it "
        "asks for the password; mnm never keeps it)",
    ]


SPEC = ComposeSpec(
    App.LIBRECHAT,
    compose.LIBRECHAT_FOLDERS,
    files,
    compose_for,
    notes=notes,
    data_note="its accounts, chats, uploads and search index",
)
