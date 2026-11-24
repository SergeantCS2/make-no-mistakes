"""OpenHands installed by mnm: Agent Canvas in Docker, never the Docker
socket (`compose.openhands`), and its LLM and MCP settings set by mnm once it answers.

Agent Canvas keeps its settings encrypted in `~/.openhands/settings.json` (with the key its
entrypoint generates, `secret-key.txt`), so mnm does not write that file: it sets them through
Agent Canvas's own settings API, `PATCH /api/settings` with `X-Session-API-Key` (its OpenAPI
at `/openapi.json`, read from the 1.31.0 image on 2026-10-23): `agent_settings_diff.llm`
(`model: openai/<id>`, `base_url`, `api_key`, `max_input_tokens`, which the docs do not list
but the API takes: the instance's per-request context, so OpenHands' condenser budgets with
what the plan gave it) and `agent_settings_diff.mcp_config.mnm` (`url`, `transport:
streamable-http`, a bearer `auth` when the router asks for the key). The values are its
card's (`apps/openhands.py`). Set at install and update only: a model the user picks in its
Settings later stays.

The session key (`OH_SESSION_API_KEYS_0`, below) is generated into the secrets file as
`MNM_OPENHANDS_KEY` and handed to the container in its `.env` (0600); the page asks for it
once per browser."""

from __future__ import annotations

from pathlib import Path

from makenomistakes import config
from makenomistakes.apps import compose
from makenomistakes.apps.card import AppFile, pick
from makenomistakes.apps.cards import card_for
from makenomistakes.apps.compose_app import ComposeSpec
from makenomistakes.apps.context import AppContext, generated
from makenomistakes.apps.defs import DEFS, ImagesPin
from makenomistakes.apps.exposure import bind_host
from makenomistakes.apps.names import App
from makenomistakes.apps.projects import checked_projects_dir
from makenomistakes.consumers import origin
from makenomistakes.router_bind import LOOPBACK
from makenomistakes.setup import write_setting

KEY = "MNM_OPENHANDS_KEY"
# The agent server's own setting, not the docs' LOCAL_BACKEND_API_KEY: the 1.31.0 entrypoint
# exports OH_SESSION_API_KEYS_0 only when it generates a key itself, so a given
# LOCAL_BACKEND_API_KEY reaches the page's proxy but not the agent server, which then answers
# every /api request without one (GET and PATCH /api/settings answer 200 with no key, 401
# with OH_SESSION_API_KEYS_0 set).
SESSION_KEYS = "OH_SESSION_API_KEYS_0"
PORT = DEFS[App.OPENHANDS].port or 8000
SETTINGS_URL = origin(LOOPBACK, PORT) + "/api/settings"


def session_key(ctx: AppContext) -> str:
    found = ctx.settings.app_secret(KEY)
    if found:
        return found
    made = generated(32)
    write_setting(config.SECRETS_FILE, KEY, made)
    ctx.say(f"generated OpenHands' session key into the secrets file as {KEY}")
    return made


def files(
    ctx: AppContext, folder: Path
) -> tuple[list[tuple[AppFile, Path]], list[tuple[str, str]]]:
    key = session_key(ctx)
    env = AppFile(f"openhands/{compose.ENV_FILE}", None, "env", {SESSION_KEYS: key})
    return [(env, folder / compose.ENV_FILE)], [(KEY, key)]


def compose_for(ctx: AppContext, pin: ImagesPin, folder: Path) -> compose.Compose:
    projects = checked_projects_dir(ctx.settings)
    projects.mkdir(parents=True, exist_ok=True)
    host = bind_host(ctx, App.OPENHANDS)
    return compose.openhands(pin, folder, projects, PORT, compose.owner(), host)


def settings_diff(ctx: AppContext) -> dict[str, object]:
    """What mnm sets in Agent Canvas: the card's model, base URL and MCP server."""
    manifest = ctx.manifest()
    card = card_for(App.OPENHANDS, manifest, ctx.settings)
    key = ctx.settings.api_key_value()
    chosen = pick(manifest, ("coder", "chat"))
    llm: dict[str, object] = {
        "model": f"openai/{card.uses.chat_model or 'coder'}",
        "base_url": card.uses.base_url,
        "api_key": key or "mnm",  # Agent Canvas wants one; the router asks none
    }
    if chosen is not None and chosen.entry.context:
        llm["max_input_tokens"] = chosen.entry.context
    mcp: dict[str, object] = {"url": card.uses.mcp_url, "transport": "streamable-http"}
    if key:
        mcp["auth"] = {"strategy": "bearer", "value": key}
    return {"agent_settings_diff": {"llm": llm, "mcp_config": {"mnm": mcp}}}


def apply_settings(ctx: AppContext, _folder: Path) -> None:
    """Said, never raised, when it does not take: OpenHands runs either way. First, that its
    API refuses a request without the session key (an open one would let any page or program
    on this machine drive its agent)."""
    if ctx.check(SETTINGS_URL, None) == 200:
        ctx.say(
            "WARNING: OpenHands answers its settings without the session key: anything on this "
            "machine can drive its agent. `mnm apps stop openhands` until `mnm apps install "
            "openhands` again shows no such line."
        )
    session = session_key(ctx)
    headers = {"X-Session-API-Key": session, "Content-Type": "application/json"}
    status, text = ctx.patch(SETTINGS_URL, headers, settings_diff(ctx))
    if status != 200:  # a 422 can echo the request, its llm.api_key included
        sent = [(KEY, session)]
        if (router_key := ctx.settings.api_key_value()) is not None:
            sent.append(("MNM_API_KEY", router_key))
        shown = ctx.redacted(text, sent)[:200]
        ctx.say(
            f"OpenHands answered {status or 'nothing'} to mnm's settings ({shown}): it "
            "runs, but with its own LLM settings; enter the card's (`mnm url --for openhands`) "
            "in its Settings, or `mnm apps install openhands` again."
        )
        return
    ctx.say("set OpenHands' model and mnm's MCP tools (through its settings API)")


def notes(ctx: AppContext) -> list[str]:
    return [
        f"the page asks for its session key once per browser: {KEY} in the secrets file "
        f"({config.SECRETS_FILE}); never printed",
        f"the agent works in {ctx.settings.expanded_projects_dir()} (mounted at /projects), "
        "in its own container: no Docker socket",
    ]


SPEC = ComposeSpec(
    App.OPENHANDS,
    ("home",),
    files,
    compose_for,
    after_start=apply_settings,
    notes=notes,
    data_note="its home: settings, conversations and its automation database",
)
