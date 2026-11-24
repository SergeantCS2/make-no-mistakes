"""OpenHands' settings card: Agent Canvas keeps its LLM and MCP settings in its own
`~/.openhands/settings.json`, saved encrypted, so mnm writes nothing and the card says what to
enter.

From OpenHands' docs, read 2026-10-22 (Agent Canvas 1.31.0):
docs.openhands.dev/openhands/usage/agent-canvas/model-configuration and …/settings/llm-settings
(Settings → LLM → Advanced: Custom Model `openai/<id from /v1/models>`, Base URL, any API key;
from Docker the base URL is `host.docker.internal`), …/settings/mcp-settings (Customize → MCP
Servers, SHTTP: URL, Authentication None/Bearer, API Key sent as `Authorization`). No field for
the context exists in Agent Canvas (V0's `max_input_tokens` is gone): the 32k floor is held by
mnm's plan for the instance (`--for-agents`), which the card checks."""

from __future__ import annotations

from makenomistakes.apps.card import (
    KEY_PLACEHOLDER,
    AppCard,
    Uses,
    context_line,
    key_line,
    pick,
    pick_line,
)
from makenomistakes.apps.names import App
from makenomistakes.client_view import DOCKER_HOST
from makenomistakes.manifest import Manifest

NO_WRITE = (
    "OpenHands keeps these settings encrypted in its own settings.json, which mnm does not "
    "write: enter them in its Settings, or `mnm apps install openhands` sets them through its "
    "settings API."
)


def card(manifest: Manifest) -> AppCard:
    chosen = pick(manifest, ("coder", "chat"))
    model = chosen.id if chosen else "coder"
    keyed = manifest.key_required
    key = "your MNM_API_KEY" if keyed else f"{KEY_PLACEHOLDER} (any text: no key is set)"
    settings = (
        "Settings → LLM → Advanced\n"
        f"  Custom Model: openai/{model}\n"
        f"  Base URL:     {manifest.openai.containers}\n"
        f"  API Key:      {key}\n"
        "Customize → MCP Servers → Add server (SHTTP)\n"
        f"  URL:            {manifest.mcp.containers}\n"
        f"  Authentication: {'Bearer, API Key: your MNM_API_KEY' if keyed else 'None'}\n"
    )
    lines = [
        pick_line(App.OPENHANDS, chosen, "coder", "an agent: it edits, runs and tests code"),
        *([line] if (line := context_line(App.OPENHANDS, chosen)) else []),
        "OpenHands has no context field: what the plan gives the instance is what it gets.",
        key_line(manifest, "enter it as the API key and as the MCP server's Bearer key"),
        *(
            [
                "OpenHands runs in Docker here, so the URLs name host.docker.internal; its "
                "container needs `--add-host host.docker.internal:host-gateway` (`mnm apps "
                "install` maps it)."
            ]
            if DOCKER_HOST in manifest.openai.containers
            else []
        ),
    ]
    return AppCard(
        App.OPENHANDS,
        lines,
        Uses(
            base_url=manifest.openai.containers,
            chat_model=chosen.id if chosen else None,
            mcp_url=manifest.mcp.containers,
        ),
        blocks=[("OpenHands' settings", settings)],
        no_write=NO_WRITE,
    )
