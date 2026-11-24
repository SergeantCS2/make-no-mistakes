"""OpenCode 1.x's file: `~/.config/opencode/opencode.json` with one `mnm` provider
listing the desk's roles and every chat model, and mnm's MCP tools.

From OpenCode's docs, read 2026-10-22 (the 1.x line, pinned at 1.19.4; 2.x changes both
schemas): opencode.ai/docs/providers (a provider is `npm: @ai-sdk/openai-compatible`,
`options.baseURL`, `options.apiKey`, `models.<id>.limit.context`/`.output`, the max input and
generated tokens OpenCode budgets with), opencode.ai/docs/config (the global file, JSON or
JSONC, schema https://opencode.ai/config.json; `{env:NAME}` reads the environment),
opencode.ai/docs/mcp-servers (`mcp.<name>`: `type: remote`, `url`, `headers`, `enabled`,
`oauth`). OpenCode runs on this machine (native), so it reaches the
router at 127.0.0.1. Merged into a file of the user's: `provider.mnm` and `mcp.mnm` only.
"""

from __future__ import annotations

from typing import Any

from makenomistakes.apps.card import (
    KEY_PLACEHOLDER,
    AppCard,
    AppFile,
    Doc,
    Offer,
    Uses,
    chat_offers,
    context_line,
    key_line,
    pick,
    pick_line,
    with_keys,
)
from makenomistakes.apps.names import App, config_home
from makenomistakes.manifest import Manifest

PROVIDER = "mnm"
KEY_REF = "{env:MNM_API_KEY}"


def _model(offer: Offer) -> dict[str, Any]:
    entry = offer.entry
    shown = f"{offer.id} ({entry.name}, local)" if offer.alias else f"{offer.id} (local)"
    found: dict[str, Any] = {"name": shown, "tool_call": bool(entry.tools)}
    if entry.context and entry.max_output:  # a limit OpenCode budgets with, only when known
        found["limit"] = {"context": entry.context, "output": entry.max_output}
    return found


def file_value(manifest: Manifest) -> Doc:
    """The whole file mnm writes when it is new or its own."""
    keyed = manifest.key_required
    mcp: Doc = {"type": "remote", "url": manifest.mcp.loopback, "enabled": True, "oauth": False}
    if keyed:
        mcp["headers"] = {"Authorization": f"Bearer {KEY_REF}"}
    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            PROVIDER: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "makenomistakes",
                "options": {
                    "baseURL": manifest.openai.loopback,
                    "apiKey": KEY_REF if keyed else KEY_PLACEHOLDER,
                },
                "models": {o.id: _model(o) for o in chat_offers(manifest)},
            }
        },
        "mcp": {PROVIDER: mcp},
    }


def card(manifest: Manifest) -> AppCard:
    chosen = pick(manifest, ("coder", "chat"))
    value = file_value(manifest)
    merge = with_keys(
        {
            ("provider", PROVIDER): value["provider"][PROVIDER],
            ("mcp", PROVIDER): value["mcp"][PROVIDER],
        }
    )
    lines = [
        pick_line(App.OPENCODE, chosen, "coder", "an agent: its tools edit and run code"),
        *(
            [
                f'Run `opencode -m {PROVIDER}/{chosen.id}`, or set "model": '
                f'"{PROVIDER}/{chosen.id}" in the file.'
            ]
            if chosen
            else []
        ),
        *([line] if (line := context_line(App.OPENCODE, chosen)) else []),
        key_line(
            manifest,
            "OpenCode sends {env:MNM_API_KEY}, so export it in the shell that runs opencode",
        ),
        "mnm's MCP tools (its workflows, the short film, run_status) are the `mnm` server.",
    ]
    return AppCard(
        App.OPENCODE,
        lines,
        Uses(
            base_url=manifest.openai.loopback,
            chat_model=chosen.id if chosen else None,
            mcp_url=manifest.mcp.loopback,
        ),
        files=[
            AppFile(
                "~/.config/opencode/opencode.json",
                config_home() / "opencode" / "opencode.json",
                "json",
                value,
                merge,
            )
        ],
    )
