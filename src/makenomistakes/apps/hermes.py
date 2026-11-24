"""Hermes Agent's card: a card only, no file. From its docs (read 2026-10-22): any
OpenAI-compatible `/v1` backend as a custom provider, tool calls (llama.cpp's `--jinja`, which
every mnm chat recipe runs), and **64k context per request, refused at startup below it**,
which the card checks. Its `config.yaml` field names were not read, so mnm writes nothing and
names none: the card gives the values to enter."""

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
from makenomistakes.manifest import Manifest

NO_WRITE = (
    "mnm writes no file for Hermes Agent yet (its config.yaml's fields were not checked): "
    "enter the values on the card as its custom provider."
)


def card(manifest: Manifest) -> AppCard:
    chosen = pick(manifest, ("coder", "chat"))
    model = chosen.id if chosen else "coder"
    key = "your MNM_API_KEY" if manifest.key_required else f"{KEY_PLACEHOLDER} (any text)"
    values = (
        "A custom (OpenAI-compatible) provider:\n"
        f"  base URL: {manifest.openai.loopback}\n"
        f"  model:    {model}\n"
        f"  API key:  {key}\n"
        f"mnm's MCP tools: {manifest.mcp.loopback} (Streamable HTTP)\n"
    )
    lines = [
        pick_line(App.HERMES, chosen, "coder", "an agent that calls tools in long loops"),
        *([line] if (line := context_line(App.HERMES, chosen)) else []),
        "Hermes wants one request at a time on its model: an instance with `parallel=1` keeps "
        "the whole context for it.",
        key_line(manifest, "enter it as the provider's API key"),
    ]
    return AppCard(
        App.HERMES,
        lines,
        Uses(
            base_url=manifest.openai.loopback,
            chat_model=chosen.id if chosen else None,
            mcp_url=manifest.mcp.loopback,
        ),
        blocks=[("Hermes Agent's provider", values)],
        no_write=NO_WRITE,
    )
