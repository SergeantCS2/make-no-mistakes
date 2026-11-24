"""Goose's file: `~/.config/goose/custom_providers/mnm.json`, a custom provider with
the desk's roles and every chat model, and the `extensions` block for mnm's MCP tools.

From Goose's docs, read 2026-10-22 (v1.55.0, goose-docs.ai; the repo moved to aaif-goose):
goose-docs.ai/docs/getting-started/providers (a JSON in `custom_providers/`: `name`, `engine:
openai`, `display_name`, `description`, `api_key_env`, `base_url` as the **full**
`/v1/chat/completions` URL, `models` with `name` and `context_limit`, `headers`,
`supports_streaming`, `requires_auth`; on Windows `%APPDATA%\\Block\\goose\\config\\
custom_providers\\`), goose-docs.ai/docs/getting-started/using-extensions (config.yaml
`extensions.<name>`: `type: streamable_http`, `uri`, `enabled`, `timeout`; a `headers` map is
not documented there), goose-docs.ai/docs/guides/environment-variables (`GOOSE_PROVIDER`,
`GOOSE_MODEL`). The provider file is mnm's own (one file, written whole); config.yaml is
Goose's, which `goose configure` rewrites, so the extension is a block to paste. Goose runs on
this machine or as the Windows desktop app (its documented route), both through localhost.
"""

from __future__ import annotations

from makenomistakes.apps.card import (
    AppCard,
    AppFile,
    Doc,
    Uses,
    chat_offers,
    context_line,
    key_line,
    pick,
    pick_line,
    render,
)
from makenomistakes.apps.names import App, config_home
from makenomistakes.manifest import Manifest

PROVIDER = "mnm"
LOCALHOST = "localhost"  # Windows reaches WSL's ports by this name (localhost forwarding)


def _local(url: str) -> str:
    return url.replace("127.0.0.1", LOCALHOST, 1)


def file_value(manifest: Manifest) -> Doc:
    return {
        "name": PROVIDER,
        "engine": "openai",
        "display_name": "makenomistakes",
        "description": "Local models through mnm's router: the desk's roles first",
        "api_key_env": "MNM_API_KEY",
        "base_url": f"{_local(manifest.openai.loopback)}/chat/completions",
        "models": [  # a context_limit only when known (Goose's own default stands otherwise)
            {"name": o.id, **({"context_limit": o.entry.context} if o.entry.context else {})}
            for o in chat_offers(manifest)
        ],
        "headers": {},
        "supports_streaming": True,
        "requires_auth": manifest.key_required,
    }


def extension(manifest: Manifest) -> Doc:
    return {
        "extensions": {
            PROVIDER: {
                "name": PROVIDER,
                "type": "streamable_http",
                "uri": _local(manifest.mcp.loopback),
                "enabled": True,
                "timeout": 300,
            }
        }
    }


def card(manifest: Manifest) -> AppCard:
    chosen = pick(manifest, ("coder", "chat"))
    model = chosen.id if chosen else "coder"
    mcp_key = (
        " Goose documents no header for an extension, so with MNM_API_KEY set the MCP tools "
        "cannot be reached from it (the router asks every request for the key)."
        if manifest.key_required
        else ""
    )
    lines = [
        pick_line(App.GOOSE, chosen, "coder", "an agent: its tools act for you"),
        f"Then `GOOSE_PROVIDER={PROVIDER} GOOSE_MODEL={model} goose session`, or pick "
        f"makenomistakes and {model} in Goose's settings.",
        *([line] if (line := context_line(App.GOOSE, chosen)) else []),
        key_line(
            manifest,
            "Goose sends $MNM_API_KEY (api_key_env), so export it first or "
            "enter it when Goose asks",
        ),
        "The Windows desktop app reads the same file from %APPDATA%\\Block\\goose\\config\\"
        "custom_providers\\mnm.json; localhost reaches WSL from Windows.",
        "mnm's MCP tools: add the extensions block below to Goose's config.yaml (or "
        f"`goose configure` → Add Extension → Remote Extension (Streamable HTTP)).{mcp_key}",
    ]
    return AppCard(
        App.GOOSE,
        lines,
        Uses(
            base_url=_local(manifest.openai.loopback),
            chat_model=chosen.id if chosen else None,
            mcp_url=_local(manifest.mcp.loopback),
            mcp_takes_key=False,  # no documented header for an extension
        ),
        files=[
            AppFile(
                "~/.config/goose/custom_providers/mnm.json",
                config_home() / "goose" / "custom_providers" / "mnm.json",
                "json",
                file_value(manifest),
            )
        ],
        blocks=[("Goose's config.yaml (mnm's MCP tools)", render("yaml", extension(manifest)))],
    )
