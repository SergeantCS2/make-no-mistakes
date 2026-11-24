"""Odysseus's card: its `.env` lines (embeddings), the LLM
endpoint and the Image endpoint to add in its Settings, and the MCP Tool Server entry, all by
desk role, so a model swap never touches Odysseus.

What Odysseus takes, as the per-instance cards (`harness.py`) also say:
embeddings only from `EMBEDDING_URL` (the full `/v1/embeddings` URL) and `EMBEDDING_MODEL` in
`~/odysseus/.env`, then `docker compose up -d`; an LLM endpoint at a base URL with `/v1`
(the router's `/chat/v1` lists the chat models only); an Image endpoint at a URL of its own
(`/upstream/image/v1`, which the router resolves to the desk's image); image upload offered for
ids with `vl` or `vision` (the `vision` role has it); the MCP form has no key field
(`bridge/card.py`). `mnm apps connect odysseus` writes the `.env` lines after showing the
diff; here `--write --to PATH` merges them into a file of the user's choosing."""

from __future__ import annotations

from makenomistakes.apps.card import (
    AppCard,
    AppFile,
    Doc,
    Uses,
    context_line,
    key_line,
    one_of,
    pick,
    pick_line,
    role,
)
from makenomistakes.apps.names import App
from makenomistakes.bridge.card import mcp_card
from makenomistakes.config import Settings
from makenomistakes.hub.facts import Modality
from makenomistakes.manifest import Manifest
from makenomistakes.routing import UPSTREAM

FAMILY_NOTE = (
    "Odysseus reads thinking and tool support from the model id's family name (`qwen3` …), "
    "which `{role}` lacks: pick `{name}` itself to keep them (a model "
    "swap then touches Odysseus), or set the endpoint's supports_tools by hand."
)


def env_value(manifest: Manifest) -> Doc | None:
    memory = one_of(manifest, "memory", Modality.EMBEDDING)
    if memory is None:
        return None
    return {
        "EMBEDDING_URL": f"{manifest.openai.containers}/embeddings",
        "EMBEDDING_MODEL": memory.id,
    }


def card(manifest: Manifest, settings: Settings) -> AppCard:
    chosen = pick(manifest, ("chat", "coder"))
    root = manifest.openai.containers.removesuffix("/v1")
    image = role(manifest, "image")
    vision = role(manifest, "vision")
    picked = chosen.id if chosen else "chat"
    endpoints = (
        "Settings → Add Models → Add Local Models\n"
        f"  type LLM:   {root}/chat/v1  (lists the chat models; pick `{picked}`)\n"
        + (
            f"  type Image: {root}{UPSTREAM}image/v1  (the desk's image, {image.entry.name} now)\n"
            if image
            else ""
        )
    )
    lines = [
        pick_line(App.ODYSSEUS, chosen, "chat", "a chat"),
        *([line] if (line := context_line(App.ODYSSEUS, chosen)) else []),
        *(["`vision` carries the token Odysseus offers image upload by."] if vision else []),
        *(
            [FAMILY_NOTE.format(role=chosen.id, name=chosen.entry.name)]
            if chosen and chosen.alias
            else []
        ),
        "The LLM endpoint's /chat/v1 lists chat models only: at a /v1 URL `memory`, `ear` and "
        "`voice` would show in its chat list (it hides non-chat ids by the `embed`, `whisper` "
        "and `tts` tokens the role ids lack) unless pinned.",
        "The LLM endpoint never shares the Image endpoint's URL (Odysseus would send every "
        "message of that chat to its image tools).",
        key_line(
            manifest,
            "enter it as each endpoint's API key; the .env lines carry none, "
            "and the MCP form has no key field",
        ),
    ]
    env = env_value(manifest)
    if env is None:
        lines.append(
            "No embedding model: Odysseus keeps its built-in one (make one the desk's memory)."
        )
    else:
        lines.append(
            "Add the .env lines to ~/odysseus/.env, then `docker compose up -d` in ~/odysseus "
            "(`--write --to ~/odysseus/.env` merges them, a copy of the file kept)."
        )
    return AppCard(
        App.ODYSSEUS,
        lines,
        Uses(
            base_url=f"{root}/chat/v1",
            chat_model=chosen.id if chosen else None,
            embed_model=str(env["EMBEDDING_MODEL"]) if env else None,
            embed_base=manifest.openai.containers,
            mcp_url=manifest.mcp.containers,
            mcp_takes_key=False,  # its MCP form has no key field
        ),
        files=[AppFile("~/odysseus/.env", None, "env", env)] if env else [],
        blocks=[
            ("Odysseus's endpoints", endpoints),
            ("Odysseus's MCP Tool Server", "\n".join(mcp_card(settings)) + "\n"),
        ],
    )
