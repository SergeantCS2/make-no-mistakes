"""LibreChat's file: `librechat.yaml` with mnm as a custom endpoint, its speech and
mnm's MCP tools; the `.env` lines for its RAG embeddings beside it.

From LibreChat's docs, read 2026-10-22 (pinned at v0.8.7):
librechat.ai/docs/configuration/librechat_yaml/object_structure/custom_endpoint (`name`,
`apiKey` required, any string; `baseURL`; `models.default` required and `fetch: true` reads
`/models`; `titleConvo`, `titleMethod`, `titleModel`, `modelDisplayLabel`), …/config (`version:
1.3.15`), librechat.ai/docs/configuration/stt_tts (`speech.stt.openai`/`speech.tts.openai`
with `url`, `apiKey`, `model`, `voices`), …/mcp_servers and …/mcp_settings (`type:
streamable-http`, `url`, `headers`; `.internal` names are blocked unless `mcpSettings.
allowedAddresses` lists `host:port`), librechat.ai/docs/configuration/rag_api
(`EMBEDDINGS_PROVIDER=openai`, `EMBEDDINGS_MODEL`, `RAG_OPENAI_BASEURL`, `RAG_OPENAI_API_KEY`).
`${NAME}` in the yaml is read from LibreChat's `.env`.

Checked on 2026-10-23 against v0.8.7's own schema (`configSchema` of librechat-data-provider
0.8.509, the version its v0.8.7 tag carries): `allowedAddresses` (the SSRF exemption for private
addresses, `host:port`) sits under `endpoints` (the custom endpoint's baseURL) and `mcpSettings`;
v0.8.7 has none under `speech` (the docs read the newer line's), so none is written there.
Left out of the docs' example on purpose: `dropParams: ['stop']` (llama-server takes `stop`)
and `addParams: {max_tokens: 2048}` (it would cut an agent's reply). The TTS route is mnm's
`/v1/audio/speech`; the docs' example names `/v1/audio/synthesize`.

LibreChat runs in Docker (`host.docker.internal`); its chat endpoint is the router's `/chat/v1`,
which lists only the chat models, so `fetch: true` offers no voice or picture model to chat
with."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from makenomistakes.apps.card import (
    KEY_PLACEHOLDER,
    AppCard,
    AppFile,
    Doc,
    Uses,
    chat_offers,
    context_line,
    key_line,
    one_of,
    pick,
    pick_line,
)
from makenomistakes.apps.names import App
from makenomistakes.hub.facts import Modality
from makenomistakes.manifest import Manifest

ENDPOINT = "mnm"
VERSION = "1.3.15"
KEY_REF = "${MNM_API_KEY}"


def _host_port(url: str) -> str:
    """`host.docker.internal:7801` from a URL: what `allowedAddresses` lists."""
    return url.split("://", 1)[1].split("/", 1)[0]


def _chat_base(manifest: Manifest) -> str:
    return manifest.openai.containers.removesuffix("/v1") + "/chat/v1"


def _speech(manifest: Manifest, key: str) -> Doc:
    """STT and TTS blocks for the desk's `ear` and `voice` (or the one such instance)."""
    speech: Doc = {}
    if (ear := one_of(manifest, "ear", Modality.SPEECH_TO_TEXT)) is not None:
        url = f"{manifest.openai.containers}/audio/transcriptions"
        speech["stt"] = {"openai": {"url": url, "apiKey": key, "model": ear.id}}
    if (voice := one_of(manifest, "voice", Modality.TEXT_TO_SPEECH)) is not None:
        url = f"{manifest.openai.containers}/audio/speech"
        tts: Doc = {"url": url, "apiKey": key, "model": voice.id}
        if voice.entry.voices:
            tts["voices"] = list(voice.entry.voices)
        speech["tts"] = {"openai": tts}
    return speech


def file_value(manifest: Manifest) -> Doc:
    key = KEY_REF if manifest.key_required else KEY_PLACEHOLDER
    offers = chat_offers(manifest)
    title = pick(manifest, ("chat", "coder"))
    router = _host_port(manifest.openai.containers)
    endpoint: Doc = {
        "name": ENDPOINT,
        "apiKey": key,
        "baseURL": _chat_base(manifest),
        "models": {"default": [o.id for o in offers] or ["chat"], "fetch": True},
        "titleConvo": True,
        "titleMethod": "completion",
        "titleModel": title.id if title else "chat",
        "modelDisplayLabel": "mnm",
    }
    mcp: Doc = {"type": "streamable-http", "url": manifest.mcp.containers}
    if manifest.key_required:
        mcp["headers"] = {"Authorization": f"Bearer {KEY_REF}"}
    mcp |= {"timeout": 30000, "initTimeout": 10000, "startup": True}
    value: Doc = {
        "version": VERSION,
        "cache": True,
        "endpoints": {"allowedAddresses": [router], "custom": [endpoint]},
    }
    if speech := _speech(manifest, key):
        value["speech"] = speech
    value["mcpServers"] = {ENDPOINT: mcp}
    value["mcpSettings"] = {"allowedAddresses": [_host_port(manifest.mcp.containers)]}
    return value


def _with_address(block: object, addresses: list[str]) -> Doc:
    """`block` (a dict, or nothing) with `allowedAddresses` holding each of `addresses`."""
    found: Doc = dict(block) if isinstance(block, dict) else {}
    allowed = list(found.get("allowedAddresses") or [])
    found["allowedAddresses"] = allowed + [a for a in addresses if a not in allowed]
    return found


def merge_into(ours: Doc) -> Callable[[Doc], Doc]:
    """Into a librechat.yaml of the user's: the `mnm` custom endpoint (replaced by name, or
    added), the router's address added to `endpoints.allowedAddresses` and
    `mcpSettings.allowedAddresses`, the speech providers mnm serves, `mcpServers.mnm`;
    everything else as it was."""

    def merge(doc: Doc) -> Doc:
        merged: Doc = {**doc}
        merged.setdefault("version", ours["version"])
        endpoints = _with_address(merged.get("endpoints"), ours["endpoints"]["allowedAddresses"])
        custom = [e for e in endpoints.get("custom") or [] if (e or {}).get("name") != ENDPOINT]
        endpoints["custom"] = [*custom, ours["endpoints"]["custom"][0]]
        merged["endpoints"] = endpoints
        if "speech" in ours:
            speech = dict(merged.get("speech") or {})
            for part, block in ours["speech"].items():
                speech[part] = {**(speech.get(part) or {}), **block}
            merged["speech"] = speech
        merged["mcpServers"] = {**(merged.get("mcpServers") or {}), **ours["mcpServers"]}
        addresses = ours["mcpSettings"]["allowedAddresses"]
        merged["mcpSettings"] = _with_address(merged.get("mcpSettings"), addresses)
        return merged

    return merge


def rag_lines(manifest: Manifest) -> dict[str, str] | None:
    """The `.env` lines for LibreChat's RAG embeddings, when there is a model to embed with."""
    memory = one_of(manifest, "memory", Modality.EMBEDDING)
    if memory is None:
        return None
    key = "<your MNM_API_KEY>" if manifest.key_required else KEY_PLACEHOLDER
    lines = {
        "EMBEDDINGS_PROVIDER": "openai",
        "EMBEDDINGS_MODEL": memory.id,
        "RAG_OPENAI_BASEURL": manifest.openai.containers,
        "RAG_OPENAI_API_KEY": key,
    }
    if manifest.key_required:
        lines["MNM_API_KEY"] = key  # what `${MNM_API_KEY}` in librechat.yaml reads
    return lines


def card(manifest: Manifest, default: Path | None = None) -> AppCard:
    """`default`: where `--write` puts the file, mnm's LibreChat's folder (`mnm apps`)."""
    chosen = pick(manifest, ("chat", "coder"))
    value = file_value(manifest)
    rag = rag_lines(manifest)
    memory = one_of(manifest, "memory", Modality.EMBEDDING)
    lines = [
        pick_line(App.LIBRECHAT, chosen, "chat", "a chat; its title model too"),
        *([line] if (line := context_line(App.LIBRECHAT, chosen)) else []),
        key_line(
            manifest,
            "librechat.yaml reads ${MNM_API_KEY} from LibreChat's .env, so add "
            "the key there (the lines below)",
        ),
        "Put the file beside LibreChat's docker-compose.yml, mounted as /app/librechat.yaml "
        "(`mnm apps install librechat` does it all), then restart its api container.",
        (
            f"Files and RAG embed with `{memory.id}`: the .env lines below."
            if memory
            else "No embedding model: add one and make it the desk's memory for RAG."
        ),
    ]
    blocks = (
        [("LibreChat's .env (RAG)", "".join(f"{k}={v}\n" for k, v in rag.items()))] if rag else []
    )
    return AppCard(
        App.LIBRECHAT,
        lines,
        Uses(
            base_url=_chat_base(manifest),
            chat_model=chosen.id if chosen else None,
            embed_model=memory.id if memory else None,
            embed_base=manifest.openai.containers,
            mcp_url=manifest.mcp.containers,
        ),
        files=[AppFile("librechat.yaml", default, "yaml", value, merge_into(value))],
        blocks=blocks,
    )
