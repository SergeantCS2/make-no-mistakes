"""Open WebUI's settings from the router's manifest, by desk role, as environment variables
(Open WebUI has no config file of its own: every setting below is a variable its v0.12.3
`backend/open_webui/config.py` reads, each named in docs.openwebui.com/reference/
env-configuration, read 2026-10-23). `mnm apps install open-webui` (`open_webui_app.py`) puts
them in its compose project; `mnm url --for open-webui` prints them for an Open WebUI of the
user's own.

What is wired, each only when the desk (or the one instance of that kind) has it; a role the
desk lacks is left out, never invented:
- chat: `OPENAI_API_BASE_URL` is the router's `/chat/v1`, which lists the chat, coding and
  vision models only (at `/v1` its model picker would offer `memory` and `voice` to chat with),
  `ENABLE_OPENAI_API`, `DEFAULT_MODELS` the desk's chat (else coder); `ENABLE_OLLAMA_API=false`:
  the router's own Ollama API would list every model twice;
- speech: `AUDIO_STT_ENGINE=openai` with the `ear` (`/v1/audio/transcriptions`, multipart),
  `AUDIO_TTS_ENGINE=openai` with the `voice` and its default voice (Open WebUI's own default,
  `alloy`, is a name audio.cpp's voices refuse);
- pictures: `ENABLE_IMAGE_GENERATION`, `IMAGE_GENERATION_ENGINE=openai` with the `image`; it
  sends `{model, prompt, n, size, response_format: b64_json}` to `/v1/images/generations`
  (routers/images.py), and the router drops `size` for an image instance (unless
  `client-size=on`), so `IMAGE_SIZE` is the instance's own size and says what comes back;
- memory: `RAG_EMBEDDING_ENGINE=openai` with the `memory`; without one Open WebUI embeds with
  its own small local model;
- tools: `TOOL_SERVER_CONNECTIONS`, one MCP (Streamable HTTP) server, native since v0.6.31
  (docs.openwebui.com/features/extensibility/mcp): `type: mcp`, `auth_type` none (keyless from
  its container: `LocalPeers`) or bearer with the key; `info.id` is what a chat's tool id
  `server:mcp:mnm` names. Without `config.access_grants` it is the admin's alone (its access
  control: `has_connection_access`), which on mnm's install is the user;
- accounts: `WEBUI_AUTH=true`, `ENABLE_SIGNUP=false`. Open WebUI lets the first account sign up
  whatever `ENABLE_SIGNUP` says and makes it the admin (routers/auths.py `signup`: the gate
  applies only once a user exists), so nobody but that first account ever signs up.
- `ENABLE_PERSISTENT_CONFIG=false`: every setting above is a `ConfigVar` (its docs' word),
  which Open WebUI copies into its database at the first start and then reads from there,
  ignoring the variable. Off, the environment wins at every start, so a desk change reaches
  Open WebUI with `mnm apps install open-webui`; the price is that a change made in its Admin
  Settings lasts only until the next restart (its docs' "CRITICAL WARNING").
- telemetry off (`SCARF_NO_ANALYTICS`, `DO_NOT_TRACK`, `ANONYMIZED_TELEMETRY`, as its
  Dockerfile sets them anyway) and `ENABLE_VERSION_UPDATE_CHECK=false` (mnm pins the build:
  `mnm apps update` moves it)."""

from __future__ import annotations

import json
from collections.abc import Callable

from makenomistakes.apps.card import (
    KEY_PLACEHOLDER,
    AppCard,
    AppFile,
    Doc,
    Uses,
    context_line,
    key_line,
    one_of,
    pick,
    pick_line,
)
from makenomistakes.apps.names import App
from makenomistakes.config import Settings
from makenomistakes.engine_args import SdCppArgs
from makenomistakes.hub.facts import Modality
from makenomistakes.manifest import Manifest
from makenomistakes.store import InstanceStore, StoreError

SERVER_ID = "mnm"
KEY_SHOWN = "<your MNM_API_KEY>"  # the card's value where the key goes; mnm's install has it
# The settings that never depend on the desk (the compose file's `environment`).
FIXED = {
    "WEBUI_AUTH": "true",
    "ENABLE_SIGNUP": "false",
    "ENABLE_PERSISTENT_CONFIG": "false",
    "ENABLE_OLLAMA_API": "false",
    "ENABLE_OPENAI_API": "true",
    "ENABLE_VERSION_UPDATE_CHECK": "false",
    "SCARF_NO_ANALYTICS": "true",
    "DO_NOT_TRACK": "true",
    "ANONYMIZED_TELEMETRY": "false",
}
# The variables that carry the router's key (the `.env`, 0600, never the compose file).
KEYED = (
    "OPENAI_API_KEY",
    "AUDIO_STT_OPENAI_API_KEY",
    "AUDIO_TTS_OPENAI_API_KEY",
    "IMAGES_OPENAI_API_KEY",
    "RAG_OPENAI_API_KEY",
)
ImageSize = Callable[[str], str | None]  # an image instance's `WxH`, by its name


def sizes(settings: Settings) -> ImageSize:
    """An image instance's size from its recipe (None: no such recipe, or not sd.cpp's)."""

    def size(name: str) -> str | None:
        try:
            args = InstanceStore(settings.expanded_state_dir()).load(name).args
        except StoreError:
            return None
        return f"{args.width}x{args.height}" if isinstance(args, SdCppArgs) else None

    return size


def chat_base(manifest: Manifest) -> str:
    return manifest.openai.containers.removesuffix("/v1") + "/chat/v1"


def tool_server(manifest: Manifest, key: str) -> list[Doc]:
    """`TOOL_SERVER_CONNECTIONS`' one entry: mnm's MCP tools."""
    return [
        {
            "type": "mcp",
            "url": manifest.mcp.containers,
            "path": "",
            "auth_type": "bearer" if manifest.key_required else "none",
            "key": key if manifest.key_required else "",
            "config": {"enable": True},
            "info": {
                "id": SERVER_ID,
                "name": "mnm",
                "description": "mnm's tools: its ComfyUI workflows, the short film, a run's status",
            },
        }
    ]


def wired(manifest: Manifest, key: str, image_size: ImageSize) -> dict[str, str]:
    """The desk-dependent settings; `key` is what each `*_API_KEY` holds (the router's key, or
    `KEY_PLACEHOLDER` when it asks none: Open WebUI sends whatever it is given)."""
    v1 = manifest.openai.containers
    value: dict[str, str] = {"OPENAI_API_BASE_URL": chat_base(manifest), "OPENAI_API_KEY": key}
    if (chosen := pick(manifest, ("chat", "coder"))) is not None:
        value["DEFAULT_MODELS"] = chosen.id
    if (ear := one_of(manifest, "ear", Modality.SPEECH_TO_TEXT)) is not None:
        value |= {
            "AUDIO_STT_ENGINE": "openai",
            "AUDIO_STT_OPENAI_API_BASE_URL": v1,
            "AUDIO_STT_OPENAI_API_KEY": key,
            "AUDIO_STT_MODEL": ear.id,
        }
    if (voice := one_of(manifest, "voice", Modality.TEXT_TO_SPEECH)) is not None:
        value |= {
            "AUDIO_TTS_ENGINE": "openai",
            "AUDIO_TTS_OPENAI_API_BASE_URL": v1,
            "AUDIO_TTS_OPENAI_API_KEY": key,
            "AUDIO_TTS_MODEL": voice.id,
        }
        if voice.entry.voices:
            value["AUDIO_TTS_VOICE"] = voice.entry.voices[0]
    if (image := one_of(manifest, "image", Modality.IMAGE_GEN)) is not None:
        value |= {
            "ENABLE_IMAGE_GENERATION": "true",
            "IMAGE_GENERATION_ENGINE": "openai",
            "IMAGES_OPENAI_API_BASE_URL": v1,
            "IMAGES_OPENAI_API_KEY": key,
            "IMAGE_GENERATION_MODEL": image.id,
        }
        if (size := image_size(image.entry.name)) is not None:
            value["IMAGE_SIZE"] = size
    if (memory := one_of(manifest, "memory", Modality.EMBEDDING)) is not None:
        value |= {
            "RAG_EMBEDDING_ENGINE": "openai",
            "RAG_OPENAI_API_BASE_URL": v1,
            "RAG_OPENAI_API_KEY": key,
            "RAG_EMBEDDING_MODEL": memory.id,
        }
    value["TOOL_SERVER_CONNECTIONS"] = json.dumps(tool_server(manifest, key), separators=(",", ":"))
    return value


def environment(
    manifest: Manifest, key: str | None, image_size: ImageSize, url: str
) -> dict[str, str]:
    """Every variable mnm sets: the fixed ones, `WEBUI_URL`, the wired ones."""
    sent = (key or KEY_SHOWN) if manifest.key_required else KEY_PLACEHOLDER
    return {**FIXED, "WEBUI_URL": url, **wired(manifest, sent, image_size)}


def _lines(manifest: Manifest, env: dict[str, str]) -> list[str]:
    chosen = pick(manifest, ("chat", "coder"))
    said = {
        "AUDIO_STT_MODEL": "speech to text with `{}`",
        "AUDIO_TTS_MODEL": "read aloud with `{}`",
        "IMAGE_GENERATION_MODEL": "pictures with `{}`",
        "RAG_EMBEDDING_MODEL": "documents (RAG) embedded with `{}`",
    }
    wired_parts = [text.format(env[name]) for name, text in said.items() if name in env]
    missing = {
        "AUDIO_STT_MODEL": "ear",
        "AUDIO_TTS_MODEL": "voice",
        "IMAGE_GENERATION_MODEL": "image",
        "RAG_EMBEDDING_MODEL": "memory",
    }
    lacking = [role for name, role in missing.items() if name not in env]
    return [
        pick_line(App.OPEN_WEBUI, chosen, "chat", "a chat; new chats open on it"),
        *([line] if (line := context_line(App.OPEN_WEBUI, chosen)) else []),
        "Wired: " + ("; ".join(wired_parts) or "chat only") + "; mnm's MCP tools.",
        *(
            [
                f"No desk {', '.join(lacking)}: left out (Open WebUI's own defaults; a document "
                "is then embedded by its small local model)."
            ]
            if lacking
            else []
        ),
        key_line(manifest, "each *_API_KEY and the MCP entry's key carry it"),
        "The first account to sign up is its admin; sign-up is off for everyone after it.",
        "ENABLE_PERSISTENT_CONFIG=false: these lines win at every start (a change in its "
        "Admin Settings lasts until the next restart).",
        "`mnm apps install open-webui` runs it with all of this; for your own Open WebUI, put "
        "the lines in its environment (`--write --to <its .env>` merges them).",
    ]


def card(manifest: Manifest, image_size: ImageSize, url: str = "http://localhost:3000") -> AppCard:
    env = environment(manifest, None, image_size, url)
    chosen = pick(manifest, ("chat", "coder"))
    memory = one_of(manifest, "memory", Modality.EMBEDDING)
    return AppCard(
        App.OPEN_WEBUI,
        _lines(manifest, env),
        Uses(
            base_url=chat_base(manifest),
            chat_model=chosen.id if chosen else None,
            embed_model=memory.id if memory else None,
            embed_base=manifest.openai.containers,
            mcp_url=manifest.mcp.containers,
        ),
        files=[AppFile("Open WebUI's environment (.env)", None, "env", dict(env))],
    )
