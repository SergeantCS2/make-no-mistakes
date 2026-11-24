"""Cards for other apps: what each one's own settings call each field, filled in
with the router's URL, the instance's name and the context a request gets. Pure data, like
`harness`. The field names come from each app's documentation, read 2026-10-22:

- Home Assistant's Ollama integration: home-assistant.io/integrations/ollama (URL, Model, API
  Key; options Keep alive, default -1, and Context window size). It speaks Ollama's API: the
  router's `/api` (`ollama_http`), on `MNM_OLLAMA_PORT` when set.
- Karakeep: docs.karakeep.app/configuration (OPENAI_API_KEY, OPENAI_BASE_URL,
  INFERENCE_TEXT_MODEL, INFERENCE_IMAGE_MODEL, EMBEDDING_TEXT_MODEL, INFERENCE_CONTEXT_LENGTH
  default 2048, INFERENCE_MAX_OUTPUT_TOKENS, INFERENCE_JOB_TIMEOUT_SEC default 30).
- Frigate GenAI: docs.frigate.video/configuration/genai/genai_config (a named provider under
  `genai:` with `provider: openai`, `base_url`, `api_key`, `model`, `provider_options:
  context_size`; OPENAI_BASE_URL for the OpenAI provider; "You must use a vision-capable
  model with Frigate").
- n8n: docs.n8n.io/integrations/builtin/credentials/ollama (Base URL, API Key as a Bearer
  token). Its OpenAI credential page names only API Key and Organization ID, no Base URL, so
  the card offers the Ollama credential only.
- JetBrains AI Assistant: jetbrains.com/help/ai-assistant/use-custom-models.html (the 2026.2
  page; Settings | Tools | AI Assistant | Providers & API keys, Third-party AI providers,
  OpenAI-compatible: URL, API Key, Tool calling, Test Connection; Models Assignment's Context
  window, 64 000 by default).
- Raycast: manual.raycast.com/ai/custom-providers (~/.config/raycast/ai/providers.yaml:
  `providers` with `id`, `name`, `base_url`, `api_keys`, `models` with `id`, `name`,
  `context`, `abilities` `temperature`/`vision`/`system_message`/`tools`).
- Obsidian Copilot: github.com/logancyang/obsidian-copilot docs/llm-providers.md (Settings →
  Copilot → BYOK → Add a provider → Add a custom provider: Base URL, API key, model ID, Test,
  Enable CORS; earlier versions call it "3rd party (openai-format)").
- Brave Leo: brave.com/blog/byom-nightly (Settings → Leo → Bring your own model → Add new
  model: Label, Model request name, Server endpoint (the full chat completions URL), API Key).

Zed's two providers are `harness_editors.zed_card`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from makenomistakes.consumers import (
    CONSUMER_LABELS,
    ODYSSEUS_HOST,
    Consumer,
    HarnessUrl,
    context_tokens,
    key_set,
    ollama_origin,
    output_tokens,
    router_origin,
    shared_base_url,
)
from makenomistakes.engine_args_vllm import VllmArgs
from makenomistakes.hub.facts import CHAT_LIKE, Modality
from makenomistakes.recipes.model import Recipe

KEY = "your MNM_API_KEY"
LAN = "<this machine's address>"
TAILSCALE = (
    "From another machine, the router must listen there (GUIDE 14.5) or come over Tailscale "
    "(GUIDE 13.4)."
)


@dataclass(frozen=True)
class App:
    needs: int  # the context the app's own prompts take, in tokens
    why: str
    takes: frozenset[Modality]  # the kinds of model it can use
    wants: str  # what it needs, for the refusal
    build: Callable[[Recipe], list[HarnessUrl]]


def app_cards(recipe: Recipe, consumer: Consumer) -> list[HarnessUrl]:
    """One app's cards, the context check on a chat model's first; or why this instance
    cannot serve it."""
    app = APPS[consumer]
    label = CONSUMER_LABELS[consumer]
    if recipe.modality not in app.takes:
        return [
            HarnessUrl(
                label=label,
                url=shared_base_url(),
                note=f"{label} needs {app.wants}; {recipe.name} is "
                f"{recipe.modality.value.replace('_', ' ')}. Open the card of one that is "
                "(`mnm ls`), or add one with `mnm add <link>`.",
            )
        ]
    first, *rest = app.build(recipe)
    if recipe.modality in CHAT_LIKE:
        first = first.model_copy(update={"note": f"{first.note} {_context(recipe, label, app)}"})
    return [first, *rest]


def _context(recipe: Recipe, label: str, app: App) -> str:
    have = context_tokens(recipe)
    if have >= app.needs:
        return f"Context: {have:,} tokens per request, enough ({label}: {app.why})."
    return (
        f"Context: {have:,} tokens per request, too few: {label} needs about {app.needs:,} "
        f"({app.why}). `mnm set {recipe.name} ctx=…` raises it; `mnm ls` then says whether it "
        "still fits."
    )


def _ollama_url(host: str) -> str:
    """Where an Ollama client goes: the router's Ollama listener, else its own port."""
    return ollama_origin(host) or router_origin(host)


def _key_line(field: str) -> str:
    return f"{field}: {KEY}" if key_set() else f"{field}: leave it empty (no MNM_API_KEY is set)"


def _home_assistant(recipe: Recipe) -> list[HarnessUrl]:
    url = _ollama_url(LAN)
    docker = _ollama_url(ODYSSEUS_HOST)
    fields = (
        "Settings > Devices & services > Add integration > Ollama\n"
        f"  URL:      {url}\n"
        f"  {_key_line('API Key')}\n"
        f"  Model:    {recipe.name}\n"
        "Then the agent's options:\n"
        f"  Context window size: {context_tokens(recipe)}\n"
        "  Keep alive:          -1 (its default)\n"
    )
    port_note = (
        ""
        if ollama_origin() is not None
        else " MNM_OLLAMA_PORT=11434 in the secrets file (then `mnm service install`) gives it "
        "Ollama's usual port; the router's own port works as above."
    )
    return [
        HarnessUrl(
            label="Home Assistant (the Ollama integration)",
            url=url,
            note="Home Assistant speaks Ollama's API, which the router serves. Put this "
            f"machine's address in the URL (Home Assistant in Docker here: {docker})."
            f"{port_note} Pick the model it lists, never one it offers to download: mnm adds "
            "models with `mnm add`. Keep alive -1 switches the router's idle unload off for "
            f"the model; it is still unloaded when another needs the room. {TAILSCALE}",
            snippet=fields,
        )
    ]


def _karakeep(recipe: Recipe) -> list[HarnessUrl]:
    base = shared_base_url(ODYSSEUS_HOST)
    key = "OPENAI_API_KEY=<your MNM_API_KEY>" if key_set() else "OPENAI_API_KEY=local"
    if recipe.modality is Modality.EMBEDDING:
        lines = f"{key}\nOPENAI_BASE_URL={base}\nEMBEDDING_TEXT_MODEL={recipe.name}\n"
        note = "Add to Karakeep's .env beside its text model's lines, then `docker compose up -d`."
    else:
        ctx = context_tokens(recipe)
        out = min(output_tokens(recipe), 2048)
        image = (
            f"INFERENCE_IMAGE_MODEL={recipe.name}\n"
            if recipe.modality is Modality.VISION_CHAT
            else ""
        )
        lines = (
            f"{key}\nOPENAI_BASE_URL={base}\nINFERENCE_TEXT_MODEL={recipe.name}\n{image}"
            f"INFERENCE_CONTEXT_LENGTH={max(ctx - out, 0)}\nINFERENCE_MAX_OUTPUT_TOKENS={out}\n"
            "INFERENCE_JOB_TIMEOUT_SEC=180\n"
        )
        note = (
            "Add to Karakeep's .env (it runs in Docker), then `docker compose up -d`. Its "
            "context is the instance's less the reply; the 180 s timeout leaves room for a "
            "parked model to load (Karakeep's default is 30 s)."
            + (
                ""
                if recipe.modality is Modality.VISION_CHAT
                else " Pictures need a vision model as INFERENCE_IMAGE_MODEL."
            )
        )
    return [HarnessUrl(label="Karakeep: .env", url=base, note=note, snippet=lines)]


def _frigate(recipe: Recipe) -> list[HarnessUrl]:
    base = shared_base_url(ODYSSEUS_HOST)
    key = '"{FRIGATE_MNM_API_KEY}"' if key_set() else "local"
    config = (
        "genai:\n"
        "  mnm:\n"
        "    provider: openai\n"
        f"    base_url: {base}\n"
        f"    api_key: {key}\n"
        f"    model: {recipe.name}\n"
        "    provider_options:\n"
        f"      context_size: {context_tokens(recipe)}\n"
    )
    secret = " Put the key in Frigate's environment as FRIGATE_MNM_API_KEY." if key_set() else ""
    return [
        HarnessUrl(
            label="Frigate: config.yml (GenAI)",
            url=base,
            note="Merge into Frigate's config.yml and restart it. Frigate 0.16 ignored "
            "`base_url` for the OpenAI provider: there, set OPENAI_BASE_URL to the same URL "
            f"under `environment_vars:` instead.{secret}",
            snippet=config,
        )
    ]


def _n8n(recipe: Recipe) -> list[HarnessUrl]:
    url = _ollama_url(ODYSSEUS_HOST)
    fields = (
        "Credentials > Add credential > Ollama\n"
        f"  Base URL: {url}\n"
        f"  {_key_line('API Key')}\n"
        f"Then in an Ollama Chat Model node, Model: {recipe.name}\n"
    )
    return [
        HarnessUrl(
            label="n8n: the Ollama credential",
            url=url,
            note="n8n in Docker on this machine; n8n installed here without Docker: "
            f"{_ollama_url('127.0.0.1')}. n8n's OpenAI credential has no documented base URL, "
            "so the Ollama credential is the way in.",
            snippet=fields,
        )
    ]


def _jetbrains(recipe: Recipe) -> list[HarnessUrl]:
    base = shared_base_url()
    tools = not isinstance(recipe.args, VllmArgs) or recipe.args.tool_parser is not None
    fields = (
        "Settings | Tools | AI Assistant | Providers & API keys > Third-party AI providers\n"
        "  Provider:     OpenAI-compatible\n"
        f"  URL:          {base}\n"
        f"  {_key_line('API Key')}\n"
        f"  Tool calling: {'on' if tools else 'off (vLLM has no tool parser for it)'}\n"
        "  Test Connection, then Apply\n"
        f"Models Assignment: pick {recipe.name}; Context window: {context_tokens(recipe)}\n"
    )
    return [
        HarnessUrl(
            label="JetBrains AI Assistant",
            url=base,
            note="Every chat model mnm runs appears in the AI Chat model list. Models from an "
            "OpenAI-compatible provider serve AI Chat until you assign them to other features.",
            snippet=fields,
        )
    ]


def _raycast(recipe: Recipe) -> list[HarnessUrl]:
    tools = not isinstance(recipe.args, VllmArgs) or recipe.args.tool_parser is not None
    vision = recipe.modality is Modality.VISION_CHAT
    keys = "    api_keys:\n      mnm: <your MNM_API_KEY>\n" if key_set() else ""
    block = (
        "providers:\n"
        "  - id: mnm\n"
        "    name: makenomistakes\n"
        f"    base_url: {shared_base_url()}\n"
        f"{keys}"
        "    models:\n"
        f"      - id: {recipe.name}\n"
        f'        name: "{recipe.name} (local)"\n'
        f"        context: {context_tokens(recipe)}\n"
        "        abilities:\n"
        "          temperature:\n            supported: true\n"
        f"          vision:\n            supported: {str(vision).lower()}\n"
        "          system_message:\n            supported: true\n"
        f"          tools:\n            supported: {str(tools).lower()}\n"
    )
    return [
        HarnessUrl(
            label="Raycast: ~/.config/raycast/ai/providers.yaml",
            url=shared_base_url(),
            note="Settings → AI → Reveal Providers Config; add the model to an `mnm` provider "
            "already there rather than a second one. Raycast reads the file as it changes.",
            snippet=block,
        )
    ]


def _obsidian(recipe: Recipe) -> list[HarnessUrl]:
    base = shared_base_url()
    fields = (
        "Settings → Copilot → BYOK → Add a provider → Add a custom provider\n"
        f"  Base URL: {base}\n"
        f"  {_key_line('API key')}\n"
        f"  Model ID: {recipe.name}\n"
        "  Test, then Save\n"
    )
    return [
        HarnessUrl(
            label="Obsidian Copilot",
            url=base,
            note="Earlier versions call it 3rd party (openai-format) under Custom Model. If Test "
            "passes but a chat never answers, turn on Enable CORS (replies then come whole). "
            "A router without a key refuses what Obsidian's window sends itself (its origin, "
            "app://obsidian.md, is another site to the router): Enable CORS, or the key.",
            snippet=fields,
        )
    ]


def _brave_leo(recipe: Recipe) -> list[HarnessUrl]:
    endpoint = f"{shared_base_url()}/chat/completions"
    fields = (
        "Settings → Leo → Bring your own model → Add new model\n"
        f"  Label:              {recipe.name} (local)\n"
        f"  Model request name: {recipe.name}\n"
        f"  Server endpoint:    {endpoint}\n"
        f"  {_key_line('API Key')}\n"
    )
    return [
        HarnessUrl(
            label="Brave Leo (bring your own model)",
            url=endpoint,
            note="Leo takes the whole chat completions URL, not a base URL. It sends text only.",
            snippet=fields,
        )
    ]


EMBED = frozenset({Modality.EMBEDDING})
VISION = frozenset({Modality.VISION_CHAT})
CHAT = frozenset(CHAT_LIKE)

APPS: dict[Consumer, App] = {
    Consumer.HOME_ASSISTANT: App(
        8_192,
        "its default window; the entities it exposes and the Assist tools fill it; not "
        "measured here",
        CHAT,
        "a chat model",
        _home_assistant,
    ),
    Consumer.KARAKEEP: App(
        4_096,
        "a bookmark's text up to INFERENCE_CONTEXT_LENGTH and the tagging prompt",
        CHAT | EMBED,
        "a chat model (an embedding one for EMBEDDING_TEXT_MODEL)",
        _karakeep,
    ),
    Consumer.FRIGATE: App(
        8_192,
        "camera frames and the review prompt; not measured here",
        VISION,
        "a vision model",
        _frigate,
    ),
    Consumer.N8N: App(
        4_096, "whatever the workflow sends; not measured here", CHAT, "a chat model", _n8n
    ),
    Consumer.JETBRAINS: App(
        16_384,
        "the open files and its chat instructions; not measured here",
        CHAT,
        "a chat model",
        _jetbrains,
    ),
    Consumer.RAYCAST: App(
        8_192,
        "a selection or a page and its instructions; not measured here",
        CHAT,
        "a chat model",
        _raycast,
    ),
    Consumer.OBSIDIAN: App(
        8_192,
        "the note in view and its chat history; not measured here",
        CHAT,
        "a chat model",
        _obsidian,
    ),
    Consumer.BRAVE_LEO: App(
        8_192,
        "the page it summarises; not measured here",
        CHAT,
        "a chat model",
        _brave_leo,
    ),
}
