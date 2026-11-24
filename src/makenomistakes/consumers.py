"""Who pastes a card and what a card is: shared by the chat-client cards (`harness`) and the
coding-agent cards (`harness_agents`), so neither imports the other's module."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from makenomistakes.client_view import DOCKER_HOST, current
from makenomistakes.config import Settings
from makenomistakes.hub.facts import CHAT_LIKE
from makenomistakes.model_meta import serving_context
from makenomistakes.recipes.model import Recipe
from makenomistakes.routing import CHAT, UPSTREAM

ODYSSEUS_HOST = DOCKER_HOST
LOCAL_HOST = "localhost"


class Consumer(StrEnum):
    """Who is pasting: decides the host (Docker bridge vs localhost) and the instructions."""

    ODYSSEUS = "odysseus"
    OPEN_WEBUI = "open-webui"
    CONTINUE = "continue"
    OPENAI_SDK = "openai-sdk"
    CURL = "curl"
    # coding agents and editors (harness_agents)
    CLAUDE_CODE = "claude-code"
    CODEX = "codex"
    OPENCODE = "opencode"
    QWEN_CODE = "qwen-code"
    AIDER = "aider"
    CLINE = "cline"
    ZED = "zed"
    LLAMA_VSCODE = "llama-vscode"
    # other apps (harness_apps)
    HOME_ASSISTANT = "home-assistant"
    KARAKEEP = "karakeep"
    FRIGATE = "frigate"
    N8N = "n8n"
    JETBRAINS = "jetbrains"
    RAYCAST = "raycast"
    OBSIDIAN = "obsidian-copilot"
    BRAVE_LEO = "brave-leo"


CONSUMER_LABELS: dict[Consumer, str] = {
    Consumer.ODYSSEUS: "Odysseus",
    Consumer.OPEN_WEBUI: "Open WebUI",
    Consumer.CONTINUE: "Continue (VS Code)",
    Consumer.OPENAI_SDK: "OpenAI Python SDK",
    Consumer.CURL: "curl",
    Consumer.CLAUDE_CODE: "Claude Code",
    Consumer.CODEX: "Codex CLI",
    Consumer.OPENCODE: "OpenCode",
    Consumer.QWEN_CODE: "Qwen Code",
    Consumer.AIDER: "Aider",
    Consumer.CLINE: "Cline",
    Consumer.ZED: "Zed",
    Consumer.LLAMA_VSCODE: "llama.vscode",
    Consumer.HOME_ASSISTANT: "Home Assistant",
    Consumer.KARAKEEP: "Karakeep",
    Consumer.FRIGATE: "Frigate",
    Consumer.N8N: "n8n",
    Consumer.JETBRAINS: "JetBrains AI Assistant",
    Consumer.RAYCAST: "Raycast",
    Consumer.OBSIDIAN: "Obsidian Copilot",
    Consumer.BRAVE_LEO: "Brave Leo",
}


class HarnessUrl(BaseModel):
    label: str
    url: str
    note: str
    opens_in_browser: bool = False  # a plain GET page; rendered as a link, not a copy field
    snippet: str | None = None  # a config block or command to copy whole, shown under the URL


KEY_HINT = (
    " This router asks for your MNM_API_KEY (the one in ~/.config/makenomistakes/secrets.env). "
    "A snippet that reads `$MNM_API_KEY` or the environment needs it exported in that shell "
    "first (`export MNM_API_KEY=…`; nothing loads secrets.env into a shell); elsewhere, paste "
    "it where the client asks for an API key."
)


def key_set() -> bool:
    """Whether the router's service asks for `MNM_API_KEY`: the secrets file's, which
    is what that service reads. Cards say so and never print it."""
    return Settings().api_key_value() is not None


def curl_key() -> str:
    """The header a curl line sends when the router asks for its key, from the shell's
    `$MNM_API_KEY` (export it first); nothing when no key is set."""
    return ' -H "Authorization: Bearer $MNM_API_KEY"' if key_set() else ""


def with_key_hint(cards: list[HarnessUrl]) -> list[HarnessUrl]:
    """The cards, the first one saying the router wants the key when one is set."""
    if not cards or not key_set():
        return cards
    first = cards[0].model_copy(update={"note": cards[0].note + KEY_HINT})
    return [first, *cards[1:]]


def router_origin(host: str = LOCAL_HOST) -> str:
    """The router: the one port every client reaches every instance through."""
    return origin(host, Settings().router_port)


def origin(host: str, port: int) -> str:
    """One of the router's doors at a host: the manifest names each. Inside
    `client_view.seen_from` (`mnm url --from`, a client install) as that place sees it."""
    seen = current()
    return seen.origin(host, port) if seen is not None else f"http://{host}:{port}"


def ollama_origin(host: str = LOCAL_HOST) -> str | None:
    """The router's Ollama listener (`MNM_OLLAMA_PORT`), or None when it has none (the
    router's own port serves `/api` too: `router_origin`)."""
    port = Settings().ollama_port
    return None if port is None else origin(host, port)


def web_origin(port: int, web_url: str = "") -> str:
    """mnm's web UI for the links a run's tools hand back: `MNM_WEB_URL` when the
    user set where another machine opens it, else this machine's localhost."""
    return web_url or f"http://{LOCAL_HOST}:{port}"


def copy_url(recipe: Recipe) -> str:
    """The one URL to copy for an instance, from this machine: the shared one for a chat-like
    model (its name is the model; the "Windows or WSL apps" card), its own
    `/upstream/<name>/v1` for the rest."""
    return shared_base_url() if recipe.modality in CHAT_LIKE else local_base_url(recipe)


def shared_base_url(host: str = LOCAL_HOST, chat_only: bool = False) -> str:
    """One base URL for every instance, the instance name as the model: `/chat/v1` lists and
    routes only the chat, coding and vision ones (an Odysseus LLM endpoint)."""
    return f"{router_origin(host)}{CHAT if chat_only else ''}/v1"


def instance_origin(recipe: Recipe, host: str = LOCAL_HOST) -> str:
    """One instance's own root on the router, without `/v1`: for a client that needs a URL of
    its own (an Odysseus image endpoint, an agent, an editor). With `own_origin`, the one
    place a client URL is built."""
    return f"{router_origin(host)}{UPSTREAM}{recipe.name}"


def own_origin(recipe: Recipe) -> str:
    """Where mnm's own client requests go (`mnm try`): the door clients use, on this machine,
    so a try goes through the same rewrites a client's request does. `127.0.0.1`, not
    `localhost`: that can resolve to `::1`, where nothing listens (the router binds IPv4)."""
    return f"{router_origin('127.0.0.1')}{UPSTREAM}{recipe.name}"


BROWSER_KEY_NOTE = (
    " A browser cannot send the router's key, so this opens the engine's own port on this "
    "machine: start the instance first (`mnm start`), since nothing loads it on demand here."
)


def browser_origin(recipe: Recipe) -> str:
    """Where a browser opens the instance's own pages: through the router, or, with a key set
    (which a browser cannot send), the engine's loopback port, reachable from this machine's
    browser only (WSL forwards Windows' localhost to it)."""
    return engine_origin(recipe) if key_set() else instance_origin(recipe)


def browser_note() -> str:
    return BROWSER_KEY_NOTE if key_set() else ""


def engine_origin(recipe: Recipe) -> str:
    """The engine itself, on this machine only (`mnm url --direct`): no router in between, so
    no loading on demand and none of its rewrites. For debugging on the box."""
    return f"http://127.0.0.1:{recipe.port}"


def local_base_url(recipe: Recipe) -> str:
    """What Windows and WSL apps paste."""
    return f"{instance_origin(recipe)}/v1"


def docker_base_url(recipe: Recipe) -> str:
    """What a client in Docker pastes (Odysseus, Open WebUI): WSL through the bridge host."""
    return f"{instance_origin(recipe, ODYSSEUS_HOST)}/v1"


def context_tokens(recipe: Recipe) -> int:
    """The context one request gets (`model_meta.serving_context`), 0 for an engine without
    one."""
    return serving_context(recipe) or 0


def output_tokens(recipe: Recipe) -> int:
    """A reply budget that leaves a client room for its prompt: a quarter of the context, at
    most 8,192 (a longer reply rarely helps a coding turn and halves what fits)."""
    return min(8_192, context_tokens(recipe) // 4)


def provider_id(recipe: Recipe) -> str:
    """The name each client files this instance under: one per instance, never a vendor's."""
    return f"mnm-{recipe.name}"
