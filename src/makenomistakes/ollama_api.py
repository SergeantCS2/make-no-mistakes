"""Ollama's API on the router: what `/api/tags`, `/api/show` and `/api/ps` list, and
an Ollama chat, generate or embed request as the OpenAI request the instance's engine takes.
Pure functions over typed requests; `ollama_http` serves them, `ollama_stream` turns the
engine's answers back into Ollama's.

The shapes are Ollama's own documentation (github.com/ollama/ollama, docs/api.md, read
2026-10-22). Clients that only speak
Ollama reach mnm's models this way: Home Assistant's Ollama integration, n8n's Ollama
credential, Enchanted, Reins, and Odysseus's Cookbook scan (`routes/cookbook_helpers.py:559`
in 0.20.19 asks `127.0.0.1:11434/api/tags` and `host.docker.internal:11434/api/tags`).

- A model's `name` is the instance name; `name:latest` names it too (clients add the tag).
- Only what the API can serve is listed: chat, coding and vision instances (`/api/chat`,
  `/api/generate`) and embedding ones (`/api/embed`); pictures, clips and voices are not.
- `options.num_ctx` is never applied (the instance's context is its recipe's: `mnm set`);
  the options llama-server and vLLM take (temperature, top_p, top_k, min_p, seed, stop,
  num_predict as max_tokens, the penalties) pass on, the rest are named in the journal
  (`ollama_http.left_out`), `num_ctx` only when it asks for more than the instance serves.
- `keep_alive`: 0 parks the instance once the reply is done (never a pinned one or a member
  of a set in use: `Router.park`), a negative one switches its idle unload off, a duration
  (whole seconds, rounded up) sets its idle limit; held by the router, never saved
  (`IdleClock.keep`).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from makenomistakes import __version__
from makenomistakes.engine_answers import EmbeddingAnswer
from makenomistakes.engine_args_vllm import VllmArgs
from makenomistakes.hub.facts import CHAT_LIKE, Modality
from makenomistakes.instance_types import InstanceStatus
from makenomistakes.model_meta import FORMATS, ModelMeta, serving_context
from makenomistakes.recipes import Recipe

SERVED = CHAT_LIKE | {Modality.EMBEDDING}
LATEST = ":latest"
# Ollama options the engines take under the same name in an OpenAI request (llama-server and
# vLLM both accept the sampling extras), and the one they take under another.
SAME_OPTIONS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "repeat_penalty",
)
RENAMED_OPTIONS = {"num_predict": "max_tokens"}
NEVER = "9999-12-31T23:59:59Z"  # /api/ps's expiry of a model that is never unloaded for idleness


class _Loose(BaseModel):
    model_config = ConfigDict(extra="ignore")


class OllamaFunction(_Loose):
    name: str = ""
    arguments: dict[str, Any] | str = Field(default_factory=dict)  # Ollama's is an object


class OllamaToolCall(_Loose):
    function: OllamaFunction = OllamaFunction()


class OllamaMessage(_Loose):
    role: str = "user"
    content: str = ""
    images: list[str] | None = None
    thinking: str | None = None
    tool_calls: list[OllamaToolCall] | None = None  # an assistant's, sent back with the result
    tool_name: str | None = None


class OllamaChat(_Loose):
    model: str
    messages: list[OllamaMessage] = Field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    format: str | dict[str, Any] | None = None
    options: dict[str, Any] | None = None
    stream: bool = True  # Ollama streams unless told not to
    keep_alive: str | int | float | None = None
    think: bool | str | None = None


class OllamaGenerate(_Loose):
    model: str
    prompt: str = ""
    system: str | None = None
    images: list[str] | None = None
    suffix: str | None = None
    format: str | dict[str, Any] | None = None
    options: dict[str, Any] | None = None
    stream: bool = True
    raw: bool = False
    keep_alive: str | int | float | None = None
    think: bool | str | None = None


class OllamaEmbed(_Loose):
    model: str
    input: str | list[str] = Field(default_factory=list)
    options: dict[str, Any] | None = None
    keep_alive: str | int | float | None = None
    dimensions: int | None = None


class OllamaShow(_Loose):
    model: str = ""
    name: str = ""  # the older spelling, which clients still send


class KeepAliveError(ValueError):
    pass


_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h)")
_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def keep_alive_seconds(value: str | int | float | None) -> int | None:
    """Ollama's `keep_alive` in whole seconds (None: not sent; negative: forever). A number
    is seconds; a string is a number or a Go duration ("5m", "1h30m", "-1m")."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise KeepAliveError(f"keep_alive {value!r} is not a duration")
    if isinstance(value, (int, float)):
        return _seconds(value)
    text = value.strip()
    try:
        number = float(text)
    except ValueError:
        pass
    else:
        return _seconds(number)
    negative = text.startswith("-")
    body = text.lstrip("+-")
    if not body or _DURATION.sub("", body):
        raise KeepAliveError(
            f'keep_alive {value!r} is not a duration: use seconds, "5m", "1h" or -1'
        )
    seconds = sum(float(n) * _UNITS[unit] for n, unit in _DURATION.findall(body))
    return -1 if negative else _seconds(seconds)


def _seconds(value: float) -> int:
    """Negative is forever (-1); 0 is 0; anything else whole seconds rounded up, so a
    duration under a second ("300ms", 0.5) never reads as 0, which unloads. Infinity and
    NaN are refused in words (a 400, never the router's 500): -1 is forever."""
    if not math.isfinite(value):
        raise KeepAliveError(f"keep_alive {value!r} is not a duration: -1 means forever")
    if value < 0:
        return -1
    return math.ceil(value)


def wanted_name(model: str) -> str:
    """The instance a request names: `tiny`, or `tiny:latest` as clients write it."""
    return model.removesuffix(LATEST)


def serves(recipe: Recipe) -> bool:
    return recipe.modality in SERVED


def capabilities(recipe: Recipe) -> list[str]:
    """What `/api/show` says the model does: chat-like ones complete (and call tools where
    the engine parses them), vision ones see, embedding ones embed."""
    if recipe.modality is Modality.EMBEDDING:
        return ["embedding"]
    found = ["completion"]
    args = recipe.args
    if not isinstance(args, VllmArgs) or args.tool_parser is not None:
        found.append("tools")
    if recipe.modality is Modality.VISION_CHAT:
        found.append("vision")
    return found


def digest(recipe: Recipe) -> str:
    """A stable id per instance (Ollama's is its manifest's sha256; mnm has no manifest)."""
    return hashlib.sha256(recipe.name.encode()).hexdigest()


def details(recipe: Recipe, meta: ModelMeta) -> dict[str, object]:
    family = meta.architecture or ""
    return {
        "parent_model": "",
        "format": FORMATS.get(recipe.engine, ""),
        "family": family,
        "families": [family] if family else [],
        "parameter_size": meta.size_label or "",
        "quantization_level": recipe.quant,
    }


def tag(status: InstanceStatus, meta: ModelMeta) -> dict[str, object]:
    """One `/api/tags` entry."""
    recipe = status.recipe
    return {
        "name": recipe.name,
        "model": recipe.name,
        "modified_at": recipe.created_at.isoformat(),
        "size": meta.size_bytes,
        "digest": digest(recipe),
        "details": details(recipe, meta),
    }


def running(status: InstanceStatus, meta: ModelMeta, expires_at: str) -> dict[str, object]:
    """One `/api/ps` entry: a loaded instance, the VRAM its plan puts on the card."""
    found = tag(status, meta)
    del found["modified_at"]
    return {**found, "expires_at": expires_at, "size_vram": status.recipe.vram_estimate_bytes or 0}


def show(status: InstanceStatus, meta: ModelMeta) -> dict[str, object]:
    """`/api/show`: the details, what the model can do, and `model_info` with Ollama's keys
    for what the files say: `general.architecture`, `general.parameter_count`,
    `<arch>.embedding_length`, and `<arch>.context_length`, which here is the context one
    request gets (the instance's own, as `parameters` says), not the trained one
    (`/v1/models`' `meta.n_ctx_train`). With no architecture known (no file read), only what
    is not keyed by one."""
    recipe = status.recipe
    ctx = serving_context(recipe)
    arch = meta.architecture
    info: dict[str, object] = {}
    if arch:
        info["general.architecture"] = arch
    if meta.parameter_count:
        info["general.parameter_count"] = meta.parameter_count
    if arch and meta.embedding_length:
        info[f"{arch}.embedding_length"] = meta.embedding_length
    if arch and ctx is not None:
        info[f"{arch}.context_length"] = ctx
    return {
        "modelfile": f"# makenomistakes instance {recipe.name} ({recipe.repo_id}, {recipe.quant})",
        "parameters": f"num_ctx {ctx}" if ctx is not None else "",
        "template": "",
        "details": details(recipe, meta),
        "model_info": info,
        "capabilities": capabilities(recipe),
        "modified_at": recipe.created_at.isoformat(),
    }


def version() -> dict[str, str]:
    return {"version": __version__}


def now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


# --- requests: Ollama's in, the engine's OpenAI request out --------------------------------


def options(sent: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """The OpenAI fields Ollama's `options` become, and the option names left out."""
    taken: dict[str, Any] = {}
    left: list[str] = []
    for key, value in (sent or {}).items():
        if key in SAME_OPTIONS:
            taken[key] = value
        elif key in RENAMED_OPTIONS:
            taken[RENAMED_OPTIONS[key]] = value
        else:
            left.append(key)
    return taken, left


def response_format(sent: str | dict[str, Any] | None) -> dict[str, Any] | None:
    """`format: "json"` is JSON mode; a schema is a structured reply; anything else none
    (`format` is then among the options left out)."""
    if sent == "json":
        return {"type": "json_object"}
    if isinstance(sent, dict):
        return {"type": "json_schema", "json_schema": {"name": "reply", "schema": sent}}
    return None


def _image_url(data: str) -> dict[str, object]:
    """A base64 picture as an OpenAI image part, its type read from its first bytes."""
    kinds = {"/9j/": "jpeg", "iVBOR": "png", "R0lGOD": "gif", "UklGR": "webp"}
    kind = next((k for prefix, k in kinds.items() if data.startswith(prefix)), "png")
    return {"type": "image_url", "image_url": {"url": f"data:image/{kind};base64,{data}"}}


def _content(text: str, images: list[str] | None) -> object:
    if not images:
        return text
    return [{"type": "text", "text": text}, *(_image_url(i) for i in images)]


def messages(sent: list[OllamaMessage]) -> list[dict[str, object]]:
    """Ollama's messages as OpenAI's: pictures as image parts, an assistant's tool calls with
    ids and their arguments as JSON text, each tool result tied to the call it answers (by
    the tool's name, else in order: Ollama carries no ids)."""
    out: list[dict[str, object]] = []
    calls: list[tuple[str, str]] = []  # (id, tool name) of the last assistant's calls
    for message in sent:
        item: dict[str, object] = {
            "role": message.role,
            "content": _content(message.content, message.images),
        }
        if message.role == "assistant" and message.tool_calls:
            calls = []
            listed = []
            for index, call in enumerate(message.tool_calls):
                name, arguments = call.function.name, call.function.arguments
                ident = f"call_{len(out)}_{index}"
                calls.append((ident, name))
                listed.append(
                    {
                        "id": ident,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments
                            if isinstance(arguments, str)
                            else json.dumps(arguments),
                        },
                    }
                )
            item["tool_calls"] = listed
        if message.role == "assistant" and message.thinking:
            item["reasoning_content"] = message.thinking
        if message.role == "tool":
            match = next(
                (c for c in calls if c[1] == message.tool_name), calls[0] if calls else None
            )
            if match is not None:
                item["tool_call_id"] = match[0]
                calls.remove(match)
            if message.tool_name:
                item["name"] = message.tool_name
        out.append(item)
    return out


def _common(
    model: str, stream: bool, sent_format: str | dict[str, Any] | None, think: bool | str | None
) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model, "stream": stream}
    if stream:
        body["stream_options"] = {"include_usage": True}
    if (shape := response_format(sent_format)) is not None:
        body["response_format"] = shape
    if think is not None:  # a level ("low" …) is thinking on
        body["chat_template_kwargs"] = {"enable_thinking": think is not False}
    return body


def chat_body(request: OllamaChat, served: str) -> tuple[dict[str, Any], list[str]]:
    """The engine's `/v1/chat/completions` body, and the options left out."""
    body = _common(served, request.stream, request.format, request.think)
    body["messages"] = messages(request.messages)
    if request.tools:
        body["tools"] = request.tools
    taken, left = options(request.options)
    return {**body, **taken}, left + _format_left(request.format)


def _format_left(sent: str | dict[str, Any] | None) -> list[str]:
    """`format` when it is neither "json" nor a schema: the engines get none."""
    return ["format"] if sent is not None and response_format(sent) is None else []


def generate_body(request: OllamaGenerate, served: str) -> tuple[dict[str, Any], list[str]]:
    """`/api/generate` as one chat turn: the system prompt, then the prompt (and pictures)."""
    body = _common(served, request.stream, request.format, request.think)
    turns: list[dict[str, object]] = []
    if request.system:
        turns.append({"role": "system", "content": request.system})
    turns.append({"role": "user", "content": _content(request.prompt, request.images)})
    body["messages"] = turns
    taken, left = options(request.options)
    left += [name for name, sent in (("suffix", request.suffix), ("raw", request.raw)) if sent]
    left += _format_left(request.format)
    return {**body, **taken}, left


def embed_body(request: OllamaEmbed, served: str) -> dict[str, Any]:
    body: dict[str, Any] = {"model": served, "input": request.input}
    if request.dimensions is not None:
        body["dimensions"] = request.dimensions
    return body


def embed_answer(
    model: str, answer: EmbeddingAnswer, took_ns: int, load_ns: int
) -> dict[str, object]:
    """The engine's `/v1/embeddings` answer (read by `engine_answers`) as Ollama's
    `/api/embed` one, in input order."""
    data = sorted(answer.data, key=lambda d: d.index)
    return {
        "model": model,
        "embeddings": [d.embedding for d in data],
        "total_duration": took_ns,
        "load_duration": load_ns,
        "prompt_eval_count": answer.usage.prompt_tokens if answer.usage else 0,
    }
