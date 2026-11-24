"""Cards for coding agents and editors: where each one's config goes and what to
paste, filled in with the instance's URL, model id and context. Pure data, like `harness`.

Every client talks to the instance's own llama-server, which serves all four APIs they use
(checked on a CPU build of 2026-10-09's source with the flags a chat recipe emits):
- the OpenAI chat API with `tools`: OpenCode, Cline, Zed, Continue; Aider (edit formats);
- the OpenAI Responses API: Codex CLI;
- the Anthropic Messages API (`/v1/messages`, with `count_tokens`): Claude Code;
- fill-in-the-middle (`/infill`): llama.vscode and llama.vim.
A single-model server answers any model id, so a client's background model still lands here.
Each card's settings come from the client's own documentation (2026-10-14); Claude Code and
Codex CLI ran against a CPU server with them (Codex with `--strict-config`).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from makenomistakes.agent_profile import CLAUDE_CODE_FIRST, FLOORS, Floor, serves
from makenomistakes.consumers import (
    CONSUMER_LABELS,
    Consumer,
    HarnessUrl,
    context_tokens,
    instance_origin,
    key_set,
    local_base_url,
    output_tokens,
    provider_id,
)
from makenomistakes.engine_args import LlamaCppArgs
from makenomistakes.hardware import platform as platforms
from makenomistakes.harness_editors import cline_card, llama_vscode_card, zed_card
from makenomistakes.hub.facts import CHAT_LIKE, Engine
from makenomistakes.recipes.model import Recipe

AGENT_MODALITIES = CHAT_LIKE


@dataclass(frozen=True)
class _Client:
    build: Callable[[Recipe], list[HarnessUrl]]
    thinking: str | None  # what the client does with a thinking model's reasoning


# What a client needs that only llama-server provides (a vLLM instance says so).
LLAMA_SERVER_ONLY: dict[Consumer, str] = {
    Consumer.CLAUDE_CODE: "the Anthropic Messages API (/v1/messages)",
    Consumer.LLAMA_VSCODE: "fill-in-the-middle completion (/infill)",
}


# What each engine does with the Codex tool types it does not run.
CODEX_TOOL_NOTES = {
    Engine.LLAMACPP: "llama-server skips the tool types it does not run (web search, MCP "
    "namespaces) and says so in its log; Codex's own shell and file tools work.",
    Engine.VLLM: "vLLM serves Codex's Responses API too; how it treats the tool types it does "
    "not run (web search, MCP namespaces) is not checked here yet.",
}


def agent_cards(recipe: Recipe, consumer: Consumer) -> list[HarnessUrl]:
    """One client's cards, or why this instance cannot serve it. The context check goes on
    the first card."""
    client = AGENT_CLIENTS[consumer]
    label = CONSUMER_LABELS[consumer]
    if recipe.modality not in AGENT_MODALITIES:
        return [
            HarnessUrl(
                label=label,
                url=local_base_url(recipe),
                note=f"{label} needs a chat model; {recipe.name} is "
                f"{recipe.modality.value}. Add one with `mnm add <link>` (for example "
                "unsloth/Qwen3-8B-GGUF) and open its card.",
            )
        ]
    if recipe.engine is Engine.VLLM and consumer in LLAMA_SERVER_ONLY:
        return [
            HarnessUrl(
                label=label,
                url=local_base_url(recipe),
                note=f"{label} needs {LLAMA_SERVER_ONLY[consumer]}, which llama-server "
                f"serves and vLLM, which runs {recipe.name}, does not. Use a GGUF build of "
                "the model (`mnm add <org>/<model>-GGUF`) for this client.",
            )
        ]
    first, *rest = client.build(recipe)
    floor = context_note(recipe, FLOORS[consumer.value])
    thinks = "" if client.thinking is None else f" {THINKING}{client.thinking}"
    note = f"{first.note} {floor}{thinks}"
    if recipe.engine is Engine.VLLM and recipe.vllm_args.tool_parser is None:
        note += (
            f" vLLM has no tool-call parser for {recipe.served_name}'s family, so it answers "
            f"every request that offers tools with an error: {label} cannot use its tools here. "
            "A GGUF build on llama.cpp (`mnm add <org>/<model>-GGUF`) can."
        )
    checked = first.model_copy(update={"note": note})
    return [checked, *rest]


def floor_line(recipe: Recipe, floor: Floor) -> str:
    """`agent_profile.serves` for this instance: what one request gets, and from what."""
    args = recipe.args
    how = (
        f"ctx {args.ctx:,} ÷ parallel {args.parallel}"
        if isinstance(args, LlamaCppArgs)
        else "max_model_len"
    )
    return serves(floor, context_tokens(recipe), how)


def context_note(recipe: Recipe, floor: Floor) -> str:
    """A card's context line: the floor line, what to do when it is short, and (llama.cpp's
    `parallel`) how several sessions at once are served."""
    said = floor_line(recipe, floor)
    args = recipe.args
    if not isinstance(args, LlamaCppArgs):  # vLLM: one request's whole window, no slots
        return said
    have, slots = context_tokens(recipe), args.parallel
    if have < floor.tokens:
        shared = f" ({slots} requests at once share it)" if slots > 1 else ""
        said += (
            f" `mnm set {recipe.name} ctx={floor.tokens * slots}` raises it{shared}; "
            "`mnm ls` then says whether it still fits (Fits now)."
        )
    if slots > 1:
        said += f" parallel {slots} serves {slots} requests at once, each with ctx ÷ {slots}."
    each = max(have, floor.tokens)
    return said + (
        f" Several sessions at once: `mnm set {recipe.name} parallel={slots + 1} "
        f"ctx={each * (slots + 1)}` (one copy of the weights; each session gets {each:,}), or "
        "a second instance beside it in your set (`mnm sets edit <set> chat=<second>`: twice "
        "the weights, a model of its own)."
    )


THINKING = (  # once per card; GUIDE §11 has the rest
    "Thinking stays on (mnm turns it off only for the chat card's 16-token check); "
    "llama-server puts an earlier turn's reasoning back into the prompt when the client "
    "returns it. "
)


def _claude_code(recipe: Recipe) -> list[HarnessUrl]:
    """Claude Code's documented gateway settings (code.claude.com/docs/en/llm-gateway-connect,
    env-vars, model-config): the origin without `/v1`, a bearer token, every model alias and
    the subagents on the one served id, the context and reply sizes the server really has."""
    model = recipe.served_name
    origin = instance_origin(recipe)
    env = {
        "ANTHROPIC_BASE_URL": origin,
        "ANTHROPIC_AUTH_TOKEN": "$MNM_API_KEY" if key_set() else "local",  # the shell's
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,  # titles, summaries, background work
        "ANTHROPIC_DEFAULT_FABLE_MODEL": model,
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
        "CLAUDE_CODE_SUBAGENT_MODEL_FORCE": "1",
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(context_tokens(recipe)),
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(output_tokens(recipe)),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",  # no telemetry or update checks
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",  # beta fields a local server rejects
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",  # it would reach the model as prompt text
        "CLAUDE_CONFIG_DIR": f"$HOME/.claude-{recipe.name}",  # your own setup stays as it is
    }
    lines = ["env -u ANTHROPIC_API_KEY \\"]
    lines += [f'  {key}="{value}" \\' for key, value in env.items()]
    lines.append("  claude")
    return [
        HarnessUrl(
            label=f"Claude Code ({platforms.current().terminal})",
            url=origin,
            note=f"Paste in a terminal to start Claude Code on {model}. It keeps its settings "
            f"in ~/.claude-{recipe.name}, so your normal Claude Code and its login stay "
            "untouched; put the lines in an alias to keep them. It uses the Anthropic API, "
            "which llama-server serves at /v1/messages. A `[claude-code:unrecognized_model]` "
            "line naming the model is Claude Code not knowing a local model's name: harmless. "
            'Run with `-p "…"` (nobody there to answer), ask it to edit the files now and '
            "not to ask first, or it may end on a question. The context each request gets "
            f"({context_tokens(recipe):,} tokens) must hold its first request (about "
            f"{CLAUDE_CODE_FIRST:,} tokens) plus the project's CLAUDE.md, its memory and the "
            'files it reads: a large CLAUDE.md is the usual cause of "Prompt is too long" '
            "(start it in the project you mean, not a big checkout). Claude Code 2.1 needs "
            "Node 22 or newer.",
            snippet="\n".join(lines) + "\n",
        )
    ]


def _codex(recipe: Recipe) -> list[HarnessUrl]:
    """Codex reads providers only from the user's own config.toml and speaks only the
    Responses API (`wire_api = "responses"`; "chat" is gone). Its own 272k-token fallback
    window must be replaced by the server's."""
    provider = provider_id(recipe)
    base = local_base_url(recipe)
    block = (
        f"[model_providers.{provider}]\n"
        f'name = "makenomistakes: {recipe.name}"\n'
        f'base_url = "{base}"\n'
        'wire_api = "responses"\n'
        + ('env_key = "MNM_API_KEY"\n' if key_set() else "")  # sent as its Bearer token
    )
    ctx = context_tokens(recipe)
    launch = (
        f"codex -c model_provider={provider} -m {recipe.served_name} "
        f"-c model_context_window={ctx} -c model_auto_compact_token_limit={ctx * 4 // 5}\n"
    )
    return [
        HarnessUrl(
            label="Codex CLI: ~/.codex/config.toml",
            url=base,
            note="Add this block to ~/.codex/config.toml (Codex reads providers from that file "
            "only, not from a project's). It adds a provider and changes nothing else.",
            snippet=block,
        ),
        HarnessUrl(
            label="Codex CLI: start it on this instance",
            url=base,
            note="Your default model and provider stay as they are. "
            + CODEX_TOOL_NOTES.get(recipe.engine, CODEX_TOOL_NOTES[Engine.LLAMACPP]),
            snippet=launch,
        ),
    ]


def _opencode(recipe: Recipe) -> list[HarnessUrl]:
    """OpenCode's `@ai-sdk/openai-compatible` provider with per-model limits (opencode.ai/
    docs/providers, its llama.cpp section)."""
    provider = provider_id(recipe)
    model = recipe.served_name
    config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            provider: {
                "npm": "@ai-sdk/openai-compatible",
                "name": f"makenomistakes: {recipe.name}",
                "options": {
                    "baseURL": local_base_url(recipe),
                    **({"apiKey": "{env:MNM_API_KEY}"} if key_set() else {}),
                },
                "models": {
                    model: {
                        "name": f"{model} (local)",
                        "tool_call": True,
                        "limit": {
                            "context": context_tokens(recipe),
                            "output": output_tokens(recipe),
                        },
                    }
                },
            }
        },
    }
    return [
        HarnessUrl(
            label="OpenCode: ~/.config/opencode/opencode.json",
            url=local_base_url(recipe),
            note=f"Paste as the file, or add the `{provider}` entry to its `provider` "
            f"object. Then `opencode --model {provider}/{model}`; your default model stays.",
            snippet=_json(config),
        )
    ]


def _qwen_code(recipe: Recipe) -> list[HarnessUrl]:
    """Qwen Code's OpenAI-compatible provider (github.com/QwenLM/qwen-code README; its docs'
    Authentication and Model Providers pages, 2026-10-23): a `modelProviders.openai` entry
    with the window it may fill (`generationConfig.contextWindowSize`: without it a model id
    with "qwen3" in it is taken for a 262k or 1M window), started with `--auth-type openai`
    and `--model`, the key from the environment variable the entry names."""
    model = recipe.served_name
    base = local_base_url(recipe)
    settings = {
        "modelProviders": {
            "openai": [
                {
                    "id": model,
                    "name": f"makenomistakes: {recipe.name}",
                    "baseUrl": base,
                    "envKey": "OPENAI_API_KEY",
                    "generationConfig": {"contextWindowSize": context_tokens(recipe)},
                }
            ]
        }
    }
    key = 'OPENAI_API_KEY="$MNM_API_KEY"' if key_set() else "OPENAI_API_KEY=local"
    return [
        HarnessUrl(
            label="Qwen Code: ~/.qwen/settings.json",
            url=base,
            note="Add the entry to the file's `modelProviders.openai` list (or paste as the file). "
            "It adds a model and changes nothing else: your default model and login stay.",
            snippet=_json(settings),
        ),
        HarnessUrl(
            label=f"Qwen Code ({platforms.current().terminal}, in your project)",
            url=base,
            note=f"Starts Qwen Code on {model}; `qwen` alone keeps your default.",
            snippet=f"{key} qwen --auth-type openai --model {model}\n",
        ),
    ]


def _aider(recipe: Recipe) -> list[HarnessUrl]:
    """Aider goes through LiteLLM: the `openai/` prefix, a key it insists on, and a metadata
    file so it knows the window (aider.chat/docs/llms/openai-compat, adv-model-settings)."""
    model = f"openai/{recipe.served_name}"
    base = local_base_url(recipe)
    metadata = {
        model: {
            "max_input_tokens": context_tokens(recipe),
            "max_output_tokens": output_tokens(recipe),
            "max_tokens": output_tokens(recipe),
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
            "litellm_provider": "openai",
            "mode": "chat",
        }
    }
    return [
        HarnessUrl(
            label=f"Aider ({platforms.current().terminal}, in your project)",
            url=base,
            note="Aider edits by text formats, not tool calls: `--edit-format diff` suits "
            "larger models, the default `whole` small ones.",
            snippet=(  # from the environment, so `ps` never shows the key
                'OPENAI_API_KEY="$MNM_API_KEY" ' if key_set() else "OPENAI_API_KEY=local "
            )
            + f"aider --model {model} --openai-api-base {base}\n",
        ),
        HarnessUrl(
            label="Aider: ~/.aider.model.metadata.json",
            url=base,
            note="Without it Aider warns it does not know the window and never trims to it. "
            "Add the entry to the file if you already have one.",
            snippet=_json(metadata),
        ),
    ]


def _json(value: object) -> str:
    return json.dumps(value, indent=2) + "\n"


# Each client's floor is `agent_profile.FLOORS[consumer.value]`; `thinking` says what the
# client was seen doing with reasoning_content (a capture in front of the router) or that it
# was not checked.
SEEN = "seen in a test against llama-server"
NOT_CHECKED = "Whether {client} sends reasoning_content back was not checked here."
AGENT_CLIENTS: dict[Consumer, _Client] = {
    Consumer.CLAUDE_CODE: _Client(
        _claude_code,
        f"Claude Code sends each turn's thinking block back ({SEEN}), which llama-server turns "
        "into reasoning_content, so the model keeps its earlier thinking.",
    ),
    Consumer.CODEX: _Client(_codex, NOT_CHECKED.format(client="Codex")),
    Consumer.OPENCODE: _Client(
        _opencode,
        f"OpenCode sends each turn's reasoning_content back with its tool calls ({SEEN}), so "
        "the model keeps its earlier thinking.",
    ),
    Consumer.QWEN_CODE: _Client(
        _qwen_code,
        f"Qwen Code's requests carry no thinking switch of their own ({SEEN}); its settings' "
        "generationConfig.extra_body passes one through (its docs). "
        + NOT_CHECKED.format(client="Qwen Code"),
    ),
    Consumer.AIDER: _Client(_aider, NOT_CHECKED.format(client="Aider")),
    Consumer.CLINE: _Client(cline_card, NOT_CHECKED.format(client="Cline")),
    Consumer.ZED: _Client(zed_card, NOT_CHECKED.format(client="Zed")),
    Consumer.LLAMA_VSCODE: _Client(llama_vscode_card, None),  # completions do not think
}
