"""Editor cards: Cline in VS Code, Zed, and llama.vscode's code completion. Their
settings live in a form or an editor's settings file, so a card names each field. The context
check is `harness_agents`'s, which lists these with the other clients."""

from __future__ import annotations

import json

from makenomistakes.consumers import (
    HarnessUrl,
    context_tokens,
    instance_origin,
    key_set,
    local_base_url,
    output_tokens,
    provider_id,
)
from makenomistakes.hub.facts import Engine
from makenomistakes.recipes.model import Recipe


def cline_card(recipe: Recipe) -> list[HarnessUrl]:
    """Cline's "OpenAI Compatible" provider (docs.cline.bot/provider-config/openai-compatible):
    a form, no file. Left unset, it assumes a 128,000-token window, and its auto-compaction
    follows the window it is told. Cline 4.x sends native OpenAI `tools`."""
    fields = (
        "Settings (gear) > API Provider: OpenAI Compatible\n"
        f"  Base URL:    {local_base_url(recipe)}\n"
        f"  API Key:     {'your MNM_API_KEY' if key_set() else 'local'}\n"
        f"  Model ID:    {recipe.served_name}\n"
        "  Model Configuration:\n"
        f"    Supports Images:     {'on' if recipe.projector else 'off'}\n"
        f"    Context Window Size: {context_tokens(recipe)}\n"
        f"    Max Output Tokens:   {output_tokens(recipe)}\n"
    )
    card = HarnessUrl(
        label="Cline (VS Code)",
        url=local_base_url(recipe),
        note="Fill in Cline's provider form as below. Roo Code was shut down in May 2026; "
        "Cline takes the same settings.",
        snippet=fields,
    )
    return [card]


def zed_card(recipe: Recipe) -> list[HarnessUrl]:
    """Zed's `openai_compatible` provider (zed.dev/docs/ai/use-api-access): `max_tokens` is
    the window, the key comes from `<PROVIDER>_API_KEY` or the keychain, never the file, and
    a missing key fails before any request."""
    provider = provider_id(recipe)
    settings = {
        "language_models": {
            "openai_compatible": {
                provider: {
                    "api_url": local_base_url(recipe),
                    "available_models": [
                        {
                            "name": recipe.served_name,
                            "display_name": f"{recipe.served_name} (local)",
                            "max_tokens": context_tokens(recipe),
                            "max_output_tokens": output_tokens(recipe),
                            "capabilities": {
                                "tools": True,
                                "images": recipe.projector is not None,
                                "parallel_tool_calls": False,
                                "prompt_cache_key": False,
                            },
                        }
                    ],
                }
            }
        }
    }
    key = provider.upper().replace("-", "_") + "_API_KEY"
    return [
        HarnessUrl(
            label="Zed: settings.json",
            url=local_base_url(recipe),
            note="Zed > Settings > Open Settings, then merge this into the file (keep your "
            "other `language_models` entries).",
            snippet=json.dumps(settings, indent=2) + "\n",
        ),
        HarnessUrl(
            label="Zed: the API key",
            url=local_base_url(recipe),
            note=(
                "The router asks for it"
                if key_set()
                else "Nothing checks it (no MNM_API_KEY is set), but Zed needs one"
            )
            + ": set it in Agent Settings > LLM Providers, or in the environment Zed starts "
            "from.",
            snippet=f'export {key}="$MNM_API_KEY"\n' if key_set() else f"export {key}=local\n",
        ),
        *_zed_llamacpp(recipe),
    ]


def _zed_llamacpp(recipe: Recipe) -> list[HarnessUrl]:
    """Zed's own llama.cpp provider (zed.dev/docs/ai/use-a-local-model, read 2026-10-22): the
    server's root (no `/v1`) as `api_url`, one per instance, its models by hand
    (`auto_discover: false`), the key from `LLAMACPP_API_KEY` or Agent Settings."""
    if recipe.engine is not Engine.LLAMACPP:
        return []  # vLLM is not llama-server: the OpenAI-compatible provider below serves it
    settings = {
        "language_models": {
            "llama.cpp": {
                "api_url": instance_origin(recipe),
                "auto_discover": False,
                "available_models": [
                    {
                        "name": recipe.served_name,
                        "display_name": f"{recipe.name} (local)",
                        "max_tokens": context_tokens(recipe),
                        "supports_tools": True,
                        "supports_images": recipe.projector is not None,
                    }
                ],
            }
        }
    }
    key = (
        'export LLAMACPP_API_KEY="$MNM_API_KEY"  # in the environment Zed starts from\n'
        if key_set()
        else ""
    )
    return [
        HarnessUrl(
            label="Zed: settings.json (its llama.cpp provider)",
            url=instance_origin(recipe),
            note="Or Zed's own provider for llama-server, instead of the block above: one "
            "`api_url` serves one instance, so this suits the instance you use most."
            + (" The key: the line under the block." if key_set() else ""),
            snippet=json.dumps(settings, indent=2) + "\n" + key,
        )
    ]


def llama_vscode_card(recipe: Recipe) -> list[HarnessUrl]:
    """llama.vscode builds `<endpoint>/infill` from the server's root, so no `/v1`
    (github.com/ggml-org/llama.vscode). /infill needs a model trained for fill-in-the-middle;
    a plain chat model answers "Infill is not supported by this model"."""
    origin = instance_origin(recipe)
    settings = {
        "llama-vscode.endpoint": origin,
        "llama-vscode.endpoint_chat": origin,
        **({"llama-vscode.api_key": "<your MNM_API_KEY>"} if key_set() else {}),
    }
    return [
        HarnessUrl(
            label="llama.vscode: settings.json (VS Code)",
            url=origin,
            note="Code completion needs a model trained for fill-in-the-middle, such as "
            "Qwen2.5-Coder: `mnm add ggml-org/Qwen2.5-Coder-1.5B-Q8_0-GGUF --name fim` is the "
            "size llama.vscode suggests below 8 GB. A chat model such as Qwen3 answers "
            "completions with an error. llama.vim takes the same server as "
            f"'endpoint_fim': '{origin}/infill'.",
            snippet=json.dumps(settings, indent=2) + "\n",
        )
    ]
