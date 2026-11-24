"""Facts read straight off a Hub model's metadata: its architecture, parameter count, base
model, whether its architecture is a vision one, and the card's example negative prompt."""

from __future__ import annotations

from makenomistakes.hub.client import HubModel

_VISION_ARCH_TOKENS = (
    "vl",
    "vision",
    "llava",
    "paligemma",
    "gemma3forconditional",
    "idefics",
    "mllama",
    "pixtral",
    "internvl",
    "minicpmv",
    "kimi_vl",
)


# Architectures run text only whatever their repo ships: DeepSeek-V4.1's
# vision projector is a setting a box adds once it measures a gain, never part of a plan.
TEXT_ONLY_ARCHITECTURES = frozenset({"deepseek41", "deepseek_v41"})


# A transformers model type and the GGUF architecture llama.cpp names it, where they differ
# and a merge is still recent (`engine_support.MERGES` names the merge).
GGUF_ARCHITECTURES = {"deepseek_v41": "deepseek41"}


def vision_architecture(info: HubModel) -> bool:
    archs = " ".join(info.config.architectures).lower()
    return any(token in archs for token in _VISION_ARCH_TOKENS)


def model_architecture(info: HubModel) -> str | None:
    if info.gguf.architecture:
        return info.gguf.architecture
    if info.config.model_type:
        return info.config.model_type
    return info.config.architectures[0] if info.config.architectures else None


def param_billions(info: HubModel) -> float | None:
    total = info.gguf.total or info.safetensors.total
    return round(total / 1e9, 3) if total else None


def card_base_model(info: HubModel) -> str | None:
    for tag in info.tags:
        if tag.startswith("base_model:") and tag.count(":") == 1:
            return tag.split(":", 1)[1]
    return info.card_data.base_model


def card_negative_prompt(info: HubModel) -> str:
    """The first example negative prompt on the card, as the publisher wrote it."""
    for example in info.card_data.widget:
        text = (example.parameters.negative_prompt or "").strip().rstrip(",").strip()
        if text:
            return text
    return ""
