"""`mnm set <name> gpus=…` and `split=…`: which cards an instance runs on,
and how llama.cpp splits it across them. Both are settings only when `MNM_GPU` names
several cards, so a one-card box lists (and refuses) the settings it always did.

`split=tensor` is llama.cpp's `-sm tensor` (ggml-org/llama.cpp#19378, merged 2026-04-09;
llama.cpp's `docs/multi-gpu.md`): each layer's weights and KV split across the cards,
experimental; it needs flash attention on and an unquantised KV cache, and llama.cpp's own
fitter (`--fit`) does not work with it, so mnm allows it only where its plan put every
layer on the cards. Its gain depends on the link between the cards (published readings:
+60% decode on two 4090s at PCIe 4.0 x16, ~10x slower prompts on two 3090s over PCIe 2.0
x1): the user's call per instance.
"""

from __future__ import annotations

from collections.abc import Callable

from makenomistakes.controls_catalog import (
    ALL_MODALITIES,
    CHAT_MODALITIES,
    Control,
    Kind,
    check_controls,
)
from makenomistakes.engine_args import KvType, LlamaCppArgs
from makenomistakes.engine_args_vllm import VllmArgs
from makenomistakes.fit.estimate import FitLevel
from makenomistakes.hardware.cards import equal
from makenomistakes.hardware.gpu_select import chosen, parse_choice
from makenomistakes.hardware.probe import Gpu
from makenomistakes.hub.facts import Modality
from makenomistakes.recipes.cards import ONE_CARD_ENGINES, placed_here, recipe_cards
from makenomistakes.recipes.model import Recipe

Cards = Callable[[], list[Gpu]]  # the cards nvidia-smi lists now (the service's probe)

GPUS_CONTROL = Control(
    key="gpus",
    field="gpus",
    label="Cards",
    help=(
        "The cards this instance runs on, by nvidia-smi's index (0, or 0,1), among those "
        "MNM_GPU names. Empty: llama.cpp on every one of them (its layers split by each "
        "card's free VRAM), vLLM on all of them only when they are equal, and pictures, "
        "speech and voices on the first. Applies at the next start."
    ),
    kind=Kind.TEXT,
    nullable=True,
    recommended="empty; one card for a model that fits it, so the other stays free",
    modalities=frozenset(Modality),
    on_recipe=True,
)

SPLIT_CONTROL = Control(
    key="split",
    field="split_mode",
    label="Split across cards",
    help=(
        "layer (llama.cpp's default) gives each card whole layers, in turn for every token; "
        "tensor splits every layer across the cards at once (experimental: flash attention "
        "on, kv=f16, every layer on the cards), faster only over a fast link between them."
    ),
    kind=Kind.CHOICE,
    choices=("layer", "tensor"),
    nullable=True,
    recommended="layer; tensor only for cards on fast PCIe slots, read with `mnm try --speed`",
    modalities=ALL_MODALITIES,
)

EXPERTS_CONTROL = Control(
    key="experts",
    field="experts",
    label="Experts across cards",
    help=(
        "A mixture of experts that does not fit whole on the cards: auto fills each "
        "card's room with experts where that beats reading them from RAM; ram keeps every "
        "expert in RAM (the cards hold the blocks only); cards fills the room regardless. "
        "Applies at the next start."
    ),
    kind=Kind.CHOICE,
    choices=("auto", "ram", "cards"),
    nullable=True,
    recommended="auto (the plan's); `mnm tune` tries other amounts per card",
    modalities=CHAT_MODALITIES,
)

CARD_EXPERTS_CONTROL = Control(
    key="card-experts",
    field="placement",
    label="Expert blocks per card",
    help=(
        "How many blocks' experts each card holds, one number per card in the order "
        "`mnm show` lists them (the main card first): 3,4,3,3. Each card's own blocks first; "
        "the rest stay in RAM. More than the estimate fits reads won't fit; the tuner sets it."
    ),
    kind=Kind.TEXT,
    recommended="the plan's (empty keeps it); the tuner tries one or two more or fewer a card",
    modalities=CHAT_MODALITIES,
)

CARD_CONTROLS = (GPUS_CONTROL, SPLIT_CONTROL, EXPERTS_CONTROL, CARD_EXPERTS_CONTROL)
PLACEMENT_KEYS = frozenset({EXPERTS_CONTROL.key, CARD_EXPERTS_CONTROL.key})
check_controls(
    [
        (GPUS_CONTROL, Recipe),
        (SPLIT_CONTROL, LlamaCppArgs),
        (EXPERTS_CONTROL, LlamaCppArgs),
        (CARD_EXPERTS_CONTROL, LlamaCppArgs),
    ]
)

# llama.cpp's tensor split is not implemented for these architectures (`docs/multi-gpu.md`
# at d854d98: "not implemented for DeepSeek2"; deepseek41 builds on it and is listed neither
# way, so it is refused too)
NO_TENSOR_SPLIT = frozenset({"deepseek2", "deepseek41"})


class CardSettingError(ValueError):
    pass


def several_cards() -> bool:
    """`MNM_GPU` names several cards: only then are `gpus=` and `split=` settings."""
    try:
        return chosen().several
    except ValueError:  # MNM_GPU is unreadable: every reading says so already
        return False


def with_gpus(recipe: Recipe, raw: str, cards: Cards | None) -> Recipe:
    """The recipe pinned to the cards `raw` names (empty clears the pin), refused with the
    reason when the cards are not chosen, or the engine cannot use them."""
    text = raw.strip().lower()
    if text in ("", "auto", "none", "default"):
        return _fitted(recipe.model_copy(update={"gpus": None}))
    try:
        picked = parse_choice(text)
        selection = chosen()
    except ValueError:
        raise CardSettingError(
            f"gpus takes nvidia-smi's card numbers, one (0) or several (0,1), not '{raw}'"
        ) from None
    if picked.every:
        raise CardSettingError("gpus names cards by number; `all` is MNM_GPU's to say")
    wanted = picked.indices
    if not selection.every and not set(wanted) <= set(selection.indices):
        names = ",".join(str(i) for i in selection.indices)
        raise CardSettingError(
            f"MNM_GPU names {names}: card{'s' if len(wanted) > 1 else ''} "
            f"{','.join(str(i) for i in wanted)} must be among them (add a card there first)"
        )
    if len(wanted) > 1 and recipe.engine in ONE_CARD_ENGINES:
        raise CardSettingError(
            f"{recipe.engine.value} runs on one card: gpus={wanted[0]} (or another one)"
        )
    several = len(wanted) > 1 and cards is not None
    listed_now = {g.index: g for g in cards()} if several and cards is not None else {}
    makers = {listed_now[i].vendor for i in wanted if i in listed_now}
    if len(makers) > 1:  # An NVIDIA and an AMD card run different images
        raise CardSettingError(
            "an instance runs on one maker's cards: NVIDIA's and AMD's take different images "
            f"(gpus={','.join(str(i) for i in wanted)} names both); pin it to one maker's cards"
        )
    if len(wanted) > 1 and isinstance(recipe.args, VllmArgs):
        listed = listed_now
        found = [listed[i] for i in wanted if i in listed]
        if len(found) != len(wanted) or not equal(found):
            said = "; ".join(f"{g.index} is {g.name}" for g in found) or "nvidia-smi said none"
            raise CardSettingError(
                "vLLM's tensor parallel needs equal cards (the same model and VRAM: every "
                f"worker must hold the same KV cache): {said}. Unequal cards would need "
                "pipeline parallel, which mnm does not plan yet."
            )
    ordered = tuple(sorted(wanted)) if len(wanted) > 1 else wanted
    return _fitted(recipe.model_copy(update={"gpus": ordered}))


def _fitted(recipe: Recipe) -> Recipe:
    """The engine's split flags for the cards it now runs on: vLLM's tensor parallel across
    them; llama.cpp's planned `-ts` only while it names as many cards (else its own split
    by free memory, until `mnm replan`), none on one card."""
    cards = recipe_cards(recipe)
    count = len(cards) if cards is not None else 0
    args = recipe.args
    if isinstance(args, VllmArgs):
        if not recipe.gpus:  # the pin cleared: the plan's own choice stands
            return recipe
        return recipe.with_args(tensor_parallel=count if count > 1 else None)
    if isinstance(args, LlamaCppArgs):
        placement = args.placement
        if placement is not None and cards is not None and placed_here(recipe, cards) is None:
            # Placed for other cards: every expert in RAM until `mnm machine adopt`
            blocks = max(c.first + c.count for c in placement.cards)
            recipe = recipe.with_args(placement=None, experts=None, cpu_moe_layers=blocks)
        if count == 1:
            return recipe.with_args(tensor_split=None, main_gpu=None, split_mode=None)
        if args.tensor_split and count and len(args.tensor_split) != count:
            return recipe.with_args(tensor_split=None, main_gpu=None)
    return recipe


def check_split(recipe: Recipe) -> None:
    """`split=tensor` with what llama.cpp's tensor split needs, else every reason why not."""
    args = recipe.args
    if not isinstance(args, LlamaCppArgs) or args.split_mode != "tensor":
        return
    reasons = []
    cards = recipe_cards(recipe)
    if cards is not None and len(cards) < 2:
        reasons.append("it runs on one card (gpus=, MNM_GPU)")
    if args.flash_attn == "off":
        reasons.append("flash attention is off (flash-attn=on or auto)")
    if args.kv_type is not KvType.F16:
        reasons.append(f"its KV cache is quantised (kv={args.kv_type.value}; it needs kv=f16)")
    if recipe.architecture in NO_TENSOR_SPLIT:
        reasons.append(
            f"llama.cpp does not split {recipe.architecture} by tensor (not implemented "
            "upstream; split=layer)"
        )
    if args.experts_in_ram or recipe.fit_level is FitLevel.OFFLOAD:
        reasons.append(
            "its plan keeps layers or experts in RAM, which only llama.cpp's own fitter "
            "(--fit) could place, and --fit does not work with a tensor split"
        )
    if reasons:
        raise CardSettingError("split=tensor is refused: " + "; ".join(reasons) + ".")
