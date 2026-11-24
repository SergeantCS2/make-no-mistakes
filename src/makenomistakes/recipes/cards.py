"""Which cards an instance runs on, and what Docker and llama.cpp are told.

`Recipe.gpus` pins an instance to cards (`mnm set <name> gpus=0` or `gpus=0,1`). Without
it, a llama.cpp instance runs on every card `MNM_GPU` names, a vLLM one on all of them when
its plan took tensor parallel (equal cards only) and on the first otherwise, and sd.cpp,
whisper.cpp and audio.cpp on the first: they run on one card. With `MNM_GPU` one index
every instance is on that card.

llama.cpp across cards (`docs/multi-gpu.md`, `common/arg.cpp`, `src/llama-model.cpp`, read
2026-10-22): `-sm layer` (its default, said anyway) splits the `-ngl` layers and their KV
across the cards in order, `-ts` gives each card's share (proportions; without it llama.cpp
splits by each card's free memory), `-mg` the card for intermediate results; `-sm tensor`
splits each layer's weights and KV (experimental: flash attention on, an unquantised KV).
"""

from __future__ import annotations

from makenomistakes.backend import Backend, device_argv, device_name
from makenomistakes.backend_now import host_devices, recipe_backend
from makenomistakes.card_ids import live_ids_now
from makenomistakes.card_placement import Placement
from makenomistakes.hardware.gpu_select import CardChoice, chosen
from makenomistakes.hub.facts import Engine
from makenomistakes.recipes.model import Recipe

ONE_CARD_ENGINES = frozenset({Engine.SDCPP, Engine.WHISPERCPP, Engine.AUDIOCPP})
ELSEWHERE = (
    "{name} was placed on other cards: every expert stays in RAM until `mnm machine adopt` "
    "places it again"
)


def recipe_cards(
    recipe: Recipe, choice: CardChoice | None = None, listed: list[int] | None = None
) -> tuple[int, ...] | None:
    """The nvidia-smi indices the instance runs on, in `MNM_GPU`'s order; None for every
    card (`MNM_GPU=all` with nothing listed to resolve it against)."""
    if recipe.gpus:
        return recipe.gpus
    choice = choice or chosen()
    if recipe.engine in ONE_CARD_ENGINES or not choice.several:
        return (choice.first,)
    if recipe.engine is Engine.VLLM and not recipe.vllm_args.tensor_parallel:
        return (choice.first,)
    if choice.every:
        return tuple(listed) if listed else None
    return choice.indices


def device_args(recipe: Recipe) -> list[str]:
    """What `docker run` is told to reach the instance's cards, by its image's backend
    (`backend.device_argv`): `--gpus` and `CUDA_DEVICE_ORDER` for CUDA; ROCm's
    and Vulkan's devices; nothing for the CPU images."""
    backend = recipe_backend(recipe)
    return device_argv(backend, recipe_cards(recipe), host_devices(backend))


def split_argv(recipe: Recipe) -> list[str]:
    """llama.cpp's split flags: none on one card; `-sm`, and `-ts`/`-mg` as
    planned when the plan's split names as many cards as the instance runs on (a split
    planned for other cards is left to llama.cpp's own, by free memory). A placement
    renders `--device`, `-ts`, `-mg` and its one `-ot` while it names exactly the cards the
    instance runs on, found by their ids where they are now (a card that moved slot
    keeps its part); on other cards (or one) every expert stays in RAM (`--cpu-moe`, safe on any
    cards, slow) until `mnm machine adopt` places it again (`placement_note` says so)."""
    args = recipe.llama_args
    backend = recipe_backend(recipe)
    if backend is Backend.CPU:
        return []  # no card to split across or place on
    cards = recipe_cards(recipe)
    one = cards is not None and len(cards) < 2
    if one and args.placement is None:
        return []
    argv = [] if one else ["-sm", args.split_mode or "layer"]
    if args.placement is not None:
        placement = placed_here(recipe, cards)
        if placement is not None and cards is not None:
            prefix = device_name(backend, 0)[:-1]  # `CUDA`, `ROCm`, `Vulkan`
            return argv + placement.argv(cards, args.lazy_tensors, prefix)
        lazy = ",".join(f"{t}=CPU" for t in args.lazy_tensors)
        return argv + ["--cpu-moe"] + (["-ot", lazy] if lazy else [])
    split = args.tensor_split
    if split and (cards is None or len(split) == len(cards)):
        argv += ["-ts", ",".join(str(n) for n in split)]
        if args.main_gpu is not None:
            argv += ["-mg", str(args.main_gpu)]
    return argv


def live_placement(recipe: Recipe) -> Placement | None:
    """The recipe's placement on the cards where they are now (`Placement.remapped` over
    this box as last recorded); None without one, or when a card it names is gone."""
    placement = recipe.llama_args.placement if recipe.engine is Engine.LLAMACPP else None
    return placement.remapped(live_ids_now()) if placement is not None else None


def placed_here(recipe: Recipe, cards: tuple[int, ...] | None) -> Placement | None:
    """The placement as it renders on `cards` (the instance's), or None: it names other
    cards, or none (`MNM_GPU=all` unresolved)."""
    placement = live_placement(recipe)
    if placement is None or cards is None or set(cards) != set(placement.indices):
        return None
    return placement


def placement_note(recipe: Recipe) -> str | None:
    """The one line a start (the CLI's, the router's journal) says when a placement no
    longer lands on the instance's cards; None when it does or there is none."""
    if recipe.engine is not Engine.LLAMACPP or recipe.llama_args.placement is None:
        return None
    if placed_here(recipe, recipe_cards(recipe)) is not None:
        return None
    return ELSEWHERE.format(name=recipe.name)
