"""The parts of `InstanceService.start` that read the model files on disk: the once-only
`-ngl` raise, the switch to a tiled VAE, and what a planned auto-fix retry is sized with.
Pure: the service saves and words the results."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from makenomistakes.autofix import Resources
from makenomistakes.calibration import Correction
from makenomistakes.card_room import instance_cards
from makenomistakes.engine_args import LlamaCppArgs, SdCppArgs
from makenomistakes.engine_args_vllm import VllmArgs
from makenomistakes.fit.estimate import GIB, FitLevel, headroom_level
from makenomistakes.fit.vllm import MAX_SHARE, POOL_SHARE_CAP, share_for, vllm_estimate
from makenomistakes.hardware.probe import MemorySnapshot
from makenomistakes.hub.facts import Modality
from makenomistakes.model_files import read_model_files
from makenomistakes.plan import budget_on
from makenomistakes.recipes import Recipe
from makenomistakes.settings_estimates import settings_estimate


def all_layers_upgrade(
    recipe: Recipe, models_dir: Path
) -> tuple[Callable[[Recipe], Recipe], str] | None:
    """A llama.cpp recipe saved by an older mnm runs `-ngl <blocks>`, which
    keeps block 0 on the CPU for every token: llama.cpp offloads blocks + 1 layers, the
    output layer last. The change to save (raised once, a user's edit meanwhile wins) and
    what to say; None when nothing needs it. A planned offload lands on exactly the block
    count when the file is just over the free VRAM (`fit_estimate`): that is the planner's
    split and stays."""
    args = recipe.args
    if not isinstance(args, LlamaCppArgs) or recipe.fit_level is FitLevel.OFFLOAD:
        return None
    files = read_model_files(recipe, models_dir)
    blocks = files.header.block_count if files and files.header else None
    if not blocks or args.gpu_layers != blocks:
        return None

    def all_layers(saved: Recipe) -> Recipe:
        if isinstance(saved.args, LlamaCppArgs) and saved.args.gpu_layers == blocks:
            return saved.with_args(gpu_layers=blocks + 1)
        return saved  # changed meanwhile: the user's value wins

    return all_layers, (
        f"GPU layers: {blocks} -> {blocks + 1}, so every block runs on the GPU (llama.cpp "
        "counts the output layer as one more; block 0 had stayed on the CPU)."
    )


def tiling_upgrade(
    recipe: Recipe, models_dir: Path, memory: Callable[[], MemorySnapshot]
) -> tuple[Callable[[Recipe], Recipe], str] | None:
    """An image instance whose whole-picture VAE pass does not fit comfortably in the VRAM
    free now runs with a tiled VAE. sd.cpp retries a decode in tiles when memory runs out,
    but not the encode a repaint, inpaint or edit starts with: Z-Image's engine dies with
    exit 139 on the first picture request. The change to save and what to
    say; None when nothing needs it (tiled already, the VAE on the CPU, video, or room)."""
    args = recipe.args
    if (
        not isinstance(args, SdCppArgs)
        or args.vae_tiling
        or args.vae_on_cpu
        or recipe.modality not in (Modality.IMAGE_GEN, Modality.IMAGE_EDIT)
    ):
        return None
    files, snapshot = read_model_files(recipe, models_dir), memory()  # the card read last
    card = next(iter(instance_cards(recipe, snapshot)), None)  # its own card
    if files is None or card is None:
        return None
    estimate = settings_estimate(recipe, files)
    free = card.vram_free_bytes
    whole, tiled = estimate.bytes, estimate.image.tiled_bytes
    if whole is None or tiled is None or headroom_level(whole, free) is FitLevel.COMFORTABLE:
        return None

    def tiles(saved: Recipe) -> Recipe:
        if isinstance(saved.args, SdCppArgs) and not saved.args.vae_tiling:
            return saved.with_args(vae_tiling=True)
        return saved  # changed meanwhile: the user's value wins

    return tiles, (
        f"VAE tiling: off -> on. A whole {args.width}x{args.height} picture needs about "
        f"{whole / GIB:.1f} GiB and {free / GIB:.1f} GiB is free; in tiles about "
        f"{tiled / GIB:.1f} GiB. A repaint or edit encodes the picture first, which sd.cpp "
        "cannot retry in tiles (`mnm set <name> vae-tiling=off` to undo)."
    )


class StartRefusedError(ValueError):
    """Settings that cannot start on this card at all; `InstanceService.start` says why."""


def vllm_share_upgrade(
    recipe: Recipe, models_dir: Path, memory: Callable[[], MemorySnapshot]
) -> tuple[Callable[[Recipe], Recipe], str] | None:
    """A vLLM instance's share of the card, sized again for its settings against the real
    card: raised when a longer context needs more (`mnm set ctx=…`), lowered when it was
    planned against the free VRAM alone (`--vram`) and would ask more than is there. vLLM
    refuses to start both when its share is not free and when its KV cache cannot hold one
    full request; settings that need more than `MAX_SHARE` of the card are refused here."""
    args = recipe.args
    if not isinstance(args, VllmArgs):
        return None
    files, snapshot = read_model_files(recipe, models_dir), memory()
    gpu = next(iter(instance_cards(recipe, snapshot)), None)  # its own card (instances 10)
    if files is None or files.hf_config is None or gpu is None:
        return None
    card, most, of = gpu.vram_total_bytes, MAX_SHARE, "the card"
    pool = gpu.unified or gpu.carve_out
    if pool and snapshot.memory is not None:  # A share of the whole pool, capped firmly
        card, most, of = snapshot.memory.total_bytes, POOL_SHARE_CAP, "the unified memory"
    needed = share_for(vllm_estimate(files.weights_bytes, files.hf_config, args), card)
    if needed > most:
        raise StartRefusedError(
            f"{recipe.name}'s settings need {needed:.0%} of {of} (context "
            f"{args.max_model_len:,}); vLLM is never given more than {most:.0%}. Lower "
            f"the context (`mnm set {recipe.name} ctx=…`) or the KV precision (`kv=fp8`)."
        )
    if needed == args.gpu_memory_utilization:
        return None
    before = args.gpu_memory_utilization

    def share(saved: Recipe) -> Recipe:
        if isinstance(saved.args, VllmArgs) and saved.args.gpu_memory_utilization == before:
            return saved.with_args(gpu_memory_utilization=needed)
        return saved  # changed meanwhile: the newer value wins

    return share, (
        f"GPU share: {before:.0%} -> {needed:.0%}, what a {args.max_model_len:,}-token "
        "request needs on this card (vLLM refuses to start otherwise)."
    )


def retry_resources(
    recipe: Recipe, models_dir: Path, snapshot: MemorySnapshot, correction: Correction | None = None
) -> Resources | None:
    """Header, file sizes and free memory for a planned retry, priced with the engine's
    measured correction; None when unavailable."""
    files = read_model_files(recipe, models_dir)
    if files is None or files.header is None:
        return None
    cards = instance_cards(recipe, snapshot)  # every card it runs on
    ram = snapshot.memory.available_bytes if snapshot.memory else None
    # Without both the plan would guess: no budget means one blind step, counted from
    # the header's layers (a recipe set to 99 has the model's count on the GPU).
    budget = budget_on(cards, ram) if cards and ram is not None else None
    if budget is not None and correction is not None:
        budget = budget.corrected(correction.ratio, correction.note)
    return Resources(files.header, files.weights_bytes, files.projector_bytes, budget)


Upgrade = tuple[Callable[[Recipe], Recipe], str]


def start_upgrades(
    recipe: Recipe, models_dir: Path, memory: Callable[[], MemorySnapshot]
) -> list[Upgrade]:
    """The changes a start makes before it runs (each with what to say), in order: the one
    list `InstanceService.start` saves and a self-test copy runs. StartRefusedError as a
    start would refuse."""
    found = (
        all_layers_upgrade(recipe, models_dir),
        tiling_upgrade(recipe, models_dir, memory),
        vllm_share_upgrade(recipe, models_dir, memory),
    )
    return [upgrade for upgrade in found if upgrade is not None]


def as_started(recipe: Recipe, models_dir: Path, memory: Callable[[], MemorySnapshot]) -> Recipe:
    """The recipe as `InstanceService.start` would run it now, nothing saved: what a copy of
    it runs (`engine_selftest`), so the copy is the instance."""
    for change, _what in start_upgrades(recipe, models_dir, memory):
        recipe = change(recipe)
    return recipe
