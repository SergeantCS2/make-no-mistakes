"""What a saved recipe puts on the card with its current settings, from the files on disk:
each engine's own fit model (`settings_estimate`), so *Fits now* follows `mnm set` and
auto-fix."""

from __future__ import annotations

from dataclasses import dataclass

from makenomistakes.backend import Backend
from makenomistakes.backend_now import recipe_backend_or_none
from makenomistakes.capacity import ImageFallbacks, VramEstimate
from makenomistakes.engine_args import AudioCppArgs, LlamaCppArgs, SdCppArgs
from makenomistakes.engine_args_vllm import VllmArgs
from makenomistakes.engine_args_whisper import WhisperArgs
from makenomistakes.fit.audio import audio_estimate
from makenomistakes.fit.estimate import draft_gpu_bytes, settings_gpu_bytes
from makenomistakes.fit.image_memory import KitSizes
from makenomistakes.fit.music import music_estimate
from makenomistakes.fit.sdcpp import sd_settings_estimate
from makenomistakes.fit.vllm import share_for, vllm_estimate
from makenomistakes.fit.voice import voice_estimate
from makenomistakes.gguf.reader import GgufHeader
from makenomistakes.hub.facts import EMBEDDING_MODALITIES
from makenomistakes.hub.hf_config import HfConfig
from makenomistakes.hub.music import MusicFamily
from makenomistakes.hub.voices import VoiceFamily
from makenomistakes.recipes import Recipe


@dataclass(frozen=True)
class ModelFiles:
    """A downloaded recipe's files as they are on disk, read once (`InstanceService`)."""

    weights_bytes: int
    projector_bytes: int
    kit: KitSizes  # sd.cpp companions, split by what may move to RAM
    header: GgufHeader | None  # llama.cpp only
    hf_config: HfConfig | None = None  # vLLM only: the model's config.json on disk
    # llama.cpp with a draft model: its header and its files' size, priced at the
    # recipe's own context (`draft_gpu_bytes`)
    draft_header: GgufHeader | None = None
    draft_bytes: int = 0


def settings_estimate(
    recipe: Recipe, files: ModelFiles, card_bytes: int | None = None
) -> VramEstimate:
    """Each engine's own fit model over the files and the recipe's current settings, so the
    number follows `mnm set` and auto-fix. `card_bytes` is the card's total: vLLM takes a
    share of it (sized at start for its settings), not just what it needs."""
    args = recipe.args
    if isinstance(args, SdCppArgs):
        # Always-on LoRAs load with every request (and stay in VRAM on quantized weights).
        loras = sum(lora.size_bytes for lora in recipe.loras if lora.always_on)
        family = recipe.image_family
        estimate = sd_settings_estimate(files.weights_bytes, files.kit, args, family)
        needed = estimate.total_bytes + loras
        if args.offload_to_cpu:
            return VramEstimate(
                bytes=needed,
                basis="estimated from the model files and settings",
                image=ImageFallbacks(
                    streamed_floor=estimate.total_bytes - estimate.weights_bytes + loras
                ),
            )
        if not args.vae_tiling and not args.vae_on_cpu:
            tiled = args.model_copy(update={"vae_tiling": True})
            fallback = sd_settings_estimate(
                files.weights_bytes, files.kit, tiled, family
            ).total_bytes
            return VramEstimate(
                bytes=needed,
                basis="estimated from the model files and settings",
                image=ImageFallbacks(tiled_bytes=fallback + loras),
            )
    elif isinstance(args, WhisperArgs):
        needed = audio_estimate(recipe.quant, files.weights_bytes).total_bytes
    elif isinstance(args, AudioCppArgs) and args.music:  # The family's row
        needed = music_estimate(
            MusicFamily(args.family), files.weights_bytes, recipe.files[0]
        ).total_bytes
    elif isinstance(args, AudioCppArgs):
        needed = voice_estimate(VoiceFamily(args.family), files.weights_bytes).total_bytes
    elif isinstance(args, VllmArgs):
        if files.hf_config is None:
            return VramEstimate(bytes=None, basis="the model's config.json could not be read")
        need = vllm_estimate(files.weights_bytes, files.hf_config, args)
        needed = int(share_for(need, card_bytes) * card_bytes) if card_bytes else need.total_bytes
    elif files.header is None:
        return VramEstimate(bytes=None, basis="the model files on disk could not be read")
    else:
        try:
            mtp = recipe.drafts_with_mtp
            needed = settings_gpu_bytes(
                files.header,
                files.weights_bytes,
                files.projector_bytes,
                _priced(recipe, args, files.header),
                whole_input=recipe.modality in EMBEDDING_MODALITIES,
                rollback=args.draft_max if mtp else 0,  # a hybrid's state, kept per step back
            )
            if files.draft_header is not None:  # the draft's weights and KV beside it
                needed += draft_gpu_bytes(files.draft_header, files.draft_bytes, args, mtp=mtp)
        except ValueError:  # a header without head counts (kv_cache_bytes refuses it)
            return VramEstimate(bytes=None, basis="the model's header lacks attention sizes")
    return VramEstimate(bytes=needed, basis="estimated from the model files and settings")


def _priced(recipe: Recipe, args: LlamaCppArgs, header: GgufHeader) -> LlamaCppArgs:
    """The settings an instance's room is priced with: as saved on a card; on the processor
    (`-ngl 0`, every block in RAM) every block and expert counted, since there the room is the
    RAM's (without the RAM-held weights a model would read as holding its context alone)."""
    if recipe_backend_or_none(recipe) is not Backend.CPU:
        return args
    whole = {"gpu_layers": (header.block_count or 0) + 1, "cpu_moe_layers": None,
             "placement": None, "moe_cache_mib": None}  # fmt: skip
    return args.model_copy(update=whole)
