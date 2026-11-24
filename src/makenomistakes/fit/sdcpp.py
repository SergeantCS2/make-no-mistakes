"""The one estimate of what sd-server puts on the GPU for a recipe's settings: the planner
prices its placements through it and Fits now reads it, so the two never disagree. The kit's
family decides image or video (`fit/image_memory.py`, `fit/video.py`)."""

from __future__ import annotations

from collections.abc import Mapping

from makenomistakes.engine_args import SdCppArgs
from makenomistakes.fit.estimate import MemoryEstimate
from makenomistakes.fit.image_memory import (
    VAE_TILE_PIXELS,
    KitSizes,
    image_settings_estimate,
    seconds_per_image,
)
from makenomistakes.fit.picture_factors import FAMILY_SPEEDS, SPEED_MODEL
from makenomistakes.fit.quant_ladder import quant_bits
from makenomistakes.fit.video import seconds_per_video, video_estimate, video_of, video_tokens
from makenomistakes.hub.kits import ImageFamily, VideoDefaults

__all__ = [
    "FAMILY_SPEEDS",
    "SPEED_MODEL",
    "recipe_seconds",
    "sd_seconds",
    "sd_settings_estimate",
]


def sd_settings_estimate(
    diffusion_bytes: int,
    sizes: KitSizes,
    args: SdCppArgs,
    family: ImageFamily = ImageFamily.UNKNOWN,
) -> MemoryEstimate:
    """`diffusion_bytes` is every diffusion file the instance loads (Wan 2.2 A14B's two
    halves together). A video family's clip is the recipe's frames, else the kit's."""
    video = video_of(family)
    if video is None:
        return image_settings_estimate(diffusion_bytes, sizes, args, family)
    encoders = 0 if args.text_encoder_on_cpu or args.offload_to_cpu else sizes.movable_bytes
    vae = 0 if args.vae_on_cpu else sizes.on_gpu_bytes
    pixels = args.width * args.height
    decode = 0 if args.vae_on_cpu else min(pixels, VAE_TILE_PIXELS) if args.vae_tiling else None
    frames = args.frames or video.frames
    size = (args.width, args.height)
    return video_estimate(diffusion_bytes, vae + encoders, size, frames, video, family, decode)


def family_speed(
    family: ImageFamily | None,
    model: int | None = None,
    speeds: Mapping[ImageFamily, float] | None = None,
) -> float:
    """A family's factor under a speed model (this mnm's `SPEED_MODEL` when none is given);
    `speeds` is the card's own row of today's model (`DeviceRow.picture_speeds`)."""
    if family is None:
        return 1.0
    if model is None and speeds is not None:
        return speeds.get(family, 1.0)
    return FAMILY_SPEEDS[SPEED_MODEL if model is None else model].get(family, 1.0)


def speed_refit(family: ImageFamily | None, made_with: int | None) -> float | None:
    """What a reading's seconds-over-estimate is multiplied by to compare it with today's
    estimate: its estimate came from speed model `made_with` (None: model 1, before readings
    carried one), today's from `SPEED_MODEL`. A factor speeds up the compute, so an estimate
    scales as 1/factor and the ratio as today's factor over the reading's. None for a model
    this mnm does not know (a newer mnm wrote it): such a reading corrects nothing here."""
    made = made_with or 1
    if made not in FAMILY_SPEEDS:
        return None
    return family_speed(family) / family_speed(family, made)


def sd_seconds(
    params: float,
    args: SdCppArgs,
    video: VideoDefaults | None,
    gpu_compute: float,
    restreamed: int = 0,
    family: ImageFamily | None = None,
    speeds: Mapping[ImageFamily, float] | None = None,
) -> float:
    """The fit model's seconds for one picture (or, with the family's `video`, one clip) at
    these settings: what the planner prints as s/image and s/clip, before any correction
    measured on this card (`calibration.speed_correction`). The family's measured factor
    (`family_speed`) speeds up the compute, never the weights a streamed run re-reads."""
    compute = gpu_compute * family_speed(family, speeds=speeds)
    if video is None:
        return seconds_per_image(params, args.width, args.height, args.steps, compute, restreamed)
    steps = args.steps + (args.high_noise_steps or 0)
    tokens = video_tokens(args.width, args.height, args.frames or video.frames, video)
    return seconds_per_video(params, tokens, steps, args.cfg_scale, compute, restreamed)


def recipe_seconds(
    diffusion_bytes: int,
    quant: str,
    args: SdCppArgs,
    family: ImageFamily,
    two_stage: bool,
    gpu_compute: float,
    speeds: Mapping[ImageFamily, float] | None = None,
) -> float | None:
    """The same estimate for a saved instance, its parameters read from the diffusion files
    on disk at the quant's bits (a two-stage pair runs one half per step); None when the
    settings make it no estimate: weights streamed per step (what is re-read depends on the
    card at that moment) or a client's own size (the router passed it on)."""
    if args.offload_to_cpu or args.client_size:
        return None
    params = diffusion_bytes * 8 / (quant_bits(quant) or 8.0) / (2 if two_stage else 1)
    return sd_seconds(params, args, video_of(family), gpu_compute, family=family, speeds=speeds)
