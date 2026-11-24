"""What a diffusion model served by stable-diffusion.cpp puts on the GPU.

No GGUF header is involved: VRAM is the diffusion weights, the companions kept on the GPU
(VAE always; text encoders unless moved to RAM), and the larger of two phase buffers: the
diffusion compute buffer and the VAE decode buffer. sd.cpp frees each runner's buffer before
the next phase starts (`ggml_runner.cpp:479-500`), so they never sit in VRAM together. Time
per image is compute-bound: two FLOPs per parameter per latent token per step.

The VAE decode of the KL decoder every kit's VAE uses takes 6,656 bytes per output pixel (the
F16 im2col of its 256-channel convolutions at full resolution: 1,664.06 MB at 512x512 and
3,744.14 MB at 768x768 on a CPU build of master 2988060), so a whole 1024x1024 decode needs
6.5 GiB of its own. That is the peak of every 1024x1024 image, and why the planner decodes
in tiles before it moves the text encoders to the CPU. Left untiled, sd.cpp makes room by
evicting weights (the text encoder first, then diffusion blocks) and reloading them for the
next image; that is slower, never a failure (`model_manager.cpp:1810-1947`). The diffusion
buffer starts from upstream's FLUX figure (768x768 ~600 MB with flash attention) and is
labelled uncalibrated until a measured run on the 4070 replaces it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol

from makenomistakes.engine_args import SdCppArgs
from makenomistakes.fit.budget import Budget
from makenomistakes.fit.estimate import (
    MIB,
    Fit,
    FitLevel,
    MemoryEstimate,
    headroom_level,
)
from makenomistakes.hub.kits import MOVABLE_ROLES, ImageFamily, Role
from makenomistakes.hub.variants import GgufVariant
from makenomistakes.safetensors.sd import ImageParts

IMAGE_OVERHEAD_BYTES = 256 * MIB
COMPUTE_BYTES_PER_PIXEL = 1100  # diffusion compute buffer with flash attention (FLUX 768²≈600 MB)
VAE_DECODE_BYTES_PER_PIXEL = 6656  # the KL decoder, measured: 6.5 GiB at 1024x1024 untiled
# sd.cpp logs buffers in MiB (`ggml_runner.cpp:1019` divides by 1024 * 1024), so a reading at
# 512x512 is 4 bytes per pixel per MiB. The Qwen-Image (Wan) VAE Anima, Krea 2 and Qwen-Image
# share: `wan_vae compute buffer size: 1874.50 MB` (Krea 2, CPU build).
# Qwen-Image 2.1's own VAE: 1944.25 at 512x512 and 759.47 at 320x320, 7,777 B/px both times.
WAN_VAE_DECODE_BYTES_PER_PIXEL = 7498
_VAE_DECODE_BYTES_PER_PIXEL = {
    ImageFamily.ANIMA: WAN_VAE_DECODE_BYTES_PER_PIXEL,
    ImageFamily.KREA2: WAN_VAE_DECODE_BYTES_PER_PIXEL,
    ImageFamily.QWEN_IMAGE: WAN_VAE_DECODE_BYTES_PER_PIXEL,
    ImageFamily.QWEN_IMAGE_2_1: 7777,
}


# The diffusion buffer where a family's was read (`<arch> compute buffer size` at 512x512
# with flash attention in the DiT, CPU build): FLUX.2 klein 9B 442.00 MiB,
# Krea 2 229.33, Chroma 308.50. Qwen-Image 2.1's (261.64, 1,047 B/px) stays under the default.
_DIFFUSION_BYTES_PER_PIXEL = {
    ImageFamily.FLUX2_KLEIN_9B: 1768,
    ImageFamily.KREA2: 917,
    ImageFamily.CHROMA: 1234,
}


def diffusion_bytes_per_pixel(family: ImageFamily) -> int:
    """The family's diffusion compute buffer per output pixel (upstream's FLUX figure
    where nothing was measured)."""
    return _DIFFUSION_BYTES_PER_PIXEL.get(family, COMPUTE_BYTES_PER_PIXEL)


def vae_decode_bytes_per_pixel(family: ImageFamily) -> int:
    """What the family's VAE holds per output pixel while it decodes."""
    return _VAE_DECODE_BYTES_PER_PIXEL.get(family, VAE_DECODE_BYTES_PER_PIXEL)


LATENT_PIXELS_PER_TOKEN = 256  # 8x VAE and 2x2 patching: a 1024² image is 4096 tokens
TEXT_TOKENS = 512
COMPUTE_EFFICIENCY = 0.45  # fraction of peak fp16 FLOPs a diffusion step achieves
VAE_TILE_PIXELS = 256 * 256  # sd.cpp's default tile (`vae.hpp:229`): 416 MiB to decode


class Placement(Enum):
    """Where the kit's weights live and how the VAE decodes, tried in this order until one
    fits: a tiled decode costs a moment, an encoder on the CPU seconds per prompt."""

    GPU = "gpu"  # everything in VRAM
    GPU_TILED = "gpu_tiled"  # everything in VRAM, the VAE decodes in tiles
    ENCODERS_IN_RAM = "encoders_in_ram"  # text encoders run on the CPU (`--backend te=cpu`)
    ENCODERS_IN_RAM_TILED = "encoders_in_ram_tiled"
    STREAMED = "streamed"  # weights stay in RAM and stream per stage (`--offload-to-cpu`)


# Streaming is the last resort, so it decodes in tiles too.
_TILED = frozenset({Placement.GPU_TILED, Placement.ENCODERS_IN_RAM_TILED, Placement.STREAMED})


@dataclass(frozen=True)
class KitSizes:
    """Companion sizes resolved from the Hub, split by whether they may move to RAM."""

    on_gpu_bytes: int  # the VAE (and CLIP-vision)
    movable_bytes: int  # text encoders

    @property
    def total(self) -> int:
        return self.on_gpu_bytes + self.movable_bytes


class _Placed(Protocol):
    """A kit companion (Hub planning) or a recipe's RepoFile (files on disk): both say what
    they are and where their size is filed."""

    @property
    def role(self) -> Role: ...

    @property
    def key(self) -> str: ...


def kit_sizes(companions: Iterable[_Placed], sizes: Mapping[str, int]) -> KitSizes:
    """Split companion sizes by what `--backend te=cpu` can move to RAM (missing ones are 0)."""
    on_gpu = movable = 0
    for companion in companions:
        size = sizes.get(companion.key, 0)
        if companion.role in MOVABLE_ROLES:
            movable += size
        else:
            on_gpu += size
    return KitSizes(on_gpu_bytes=on_gpu, movable_bytes=movable)


@dataclass(frozen=True)
class CheckpointSizes:
    """A safetensors checkpoint or diffusers folder as placed: what the plan prices and what
    Fits now reads back from the files on disk (both through `checkpoint_sizes`)."""

    diffusion_bytes: int  # the UNet as loaded
    sizes: KitSizes  # the VAE on the GPU (the kit's when it brings one), text encoders movable


# A streamed run (`--offload-to-cpu`, sd.cpp docs/performance.md): the weights stay in RAM,
# GPU replicas are made on demand and kept while there is room, and when weights plus compute
# exceed what is free the graph runs in segments, evicting and re-reading weights every step.
# The engine keeps a 512 MiB scratch reserve plus 128 MiB of planning headroom to itself.
STREAM_RESERVE_BYTES = 640 * MIB
PCIE_BYTES_PER_SECOND = 12 * 10**9  # host to GPU under WSL, an estimate


def checkpoint_sizes(
    parts: ImageParts, kit: KitSizes, diffusion_type: Literal["bf16", "q8_0"] | None = None
) -> CheckpointSizes:
    """`kit`: the kit's own companions. The kit's VAE replaces the file's (SDXL: fp16-fix);
    the file's text encoders are movable like any kit's. A DiT's diffusion model is sized
    for the conversion it loads with (`diffusion_type`)."""
    return CheckpointSizes(
        diffusion_bytes=parts.loaded_diffusion_bytes(diffusion_type),
        sizes=KitSizes(
            on_gpu_bytes=kit.on_gpu_bytes or parts.vae_bytes,
            movable_bytes=kit.movable_bytes + parts.text_encoder_bytes,
        ),
    )


@dataclass(frozen=True)
class _Weights:
    """Diffusion weights to place: what downloads (and the quant its label shows), and what
    they take once loaded. A GGUF quant loads as stored; a checkpoint loads its UNet."""

    variant: GgufVariant
    loaded_bytes: int
    params_b: float  # for the time per image
    diffusion_type: Literal["bf16", "q8_0"] | None = None  # a DiT converted while loading


def placement_flags(placement: Placement) -> dict[str, bool]:
    """The `SdCppArgs` flags a placement is: what a profile saves and what Fits now reads."""
    return {
        "offload_to_cpu": placement is Placement.STREAMED,
        "text_encoder_on_cpu": placement not in (Placement.GPU, Placement.GPU_TILED),
        "vae_tiling": placement in _TILED,
    }


def image_estimate(
    diffusion_bytes: int,
    companions_on_gpu: int,
    width: int,
    height: int,
    decode_pixels: int | None = None,
    family: ImageFamily = ImageFamily.UNKNOWN,
) -> MemoryEstimate:
    """`decode_pixels`: the area the VAE decodes on the GPU at once (default: the image).
    The two phase buffers never coexist: the larger one is the peak."""
    pixels = width * height
    decode = pixels if decode_pixels is None else decode_pixels
    return MemoryEstimate(
        weights_bytes=diffusion_bytes,
        compute_bytes=max(
            pixels * diffusion_bytes_per_pixel(family), decode * vae_decode_bytes_per_pixel(family)
        ),
        companion_bytes=companions_on_gpu,
        overhead_bytes=IMAGE_OVERHEAD_BYTES,
    )


def image_settings_estimate(
    diffusion_bytes: int,
    sizes: KitSizes,
    args: SdCppArgs,
    family: ImageFamily = ImageFamily.UNKNOWN,
) -> MemoryEstimate:
    """What sd-server puts on the GPU with these settings. The planner prices its placements
    through this too (`placement_flags`), so Fits now and the plan cannot disagree. Streamed
    weights (`offload_to_cpu`) take the text encoders to RAM with them. Auto-fix's VAE steps
    are counted roughly: on the CPU the VAE decodes in RAM, tiled it decodes a tile at a time."""
    weights = diffusion_bytes  # streamed too: the replicas sit in VRAM while there is room
    encoders = 0 if args.text_encoder_on_cpu or args.offload_to_cpu else sizes.movable_bytes
    vae = 0 if args.vae_on_cpu else sizes.on_gpu_bytes
    pixels = args.width * args.height
    decode = 0 if args.vae_on_cpu else min(pixels, VAE_TILE_PIXELS) if args.vae_tiling else None
    return image_estimate(weights, vae + encoders, args.width, args.height, decode, family)


def image_fit(estimate: MemoryEstimate, budget: Budget, placement: Placement) -> Fit:
    """Levels mean the same as for llama.cpp; gpu_layers is not a concept here (always 0).

    Streamed (`--offload-to-cpu`): the engine takes the free VRAM it finds, minus its reserve,
    and runs in segments when the weights do not all fit, so what it puts on the GPU is the
    smaller of the whole estimate and that free share; what must fit is the compute and VAE
    buffers alone (the weights can always be re-read). RAM is the caller's check."""
    free = budget.vram_free_bytes
    if placement is Placement.STREAMED:
        floor = estimate.total_bytes - estimate.weights_bytes  # the largest segment's own need
        if floor > free:
            return Fit(
                level=FitLevel.NO_FIT,
                gpu_layers=0,
                headroom_bytes=free - floor,
                gpu_bytes=floor,
                room_bytes=free,
            )
        on_gpu = min(estimate.total_bytes, max(free - STREAM_RESERVE_BYTES, floor))
        return Fit(
            level=FitLevel.OFFLOAD,
            gpu_layers=0,
            headroom_bytes=free - on_gpu,
            gpu_bytes=on_gpu,
            room_bytes=free,
        )
    headroom = free - estimate.total_bytes
    level = headroom_level(estimate.total_bytes, free)
    return Fit(
        level=level,
        gpu_layers=0,
        headroom_bytes=headroom,
        gpu_bytes=estimate.total_bytes,
        room_bytes=free,
    )


def restreamed_bytes(total_bytes: int, vram_free_bytes: int) -> int:
    """What a streamed run re-reads from RAM every step: the part of its total the free
    VRAM (less the engine's reserve) cannot hold; 0 when the weights stay resident."""
    return max(total_bytes - (vram_free_bytes - STREAM_RESERVE_BYTES), 0)


def seconds_per_image(
    params: float,
    width: int,
    height: int,
    steps: int,
    gpu_compute: float,
    restreamed: int = 0,
    pcie_bandwidth: int = PCIE_BYTES_PER_SECOND,
) -> float:
    """Compute time, plus the weights a streamed run re-reads over the bus each step."""
    tokens = width * height / LATENT_PIXELS_PER_TOKEN + TEXT_TOKENS
    compute = 2.0 * params * tokens * steps / (gpu_compute * COMPUTE_EFFICIENCY)
    return compute + steps * restreamed / pcie_bandwidth
