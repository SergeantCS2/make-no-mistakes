"""What a video model served by stable-diffusion.cpp puts on the GPU, and how long a clip
takes: the image fit model (`fit/image_memory.py`) with frames.

A video DiT works on latent tokens: the VAE compresses the picture by its stride (Wan 2.1 8x,
Wan 2.2's 5B and HunyuanVideo 16x, LTX-2 32x) and the frames by four (LTX-2 eight), and the
transformer patches each latent frame 2x2 (`VideoDefaults.spatial_compression` is the two
together, in pixels per token side). A 33-frame 832x480 Wan 2.1 clip is 9 latent frames of
1,560 tokens: 14,040 tokens, 3.4x a 1024x1024 image's. The diffusion buffer scales with those
tokens at the family's per-token rate (the image model's per-pixel rate times a token's
pixels); the VAE decodes a clip in groups of `temporal_compression` frames, so its buffer is
that many frames' worth of the per-pixel figure. The decode figure is fitted to one reading
on an RTX 4070; the diffusion one is still the image ceiling. The planner's placement
order is the image one's.

Time is compute-bound like an image's, with two passes per step under real CFG (every video
family here runs cfg > 1), plus the weights a streamed run re-reads per step.
"""

from __future__ import annotations

from makenomistakes.fit.estimate import MemoryEstimate
from makenomistakes.fit.image_memory import (
    COMPUTE_EFFICIENCY,
    IMAGE_OVERHEAD_BYTES,
    LATENT_PIXELS_PER_TOKEN,
    PCIE_BYTES_PER_SECOND,
    TEXT_TOKENS,
    diffusion_bytes_per_pixel,
)
from makenomistakes.hub.kits import KITS, ImageFamily, VideoDefaults

# The Wan 2.2 VAE's decode buffer: per pixel of the area decoded at once, plus per pixel per
# output frame of the clip (sd.cpp decodes every frame of a spatial tile together unless
# temporal tiling is asked for). The CPU build's shape (23.3 KB a pixel plus 2.95 KB a
# pixel-frame at 256x160) priced a run on an RTX 4070 at 7.4 GiB; the run (wan5b Q4_K_M
# at 832x480, 33 frames, tiled, `--offload-to-cpu`, which loads one stage's weights at a
# time) peaked at +5,101 MiB, the VAE stage: 1,345 MiB of VAE weights, the 256 MiB overhead
# and about 3.5 GiB of decode in a 256x256 tile. Both figures are the CPU ones x 0.464.
VIDEO_VAE_FIXED_BYTES_PER_PIXEL = 10_800
VIDEO_VAE_BYTES_PER_PIXEL_FRAME = 1_370

__all__ = [
    "VIDEO_VAE_BYTES_PER_PIXEL_FRAME",
    "VIDEO_VAE_FIXED_BYTES_PER_PIXEL",
    "latent_frames",
    "seconds_per_video",
    "video_estimate",
    "video_of",
    "video_tokens",
]


def video_of(family: ImageFamily) -> VideoDefaults | None:
    """The family's clip defaults, None for an image family (or one without a kit)."""
    kit = KITS.get(family)
    return kit.video if kit is not None else None


def latent_frames(frames: int, video: VideoDefaults) -> int:
    """The first frame stands alone, every following group of `temporal_compression` frames
    is one latent frame (33 frames at 4x: 9)."""
    return 1 + max(frames - 1, 0) // video.temporal_compression


def video_tokens(width: int, height: int, frames: int, video: VideoDefaults) -> int:
    per_frame = (width // video.spatial_compression) * (height // video.spatial_compression)
    return per_frame * latent_frames(frames, video)


def video_estimate(
    diffusion_bytes: int,
    companions_on_gpu: int,
    size: tuple[int, int],
    frames: int,
    video: VideoDefaults,
    family: ImageFamily = ImageFamily.UNKNOWN,
    decode_pixels: int | None = None,
) -> MemoryEstimate:
    """`size` is width x height; `decode_pixels` the area the VAE decodes at once (default:
    the whole frame), for every frame of the clip."""
    width, height = size
    tokens = video_tokens(width, height, frames, video)
    diffusion = tokens * LATENT_PIXELS_PER_TOKEN * diffusion_bytes_per_pixel(family)
    pixels = width * height if decode_pixels is None else decode_pixels
    decode = pixels * (VIDEO_VAE_FIXED_BYTES_PER_PIXEL + frames * VIDEO_VAE_BYTES_PER_PIXEL_FRAME)
    return MemoryEstimate(
        weights_bytes=diffusion_bytes,
        compute_bytes=max(diffusion, decode),
        companion_bytes=companions_on_gpu,
        overhead_bytes=IMAGE_OVERHEAD_BYTES,
    )


def seconds_per_video(
    params: float,
    tokens: int,
    steps: int,
    cfg_scale: float,
    gpu_compute: float,
    restreamed: int = 0,
    pcie_bandwidth: int = PCIE_BYTES_PER_SECOND,
) -> float:
    """Two FLOPs per parameter per token per pass; real CFG (cfg > 1) is two passes a step."""
    passes = 2 if cfg_scale > 1.0 else 1
    compute = 2.0 * params * (tokens + TEXT_TOKENS) * steps * passes
    return compute / (gpu_compute * COMPUTE_EFFICIENCY) + steps * restreamed / pcie_bandwidth
