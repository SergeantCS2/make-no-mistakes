"""sd-server's settings: picture, edit and video instances."""

from __future__ import annotations

from typing import get_args

from makenomistakes.controls_catalog import (
    IMAGE_MODALITIES,
    SD_MODALITIES,
    VIDEO_MODALITIES,
    Control,
    Kind,
    check_controls,
)
from makenomistakes.engine_args import SdCppArgs
from makenomistakes.hub.kits import Sampler, Scheduler

IMAGE_CONTROLS: tuple[Control, ...] = (
    Control(
        key="width",
        field="width",
        label="Default width",
        help="Pixels, a multiple of 16. Every image uses it, whatever size the client asks "
        "for, unless client-size is on.",
        kind=Kind.INT,
        recommended="SDXL (Illustrious, Pony): about one megapixel, 1024x1024, 832x1216 or "
        "1216x832; SD 1.x: 512x512 or 512x768",
        modalities=SD_MODALITIES,
    ),
    Control(
        key="height",
        field="height",
        label="Default height",
        help="Pixels, a multiple of 16. Every image uses it unless client-size is on.",
        kind=Kind.INT,
        recommended="the pair to the width: 1216 for an 832-wide SDXL portrait",
        modalities=SD_MODALITIES,
    ),
    Control(
        key="frames",
        field="frames",
        label="Frames per clip",
        help="How many frames a clip has unless the request says otherwise. The first frame "
        "stands alone and every 4 after it make one latent frame (LTX-2: 8), so 33, 49, 81 "
        "(and 65, 97, 121 for LTX-2) land exactly; at 16 fps 33 frames are two seconds.",
        kind=Kind.INT,
        recommended="33 to try, 81 for a five-second clip (VRAM and time grow with it)",
        modalities=VIDEO_MODALITIES,
    ),
    Control(
        key="fps",
        field="fps",
        label="Frames per second",
        help="The clip's frame rate: it changes the file's timing, not what the model draws.",
        kind=Kind.INT,
        recommended="16 for Wan and LingBot (how they were trained), 24 for HunyuanVideo and LTX-2",
        modalities=VIDEO_MODALITIES,
    ),
    Control(
        key="high-noise-steps",
        field="high_noise_steps",
        label="High-noise steps",
        help="Wan 2.2 A14B only: steps the high-noise model takes before the low-noise one "
        "takes `steps`. Empty leaves it to the engine's boundary.",
        kind=Kind.INT,
        recommended="8 with 10 low-noise steps (sd.cpp's example); 4 and 4 with the lightx2v LoRAs",
        nullable=True,
        modalities=VIDEO_MODALITIES,
    ),
    Control(
        key="client-size",
        field="client_size",
        label="Client picks the size",
        help=(
            "Off: mnm's router drops a client's size, so every image is this instance's "
            "width x height (Odysseus asks 512x512 for every direct message). On: a client's "
            "size request wins, as the engine alone would do."
        ),
        kind=Kind.BOOL,
        recommended="off",
        modalities=IMAGE_MODALITIES,
    ),
    Control(
        key="steps",
        field="steps",
        label="Sampling steps",
        help="More steps, more detail, linear in time. Distilled (turbo) models need 4-8.",
        kind=Kind.INT,
        recommended="the family default (set by the profile); never above 30 on this GPU",
        modalities=SD_MODALITIES,
    ),
    Control(
        key="strength",
        field="strength",
        label="Repaint strength",
        help="For a picture sent to `/v1/images/edits` (repaint, or inpaint with a mask): 0.3 "
        "keeps its layout and colours, 1.0 redraws it. Instruction editors (Kontext) ignore it.",
        kind=Kind.FLOAT,
        nullable=True,
        recommended="empty (the engine's 0.75); 0.4-0.6 to keep the composition",
        modalities=IMAGE_MODALITIES,
    ),
    Control(
        key="cfg-scale",
        field="cfg_scale",
        label="Guidance (cfg scale)",
        help=(
            "How strongly the prompt steers. Guidance-distilled models (FLUX, Z-Image Turbo) "
            "use 1.0."
        ),
        kind=Kind.FLOAT,
        recommended="1.0 for distilled models, 5-7 for SDXL and base models",
        modalities=SD_MODALITIES,
    ),
    Control(
        key="sampler",
        field="sampler",
        label="Sampler",
        help="euler for flow models (FLUX, Z-Image, SD3), euler_a for SDXL.",
        kind=Kind.CHOICE,
        choices=get_args(Sampler),
        recommended="the family default",
        modalities=SD_MODALITIES,
    ),
    Control(
        key="scheduler",
        field="scheduler",
        label="Sigma scheduler",
        help="How noise levels are spaced across the steps. Empty lets the engine pick per family.",
        kind=Kind.CHOICE,
        choices=get_args(Scheduler),
        nullable=True,
        recommended="empty",
        modalities=SD_MODALITIES,
    ),
    Control(
        key="seed",
        field="seed",
        label="Seed",
        help="Empty gives a new random image each time; a number makes the same prompt repeatable.",
        kind=Kind.INT,
        nullable=True,
        recommended="empty",
        modalities=SD_MODALITIES,
    ),
    Control(
        key="clip-skip",
        field="clip_skip",
        label="CLIP skip",
        help="Ignore the last text-encoder layers: 1 none, 2 one layer (SD1/SDXL style models).",
        kind=Kind.INT,
        nullable=True,
        recommended="empty (the engine picks per model)",
        modalities=SD_MODALITIES,
    ),
    Control(
        key="negative-prompt",
        field="negative_prompt",
        label="Default negative prompt",
        help="Used when a request sends none; typed literally (the words none/auto are prompt "
        "text here). Turbo and FLUX models ignore it (cfg 1).",
        kind=Kind.TEXT,
        recommended="empty, or e.g. 'blurry, low quality, watermark' for SDXL",
        modalities=SD_MODALITIES,
    ),
    Control(
        key="guidance",
        field="guidance",
        label="Distilled guidance",
        help="FLUX-style guidance embedding (not cfg). Empty uses the engine default of 3.5.",
        kind=Kind.FLOAT,
        nullable=True,
        recommended="empty; 3.5 for FLUX.1 dev",
        modalities=SD_MODALITIES,
    ),
    Control(
        key="flow-shift",
        field="flow_shift",
        label="Flow shift",
        help="Timestep shift for flow models. Empty lets the engine pick per model.",
        kind=Kind.FLOAT,
        nullable=True,
        recommended="empty",
        modalities=SD_MODALITIES,
    ),
    Control(
        key="flash-attn",
        field="flash_attn",
        label="Flash attention",
        help="Less memory and faster on CUDA. Turn off only if images come out black.",
        kind=Kind.BOOL,
        recommended="on",
        modalities=SD_MODALITIES,
    ),
    Control(
        key="diffusion",
        field="diffusion_type",
        label="Diffusion model precision",
        help=(
            "How a single DiT file's diffusion model is loaded: as stored (empty), or converted "
            "while loading: q8_0 halves a BF16/FP8 model's memory for a slight quality loss, "
            "bf16 keeps it. GGUF weights are quantized already and have no choice here."
        ),
        kind=Kind.CHOICE,
        choices=("bf16", "q8_0"),
        nullable=True,
        recommended="as stored when it fits; q8_0 when the profile picked it",
        modalities=SD_MODALITIES,
    ),
    Control(
        key="offload",
        field="offload_to_cpu",
        label="Stream weights from RAM",
        help=(
            "Keeps weights in system RAM and moves them to the GPU per stage: fits bigger "
            "models, slower."
        ),
        kind=Kind.BOOL,
        recommended="off when the profile fits on the GPU (it sets this)",
        modalities=SD_MODALITIES,
    ),
    Control(
        key="vae-tiling",
        field="vae_tiling",
        label="VAE tiling",
        help="Decodes the image in tiles: much less VRAM at large sizes, slight seams possible.",
        kind=Kind.BOOL,
        recommended="off below 1024x1024; on for larger images",
        modalities=SD_MODALITIES,
    ),
    Control(
        key="text-encoder-cpu",
        field="text_encoder_on_cpu",
        label="Text encoder on CPU",
        help="Runs the prompt encoder in RAM; saves its VRAM, adds a second or two per image.",
        kind=Kind.BOOL,
        recommended="off unless VRAM is short (the profile sets this)",
        modalities=SD_MODALITIES,
    ),
    Control(
        key="vae-cpu",
        field="vae_on_cpu",
        label="VAE on CPU",
        help="Decodes the final image on the CPU; saves VRAM at the cost of several seconds.",
        kind=Kind.BOOL,
        recommended="off",
        modalities=SD_MODALITIES,
    ),
    Control(
        key="threads",
        field="threads",
        label="CPU threads",
        help="For stages that run on the CPU. Empty uses the physical core count.",
        kind=Kind.INT,
        nullable=True,
        recommended="empty",
        modalities=SD_MODALITIES,
    ),
)


check_controls([(c, SdCppArgs) for c in IMAGE_CONTROLS])
