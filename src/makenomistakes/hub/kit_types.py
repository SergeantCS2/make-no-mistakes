"""The vocabulary of image kits: families, companion roles, samplers, defaults, the kit.

`hub/kits.py` holds the table of kits and the name rules; both import from here, and kits.py
re-exports every name so callers import from one place.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class ImageFamily(StrEnum):
    Z_IMAGE = "z_image"  # Z-Image / Z-Image-Turbo (6B), Qwen3-4B text encoder, FLUX.1 VAE
    FLUX2_KLEIN = "flux2_klein"  # FLUX.2 klein 4B, Qwen3-4B text encoder, FLUX.2 VAE
    FLUX1 = "flux1"  # FLUX.1 dev / schnell (12B), clip_l + t5xxl, FLUX.1 VAE
    SDXL = "sdxl"  # single checkpoint (Illustrious, Pony, NoobAI too); fp16-fix VAE
    SD1 = "sd1"  # Stable Diffusion 1.x checkpoints and fine-tunes
    QWEN_IMAGE = "qwen_image"  # 20B: recognised, too large for 12 GB
    ANIMA = "anima"  # Anima (2B Cosmos DiT), Qwen3-0.6B text encoder, Qwen-Image VAE
    FLUX2_KLEIN_9B = "flux2_klein_9b"  # klein 9B: Qwen3-8B text encoder, the FLUX.2 VAE
    QWEN_IMAGE_2_1 = "qwen_image_2_1"  # 7B, Qwen3-VL-8B text encoder, its own 64-channel VAE
    KREA2 = "krea2"  # Krea 2 (13B), Qwen3-VL-4B text encoder, the Qwen-Image VAE
    ERNIE_IMAGE = "ernie_image"  # ERNIE-Image (8B), Ministral 3B text encoder, FLUX.2 VAE
    CHROMA = "chroma"  # FLUX-derived without CLIP: T5 alone and the FLUX.1 VAE
    # Recognised from a safetensors header or a name, no kit: named in refusals, and a LoRA
    # for one of them is refused on an instance of another family. The DiT ones are the
    # versions stable-diffusion.cpp's `get_sd_version` tells apart (master 2988060).
    SD2 = "sd2"
    SDXL_REFINER = "sdxl_refiner"  # an img2img second stage, not text-to-image
    SD3 = "sd3"
    FLUX2 = "flux2"  # FLUX.2 dev (32B, a 24B text encoder)
    FLUX1_VARIANT = "flux1_variant"  # FLUX.1 Fill, Controls, Flex.2: other tasks
    CHROMA_RADIANCE = "chroma_radiance"  # pixel space, no VAE
    LONGCAT = "longcat"
    OVIS_IMAGE = "ovis_image"
    LENS = "lens"  # needs a tokenizer file (`--tokenizer`)
    MAGE_FLOW = "mage_flow"
    QWEN_IMAGE_LAYERED = "qwen_image_layered"
    Z_IMAGE_L2P = "z_image_l2p"  # pixel space, no VAE
    BOOGU = "boogu"
    PIXART = "pixart"
    SEFI = "sefi"
    LLADA_IMAGE = "llada_image"
    IDEOGRAM4 = "ideogram4"
    PID = "pid"  # a decoder that needs a reference image
    MINIT2I = "minit2i"
    IRIS = "iris"
    # Video families (Modality.VIDEO, the same engine's `vid_gen` route). The split follows
    # what each needs loaded, not sd.cpp's `SDVersion` alone: Wan 2.1 I2V adds CLIP-vision,
    # the 5B its own VAE, the A14B pair a second (high-noise) diffusion model.
    WAN = "wan"  # Wan 2.1 T2V 1.3B/14B, VACE, a single Wan 2.2 A14B file: umt5 + Wan 2.1 VAE
    WAN_I2V = "wan_i2v"  # Wan 2.1 I2V / FLF2V 14B: + clip_vision_h
    WAN22_TI2V = "wan22_ti2v"  # Wan 2.2 TI2V 5B: its own 16x VAE, text or image to video
    WAN22_A14B = "wan22_a14b"  # Wan 2.2 T2V/I2V A14B: a high- and a low-noise DiT
    WAN22_S2V = "wan22_s2v"  # speech-to-video: an audio encoder and a WAV in, no kit
    LINGBOT_VIDEO = "lingbot_video"  # LingBot-Video 1.3B: Qwen3-VL-4B text encoder, Wan 2.1 VAE
    HUNYUAN_VIDEO = "hunyuan_video"  # HunyuanVideo 1.5 (8B): Qwen2.5-VL-7B + ByT5, its own VAE
    LTX2 = "ltx2"  # LTX-2.3 / 2.5 (22B): Gemma text encoder, video and audio VAEs
    MINIMAX_H3 = "minimax_h3"  # audio-video: no kit
    UNKNOWN = "unknown"


# Families whose instances make videos (`Modality.VIDEO`): planned by `fit/video.py`,
# served through `/sdcpp/v1/vid_gen`. The engine refuses `/v1/images/generations` for them.
VIDEO_FAMILIES = frozenset(
    {
        ImageFamily.WAN,
        ImageFamily.WAN_I2V,
        ImageFamily.WAN22_TI2V,
        ImageFamily.WAN22_A14B,
        ImageFamily.WAN22_S2V,
        ImageFamily.LINGBOT_VIDEO,
        ImageFamily.HUNYUAN_VIDEO,
        ImageFamily.LTX2,
        ImageFamily.MINIMAX_H3,
    }
)


FAMILY_LABELS: dict[ImageFamily, str] = {
    ImageFamily.Z_IMAGE: "Z-Image",
    ImageFamily.FLUX2_KLEIN: "FLUX.2 klein",
    ImageFamily.FLUX1: "FLUX.1",
    ImageFamily.SDXL: "SDXL",
    ImageFamily.SD1: "Stable Diffusion 1.x",
    ImageFamily.QWEN_IMAGE: "Qwen-Image",
    ImageFamily.ANIMA: "Anima",
    ImageFamily.FLUX2_KLEIN_9B: "FLUX.2 klein 9B",
    ImageFamily.QWEN_IMAGE_2_1: "Qwen-Image 2.1",
    ImageFamily.KREA2: "Krea 2",
    ImageFamily.ERNIE_IMAGE: "ERNIE-Image",
    ImageFamily.CHROMA: "Chroma",
    ImageFamily.SD2: "Stable Diffusion 2.x",
    ImageFamily.SDXL_REFINER: "the SDXL refiner",
    ImageFamily.SD3: "SD3",
    ImageFamily.FLUX2: "FLUX.2 dev",
    ImageFamily.FLUX1_VARIANT: "a FLUX.1 Fill, Controls or Flex.2 model",
    ImageFamily.CHROMA_RADIANCE: "Chroma1-Radiance",
    ImageFamily.LONGCAT: "LongCat-Image",
    ImageFamily.OVIS_IMAGE: "Ovis-Image",
    ImageFamily.LENS: "Lens",
    ImageFamily.MAGE_FLOW: "Mage-Flow",
    ImageFamily.QWEN_IMAGE_LAYERED: "Qwen-Image-Layered",
    ImageFamily.Z_IMAGE_L2P: "Z-Image L2P",
    ImageFamily.BOOGU: "Boogu Image",
    ImageFamily.PIXART: "PixArt",
    ImageFamily.SEFI: "SeFi-Image",
    ImageFamily.LLADA_IMAGE: "LLaDA-Image",
    ImageFamily.IDEOGRAM4: "Ideogram 4",
    ImageFamily.PID: "PiD (a decoder that needs a reference image)",
    ImageFamily.MINIT2I: "MiniT2I",
    ImageFamily.IRIS: "Iris",
    ImageFamily.WAN: "Wan 2.1",
    ImageFamily.WAN_I2V: "Wan 2.1 I2V",
    ImageFamily.WAN22_TI2V: "Wan 2.2 TI2V 5B",
    ImageFamily.WAN22_A14B: "Wan 2.2 A14B",
    ImageFamily.WAN22_S2V: "Wan 2.2 S2V (speech-to-video)",
    ImageFamily.LINGBOT_VIDEO: "LingBot-Video",
    ImageFamily.HUNYUAN_VIDEO: "HunyuanVideo 1.5",
    ImageFamily.LTX2: "LTX-2",
    ImageFamily.MINIMAX_H3: "MiniMax-H3 (audio-video)",
    ImageFamily.UNKNOWN: "an unrecognised model",
}


class Role(StrEnum):
    """What a companion file is, named by the sd-server flag that loads it."""

    VAE = "vae"
    TEXT_ENCODER = "llm"
    CLIP_L = "clip_l"
    CLIP_G = "clip_g"
    T5XXL = "t5xxl"
    CLIP_VISION = "clip_vision"  # Wan 2.1 I2V: the image's embedding
    HIGH_NOISE = "high-noise-diffusion-model"  # Wan 2.2 A14B: the first-stage DiT
    AUDIO_VAE = "audio-vae"  # LTX-2: the soundtrack's decoder
    EMBEDDINGS_CONNECTORS = "embeddings-connectors"  # LTX-2.3: the text projection


def role_flag(role: Role) -> str:
    """The sd-server flag that loads a companion: the role is named after it."""
    return f"--{role.value}"


# Text encoders may run in RAM (`--backend te=cpu`); the VAE stays on the GPU.
MOVABLE_ROLES = frozenset({Role.TEXT_ENCODER, Role.CLIP_L, Role.CLIP_G, Role.T5XXL})


class Companion(BaseModel):
    role: Role
    repo_id: str
    path: str
    revision: str = "main"

    @property
    def key(self) -> str:
        return f"{self.repo_id}:{self.path}"

    @property
    def summary(self) -> str:
        """`vae ae.safetensors from Comfy-Org/z_image_turbo`: the file says which VAE a shared
        mirror provides (FLUX.1's autoencoder sits in the Z-Image repo). The role is dropped
        when the file name already says it (`clip_l.safetensors`, `flux2-vae.safetensors`)."""
        file = self.path.rsplit("/", 1)[-1]
        named = self.role.value.lower() in file.lower()
        return f"{file if named else f'{self.role.value} {file}'} from {self.repo_id}"


# stable-diffusion.cpp master `sample_method_str[]` / `scheduler_to_str[]` as printed by
# `sd-server --help` on 2026-10-10 (the `master-cuda` image follows master).
Sampler = Literal[
    "euler",
    "euler_a",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "ipndm",
    "ipndm_v",
    "lcm",
    "ddim_trailing",
    "tcd",
    "res_multistep",
    "res_2s",
    "er_sde",
    "euler_cfg_pp",
    "euler_a_cfg_pp",
    "euler_ge",
    "dpm++2m_sde",
    "dpm++2m_sde_bt",
]
Scheduler = Literal[
    "discrete",
    "karras",
    "exponential",
    "ays",
    "gits",
    "sgm_uniform",
    "simple",
    "smoothstep",
    "kl_optimal",
    "lcm",
    "bong_tangent",
    "ltx2",
    "logit_normal",
    "flux2",
    "flux",
    "beta",
    "llada_image",
]


class GenerationDefaults(BaseModel):
    width: int = 1024
    height: int = 1024
    steps: int = 20
    cfg_scale: float = 7.0
    sampler: Sampler = "euler"
    flow_shift: float | None = None


class VideoDefaults(BaseModel):
    """What a video kit adds to the generation defaults: a clip's length and rate, and how
    the family's VAE and patching turn pixels and frames into transformer tokens."""

    frames: int = 33  # (frames - 1) divides by the temporal compression: 33, 49, 81
    fps: int = 16
    spatial_compression: int = 16  # pixels per latent token side (VAE stride x patch size)
    temporal_compression: int = 4  # frames per latent frame (the first frame stands alone)
    # A two-stage family (Wan 2.2 A14B): the repo's high- and low-noise files of one quant
    # are one variant (`hub/video_files.pair_noise_halves`, `GgufVariant.high_noise`), and the
    # high-noise model takes this many steps before the low-noise one takes the profile's.
    high_noise_steps: int | None = None


class ImageKit(BaseModel):
    family: ImageFamily
    label: str
    # Standalone diffusion weights load with --diffusion-model; a full checkpoint (SDXL:
    # UNet + CLIP + VAE in one file) loads with --model.
    main_flag: Literal["--diffusion-model", "--model"] = "--diffusion-model"
    companions: list[Companion] = Field(default_factory=list)
    defaults: GenerationDefaults = Field(default_factory=GenerationDefaults)
    # A distilled family's undistilled build ("…-base-…"): its own schedule (klein base).
    base_defaults: GenerationDefaults | None = None
    # Full size first (quality and balanced), then the speed profile's preview size.
    sizes: tuple[tuple[int, int], ...] = ((1024, 1024), (768, 768))
    min_vram_bytes: int = 0  # below this the family is guidance, not a recipe
    note: str = ""
    # Safetensors checkpoints of this family plan, and a whole one holds this many text
    # encoders (SDXL: CLIP-L and OpenCLIP-G). None: the family's diffusion model runs on its
    # own (GGUF or a single safetensors DiT file) with the kit's companions.
    checkpoint_text_encoders: int | None = None
    # Repos whose files run with this kit, offered when a diffusers folder of the family is
    # pasted (sd.cpp loads only SD 1.x/2.x/SDXL UNets from diffusers folders).
    runs_from: tuple[str, ...] = ()
    # A video kit: the clip's defaults; None for image families.
    video: VideoDefaults | None = None
    # sd.cpp's reference preset for the family (`z_image_omni`, `qwen`, `krea2_ostris_edit`)
    # hands a reference picture to the text encoder as vision input, and the kit's encoder
    # GGUF has no vision part: a repaint segfaulted (exit 139 after
    # `ZImageOmniPipeline`). The router keeps an image instance's picture as the starting
    # image only (`Recipe.drops_references`, `forward.without_references`).
    references_need_vision: bool = False

    @property
    def is_video(self) -> bool:
        """A video family: clips through `vid_gen`, planned with frames (`fit/video.py`)."""
        return self.video is not None

    @property
    def takes_dit_files(self) -> bool:
        """A single safetensors diffusion-model file of this family (ComfyUI's
        `split_files/diffusion_models/…`, a fine-tune) plans with the kit's companions."""
        return self.checkpoint_text_encoders is None and self.main_flag == "--diffusion-model"

    @property
    def brings_vae(self) -> bool:
        """The kit's own VAE replaces the model's (SDXL: the fp16-fix)."""
        return any(c.role is Role.VAE for c in self.companions)
