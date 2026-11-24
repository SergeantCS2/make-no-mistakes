"""Image model kits: what a diffusion model needs besides its own weights.

stable-diffusion.cpp loads a diffusion model plus a VAE and one or more text encoders, and
on the Hub those live in different repositories. The pasted link picks the diffusion model
(and its quant); the kit for its family supplies the companions, each from an ungated
repository, with the engine flag it is passed as. Defaults come from upstream's docs for
the family.
"""

from __future__ import annotations

import re

from makenomistakes.hub.kit_types import (  # re-exported: callers import from hub.kits
    FAMILY_LABELS,
    MOVABLE_ROLES,
    VIDEO_FAMILIES,
    Companion,
    GenerationDefaults,
    ImageFamily,
    ImageKit,
    Role,
    Sampler,
    Scheduler,
    VideoDefaults,
    role_flag,
)

__all__ = [
    "FAMILY_LABELS",
    "KITS",
    "MOVABLE_ROLES",
    "VIDEO_FAMILIES",
    "Companion",
    "GenerationDefaults",
    "ImageFamily",
    "ImageKit",
    "Role",
    "Sampler",
    "Scheduler",
    "VideoDefaults",
    "defaults_for",
    "detect_family",
    "kit_for",
    "role_flag",
]

GIB = 1024**3
_QWEN3_4B = Companion(
    role=Role.TEXT_ENCODER, repo_id="unsloth/Qwen3-4B-GGUF", path="Qwen3-4B-Q4_K_M.gguf"
)
_FLUX1_VAE = Companion(  # the FLUX.1 autoencoder, from an ungated mirror
    role=Role.VAE, repo_id="Comfy-Org/z_image_turbo", path="split_files/vae/ae.safetensors"
)
_FLUX2_VAE = Companion(  # renamed on the Hub from Comfy-Org/flux2-klein-4B (Oct 2026)
    role=Role.VAE,
    repo_id="Comfy-Org/vae-text-encorder-for-flux-klein-4b",
    path="split_files/vae/flux2-vae.safetensors",
)
# The Qwen-Image VAE: the same file (sha a70580f0…) in Anima's, Krea 2's and Qwen-Image's
# repos; Anima's copy, so one download serves both kits.
_QWEN_IMAGE_VAE = Companion(
    role=Role.VAE,
    repo_id="circlestone-labs/Anima",
    path="split_files/vae/qwen_image_vae.safetensors",
)
_T5XXL = Companion(
    role=Role.T5XXL,
    repo_id="city96/t5-v1_1-xxl-encoder-gguf",
    path="t5-v1_1-xxl-encoder-Q4_K_M.gguf",
)
# A distilled family's base build: what its card and ComfyUI's template run.
_KLEIN_BASE = GenerationDefaults(steps=20, cfg_scale=4.0)
# The ESRGAN upscaler an image instance may carry (`mnm add --upscaler`, on by default, 18 MB):
# `POST /sdcpp/v1/upscale` and the engine page's upscale step then work on any picture. Not a
# companion the engine loads at start: a folder-served asset like a LoRA (`Recipe.upscaler`).
UPSCALER_REPO = "ximso/RealESRGAN_x4plus_anime_6B"
UPSCALER_PATH = "RealESRGAN_x4plus_anime_6B.pth"

# Video companions. Wan's text encoder is umT5-XXL (`--t5xxl`), 3.7 GB at Q4_K_M; both Wan
# VAEs sit in ComfyUI's Wan 2.2 repo (the 2.1 one is 254 MB, the 5B's 16x one 1.4 GB).
# Comfy-Org renamed both repos (Wan_2.2_ComfyUI_Repackaged -> Wan-2.2, Wan_2.1_ComfyUI_
# repackaged -> Wan-2.1; seen 2026-10-26). The old names still redirect (307), and a recipe
# saved under one keeps its files there; a new plan names the new repo, whose cache path a
# download then matches (the klein rename's rule).
_WAN_REPACK = "Comfy-Org/Wan-2.2"
_UMT5 = Companion(
    role=Role.T5XXL, repo_id="city96/umt5-xxl-encoder-gguf", path="umt5-xxl-encoder-Q4_K_M.gguf"
)
_WAN21_VAE = Companion(
    role=Role.VAE, repo_id=_WAN_REPACK, path="split_files/vae/wan_2.1_vae.safetensors"
)
_WAN22_VAE = Companion(
    role=Role.VAE, repo_id=_WAN_REPACK, path="split_files/vae/wan2.2_vae.safetensors"
)
_CLIP_VISION_H = Companion(
    role=Role.CLIP_VISION,
    repo_id="Comfy-Org/Wan-2.1",
    path="split_files/clip_vision/clip_vision_h.safetensors",
)
_QWEN3_VL_4B = Companion(
    role=Role.TEXT_ENCODER,
    repo_id="Qwen/Qwen3-VL-4B-Instruct-GGUF",
    path="Qwen3VL-4B-Instruct-Q4_K_M.gguf",
)
_HUNYUAN_REPACK = "Comfy-Org/HunyuanVideo_1.5_repackaged"
_LTX23 = "unsloth/LTX-2.3-GGUF"
# Wan's defaults follow sd.cpp's `docs/wan.md`: 832x480 at 16 fps, 33 frames (2 s), cfg 6,
# euler, flow shift 3. The speed profile previews at 512x288 (sides divide by 16).
_WAN_SIZES = ((832, 480), (512, 288))
_WAN_DEFAULTS = GenerationDefaults(width=832, height=480, steps=20, cfg_scale=6.0, flow_shift=3.0)
# Wan 2.1's VAE is 8x with 2x2 patches (16 px a token); Wan 2.2's 5B VAE is 16x (32 px).
_WAN_VIDEO = VideoDefaults(frames=33, fps=16, spatial_compression=16)
_WAN22_VIDEO = VideoDefaults(frames=33, fps=16, spatial_compression=32)

KITS: dict[ImageFamily, ImageKit] = {
    ImageFamily.Z_IMAGE: ImageKit(
        family=ImageFamily.Z_IMAGE,
        references_need_vision=True,
        label="Z-Image (6B)",
        runs_from=("unsloth/Z-Image-Turbo-GGUF", "Comfy-Org/z_image_turbo"),
        companions=[
            _FLUX1_VAE,
            Companion(
                role=Role.TEXT_ENCODER,
                repo_id="unsloth/Qwen3-4B-Instruct-2507-GGUF",
                path="Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
            ),
        ],
        defaults=GenerationDefaults(steps=20, cfg_scale=5.0),
        note="Turbo builds take 8 steps at cfg 1.0; the base model 20 steps at cfg 5.0.",
    ),
    ImageFamily.FLUX2_KLEIN: ImageKit(
        family=ImageFamily.FLUX2_KLEIN,
        label="FLUX.2 klein (4B)",
        runs_from=("leejet/FLUX.2-klein-4B-GGUF",),
        companions=[_FLUX2_VAE, _QWEN3_4B],
        defaults=GenerationDefaults(steps=4, cfg_scale=1.0),
        base_defaults=_KLEIN_BASE,
        note="Distilled: 4 steps, cfg 1.0. Base builds: 20 steps at cfg 4.",
    ),
    ImageFamily.FLUX2_KLEIN_9B: ImageKit(
        family=ImageFamily.FLUX2_KLEIN_9B,
        label="FLUX.2 klein (9B)",
        runs_from=("unsloth/FLUX.2-klein-9B-GGUF", "leejet/FLUX.2-klein-9B-GGUF"),
        companions=[  # Qwen3-8B: the file a qwen3-8b chat instance already downloaded
            _FLUX2_VAE,
            Companion(
                role=Role.TEXT_ENCODER,
                repo_id="unsloth/Qwen3-8B-GGUF",
                path="Qwen3-8B-Q4_K_M.gguf",
            ),
        ],
        defaults=GenerationDefaults(steps=4, cfg_scale=1.0),
        base_defaults=_KLEIN_BASE,
        note="Distilled: 4 steps, cfg 1.0. Base builds: 20 steps at cfg 4. Non-commercial license.",
    ),
    ImageFamily.FLUX1: ImageKit(
        family=ImageFamily.FLUX1,
        label="FLUX.1 (12B)",
        runs_from=("city96/FLUX.1-schnell-gguf", "city96/FLUX.1-dev-gguf"),
        companions=[
            _FLUX1_VAE,
            Companion(
                role=Role.CLIP_L,
                repo_id="comfyanonymous/flux_text_encoders",
                path="clip_l.safetensors",
            ),
            _T5XXL,
        ],
        defaults=GenerationDefaults(steps=20, cfg_scale=1.0),
        note="dev: 20 steps; schnell: 4 steps. cfg stays 1.0 (guidance-distilled).",
    ),
    ImageFamily.SDXL: ImageKit(
        family=ImageFamily.SDXL,
        label="SDXL",
        main_flag="--model",
        companions=[
            Companion(
                role=Role.VAE, repo_id="madebyollin/sdxl-vae-fp16-fix", path="sdxl_vae.safetensors"
            )
        ],
        defaults=GenerationDefaults(steps=20, cfg_scale=7.0, sampler="euler_a"),
        note="Turbo/Lightning builds: 4-8 steps at cfg 1.0.",
        checkpoint_text_encoders=2,
    ),
    ImageFamily.SD1: ImageKit(
        family=ImageFamily.SD1,
        label="Stable Diffusion 1.x",
        main_flag="--model",  # the checkpoint carries its CLIP and VAE
        defaults=GenerationDefaults(
            width=512, height=512, steps=25, cfg_scale=7.0, sampler="euler_a"
        ),
        sizes=((512, 512), (384, 384)),  # trained at 512; larger sizes repeat subjects
        note="LCM/Hyper builds: 4-8 steps at cfg 1-2.",
        checkpoint_text_encoders=1,
    ),
    ImageFamily.QWEN_IMAGE: ImageKit(
        family=ImageFamily.QWEN_IMAGE,
        references_need_vision=True,
        label="Qwen-Image (20B)",
        companions=[
            Companion(
                role=Role.VAE,
                repo_id="QuantStack/Qwen-Image-GGUF",
                path="VAE/Qwen_Image-VAE.safetensors",
            ),
            Companion(
                role=Role.TEXT_ENCODER,
                repo_id="mradermacher/Qwen2.5-VL-7B-Instruct-GGUF",
                path="Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf",
            ),
        ],
        defaults=GenerationDefaults(steps=20, cfg_scale=2.5, flow_shift=3.0),
        min_vram_bytes=20 * GIB,
        note="20B diffusion weights plus a 7B text encoder",  # read after "needs N GB"
    ),
    ImageFamily.ANIMA: ImageKit(
        family=ImageFamily.ANIMA,
        label="Anima (2B)",
        runs_from=("circlestone-labs/Anima",),
        companions=[  # the publisher's own split files (ComfyUI layout), ungated
            _QWEN_IMAGE_VAE,
            Companion(
                role=Role.TEXT_ENCODER,
                repo_id="circlestone-labs/Anima",
                path="split_files/text_encoders/qwen_3_06b_base.safetensors",
            ),
        ],
        # The model card: 30-50 steps at CFG 4-5, er_sde as the neutral default.
        defaults=GenerationDefaults(steps=30, cfg_scale=4.5, sampler="er_sde"),
        note="Turbo builds take 8-12 steps at cfg 1.0; base and aesthetic 30-50 at cfg 4-5.",
    ),
    ImageFamily.QWEN_IMAGE_2_1: ImageKit(
        family=ImageFamily.QWEN_IMAGE_2_1,
        references_need_vision=True,
        label="Qwen-Image 2.1 (7B)",
        runs_from=("unsloth/Qwen-Image-2.1-GGUF", "leejet/Qwen-Image-2.1-GGUF"),
        companions=[
            Companion(  # its own 64-channel VAE, not the Qwen-Image one
                role=Role.VAE,
                repo_id="Comfy-Org/Qwen-Image-2.1",
                path="vae/qwen_image_2.1_vae_bf16.safetensors",
            ),
            Companion(
                role=Role.TEXT_ENCODER,
                repo_id="Qwen/Qwen3-VL-8B-Instruct-GGUF",
                path="Qwen3VL-8B-Instruct-Q4_K_M.gguf",
            ),
        ],
        # ComfyUI's template: 25 steps at cfg 1 ("the official path; raise cfg only with a
        # negative prompt"); the card runs 40 steps; sd.cpp's own example cfg 6.
        defaults=GenerationDefaults(steps=25, cfg_scale=1.0),
        note="Raise cfg (up to 6) only together with a negative prompt; the card uses 40 "
        "steps. Sides divide by 32; images may carry an alpha channel. Non-commercial license.",
    ),
    ImageFamily.KREA2: ImageKit(
        family=ImageFamily.KREA2,
        references_need_vision=True,
        label="Krea 2 (13B)",
        runs_from=("realrebelai/KREA-2_GGUFs",),
        companions=[
            _QWEN_IMAGE_VAE,
            Companion(
                role=Role.TEXT_ENCODER,
                repo_id="Qwen/Qwen3-VL-4B-Instruct-GGUF",
                path="Qwen3VL-4B-Instruct-Q4_K_M.gguf",
            ),
        ],
        # Raw's own schedule is not published (its card is gated): a starting point. Turbo
        # builds run ComfyUI's 8 steps at cfg 1.
        defaults=GenerationDefaults(steps=28, cfg_scale=4.0),
        note="Turbo builds: 8 steps at cfg 1.0. Raw/Base: 28 steps at cfg 4 is a starting "
        "point (unpublished). Krea 2 community license.",
    ),
    ImageFamily.ERNIE_IMAGE: ImageKit(
        family=ImageFamily.ERNIE_IMAGE,
        label="ERNIE-Image (8B)",
        runs_from=("unsloth/ERNIE-Image-Turbo-GGUF", "unsloth/ERNIE-Image-GGUF"),
        companions=[
            _FLUX2_VAE,
            Companion(
                role=Role.TEXT_ENCODER,
                repo_id="unsloth/Ministral-3-3B-Instruct-2512-GGUF",
                path="Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
            ),
        ],
        defaults=GenerationDefaults(steps=20, cfg_scale=4.0),  # ComfyUI's; the card: 50
        note="Turbo builds: 8 steps at cfg 1.0; base 20-50 steps at cfg 4.",
    ),
    ImageFamily.CHROMA: ImageKit(
        family=ImageFamily.CHROMA,
        label="Chroma (8.9B)",
        runs_from=("silveroxides/Chroma1-HD-GGUF", "silveroxides/Chroma-GGUF"),
        companions=[_FLUX1_VAE, _T5XXL],  # FLUX.1's, without CLIP-L
        # sd.cpp's example; real CFG (two passes per step), so a negative prompt counts.
        defaults=GenerationDefaults(steps=26, cfg_scale=4.0),
        note="Real CFG: a negative prompt helps; the card runs 40 steps at cfg 3.",
    ),
    # Video kits (docs/GUIDE.md §9). Every one runs on the same engine's `vid_gen` route.
    ImageFamily.WAN: ImageKit(
        family=ImageFamily.WAN,
        label="Wan 2.1 (1.3B / 14B)",
        runs_from=("calcuis/wan-1.3b-gguf", "city96/Wan2.1-T2V-14B-gguf"),
        companions=[_WAN21_VAE, _UMT5],
        defaults=_WAN_DEFAULTS,
        sizes=_WAN_SIZES,
        video=_WAN_VIDEO,
        note="Text-to-video; `init_image` in a request makes the 1.3B's VACE builds and the "
        "14B animate a picture. Frame counts 33, 49, 81 land on whole latent frames.",
    ),
    ImageFamily.WAN_I2V: ImageKit(
        family=ImageFamily.WAN_I2V,
        label="Wan 2.1 I2V (14B)",
        runs_from=("city96/Wan2.1-I2V-14B-480P-gguf",),
        companions=[_WAN21_VAE, _UMT5, _CLIP_VISION_H],
        defaults=_WAN_DEFAULTS,
        sizes=_WAN_SIZES,
        video=_WAN_VIDEO,
        note="Image-to-video: every request needs an `init_image`. 14B weights stream from RAM "
        "on 12 GB.",
    ),
    ImageFamily.WAN22_TI2V: ImageKit(
        family=ImageFamily.WAN22_TI2V,
        label="Wan 2.2 TI2V (5B)",
        runs_from=("QuantStack/Wan2.2-TI2V-5B-GGUF",),
        companions=[_WAN22_VAE, _UMT5],
        defaults=_WAN_DEFAULTS,
        sizes=_WAN_SIZES,
        video=_WAN22_VIDEO,
        note="Text or image to video in one model: send `init_image` for the latter. Its 16x "
        "VAE makes 832x480 cheap; the card's 1280x704 needs more than 12 GB.",
    ),
    ImageFamily.WAN22_A14B: ImageKit(
        family=ImageFamily.WAN22_A14B,
        label="Wan 2.2 A14B (two 14B stages)",
        runs_from=("QuantStack/Wan2.2-T2V-A14B-GGUF", "QuantStack/Wan2.2-I2V-A14B-GGUF"),
        companions=[_WAN21_VAE, _UMT5],
        defaults=GenerationDefaults(width=832, height=480, steps=10, cfg_scale=3.5, flow_shift=3.0),
        sizes=_WAN_SIZES,
        video=VideoDefaults(frames=33, fps=16, spatial_compression=16, high_noise_steps=8),
        note="A high-noise and a low-noise 14B model (the same quant of each) run in turn: 8 "
        "high-noise steps, then 10 at cfg 3.5. Both stream from RAM; RAM must hold both.",
    ),
    ImageFamily.LINGBOT_VIDEO: ImageKit(
        family=ImageFamily.LINGBOT_VIDEO,
        label="LingBot-Video (1.3B)",
        runs_from=(
            "robbyant/lingbot-video-dense-1.3b/transformer/diffusion_pytorch_model.safetensors",
        ),
        companions=[_WAN21_VAE, _QWEN3_VL_4B],
        defaults=GenerationDefaults(width=832, height=480, steps=20, cfg_scale=3.0),
        sizes=_WAN_SIZES,
        video=_WAN_VIDEO,
        note="Prompts are JSON captions (its card shows the shape); plain text works, less "
        "well. `init_image` makes it image-to-video.",
    ),
    ImageFamily.HUNYUAN_VIDEO: ImageKit(
        family=ImageFamily.HUNYUAN_VIDEO,
        label="HunyuanVideo 1.5 (8B)",
        runs_from=(_HUNYUAN_REPACK,),
        companions=[
            Companion(
                role=Role.VAE,
                repo_id=_HUNYUAN_REPACK,
                path="split_files/vae/hunyuanvideo15_vae_fp16.safetensors",
            ),
            Companion(
                role=Role.TEXT_ENCODER,
                repo_id="mradermacher/Qwen2.5-VL-7B-Instruct-GGUF",
                path="Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf",
            ),
            Companion(
                role=Role.T5XXL,
                repo_id=_HUNYUAN_REPACK,
                path="split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors",
            ),
        ],
        defaults=GenerationDefaults(width=832, height=480, steps=20, cfg_scale=6.0),
        sizes=_WAN_SIZES,
        # A 16x causal VAE with 2x2 patches, 4 frames a latent frame; 24 fps.
        video=VideoDefaults(frames=33, fps=24, spatial_compression=32),
        note="480p and 720p builds, text-to-video and image-to-video each; the FP8 files load "
        "as Q8_0. Weights and the 7B text encoder stream from RAM on 12 GB.",
    ),
    ImageFamily.LTX2: ImageKit(
        family=ImageFamily.LTX2,
        label="LTX-2.3 (22B, video with sound)",
        runs_from=(_LTX23,),
        companions=[
            Companion(
                role=Role.VAE, repo_id=_LTX23, path="vae/ltx-2.3-22b-dev_video_vae.safetensors"
            ),
            Companion(
                role=Role.AUDIO_VAE,
                repo_id=_LTX23,
                path="vae/ltx-2.3-22b-dev_audio_vae.safetensors",
            ),
            Companion(
                role=Role.TEXT_ENCODER,
                repo_id="unsloth/gemma-3-12b-it-GGUF",
                path="gemma-3-12b-it-Q4_K_M.gguf",
            ),
            Companion(
                role=Role.EMBEDDINGS_CONNECTORS,
                repo_id=_LTX23,
                path="text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors",
            ),
        ],
        defaults=GenerationDefaults(width=768, height=512, steps=20, cfg_scale=6.0),
        sizes=((768, 512), (512, 320)),  # sides divide by 32
        # A 32x VAE, 8 frames a latent frame, 24 fps; its soundtrack decodes alongside.
        video=VideoDefaults(frames=33, fps=24, spatial_compression=32, temporal_compression=8),
        min_vram_bytes=0,
        note="22B: on 12 GB only streamed from RAM, and about 20 GB of RAM holds no quant "
        "beside its 12B text encoder; the plan says so. The audio track is part of every clip.",
    ),
}

# Fill, Redux and the control variants of FLUX.1 are other models for other inputs (a Fill
# model's masked channels, control images), so they are recognised as unsupported rather than
# misfiled as text-to-image; any image model inpaints through `/v1/images/edits` instead.
# Kontext takes a reference image through that route and plans with the FLUX.1 kit.
# An LTX-2 release other than 2.3 (2.5 …) has its own VAE, connectors and text encoder.
_OTHER_LTX_RE = re.compile(r"(?<![a-z])ltx[-_ ]?2[._](?!3(?!\d))\d", re.I)
_FLUX1_EDIT_RE = re.compile(r"(?<![a-z])fill(?![a-z])|redux|depth|canny", re.I)
_KONTEXT_RE = re.compile(r"kontext", re.I)
# A variant's own rule comes before its family's (L2P before Z-Image, 2.1 before Qwen-Image).
_FAMILY_PATTERNS: tuple[tuple[re.Pattern[str], ImageFamily], ...] = (
    # Video (the variant's own rule before the family's: the 5B and the A14B pair before Wan).
    (re.compile(r"wan[-_ .]?2[._]?2[-_ .]?(ti2v|5b)|ti2v[-_]?5b", re.I), ImageFamily.WAN22_TI2V),
    (re.compile(r"wan[-_ .]?2[._]?2.*s2v|s2v[-_]?14b", re.I), ImageFamily.WAN22_S2V),
    (
        re.compile(r"wan[-_ .]?2[._]?2.*(a14b|high[-_]?noise|low[-_]?noise|14b)", re.I),
        ImageFamily.WAN22_A14B,
    ),
    (re.compile(r"wan[-_ .]?2[._]?1.*(i2v|flf2v)", re.I), ImageFamily.WAN_I2V),
    (
        re.compile(r"(?<![a-z])wan(?:[-_ .]?2[._]?[12]|[-_](?:t2v|i2v|vace|1\.3b|14b))", re.I),
        ImageFamily.WAN,
    ),
    (re.compile(r"lingbot", re.I), ImageFamily.LINGBOT_VIDEO),
    (re.compile(r"hunyuan[-_ ]?video", re.I), ImageFamily.HUNYUAN_VIDEO),
    (_OTHER_LTX_RE, ImageFamily.UNKNOWN),  # its companions are not LTX-2.3's
    (re.compile(r"(?<![a-z])ltx[-_ ]?2", re.I), ImageFamily.LTX2),
    (re.compile(r"minimax[-_ ]?h3", re.I), ImageFamily.MINIMAX_H3),
    (re.compile(r"(?<![a-z])z[-_ ]?image[-_ ]?l2p", re.I), ImageFamily.Z_IMAGE_L2P),
    (re.compile(r"(?<![a-z])z[-_ ]?image", re.I), ImageFamily.Z_IMAGE),  # not "quartz-image"
    # klein 9B pairs with another text encoder than 4B
    (
        re.compile(r"(?=.*(?:flux[-_.]?2[-_]?klein|flux2[-_]?klein)).*(?<![0-9])9b", re.I),
        ImageFamily.FLUX2_KLEIN_9B,
    ),
    (re.compile(r"flux[-_.]?2[-_]?klein|flux2[-_]?klein", re.I), ImageFamily.FLUX2_KLEIN),
    (
        re.compile(rf"(?=.*(?:flux[-_.]?1|flux1[-_])).*(?:{_FLUX1_EDIT_RE.pattern})", re.I),
        ImageFamily.UNKNOWN,
    ),
    (re.compile(r"flux[-_.]?1|flux1[-_]", re.I), ImageFamily.FLUX1),
    (re.compile(r"sdxl|sd[-_]xl|stable[-_]diffusion[-_]xl", re.I), ImageFamily.SDXL),
    (re.compile(r"qwen[-_]?image[-_ ]?2[._]1", re.I), ImageFamily.QWEN_IMAGE_2_1),
    (re.compile(r"qwen[-_]?image[-_]?layered", re.I), ImageFamily.QWEN_IMAGE_LAYERED),
    (re.compile(r"qwen[-_]?image[-_]?edit", re.I), ImageFamily.UNKNOWN),  # needs an image
    (re.compile(r"qwen[-_]?image", re.I), ImageFamily.QWEN_IMAGE),
    (re.compile(r"krea[-_ ]?2", re.I), ImageFamily.KREA2),
    (re.compile(r"ernie[-_ ]?image", re.I), ImageFamily.ERNIE_IMAGE),
    (re.compile(r"chroma1?[-_ ]?radiance", re.I), ImageFamily.CHROMA_RADIANCE),
    (re.compile(r"(?<![a-z])chroma(?!db)", re.I), ImageFamily.CHROMA),
    # Not Animagine (an SDXL fine-tune) or AnimateDiff: "anima" as a word of its own.
    (re.compile(r"(?<![a-z])anima(?![a-z])", re.I), ImageFamily.ANIMA),
)
_TURBO_RE = re.compile(r"turbo|schnell|lightning", re.I)
_BASE_RE = re.compile(r"(?<![a-z])base(?![a-z])", re.I)


def detect_family(repo_id: str, filenames: tuple[str, ...] = ()) -> ImageFamily:
    """Family from the repo name first, then from the GGUF file names."""
    for text in (repo_id, *filenames):
        for pattern, family in _FAMILY_PATTERNS:
            if pattern.search(text):
                return family
    return ImageFamily.UNKNOWN


def kit_for(family: ImageFamily) -> ImageKit | None:
    return KITS.get(family)


def defaults_for(repo_id: str, kit: ImageKit) -> GenerationDefaults:
    """The kit's defaults, with the few-step/cfg-1 schedule for distilled builds, and a
    distilled family's own schedule for its base build."""
    if kit.base_defaults is not None and _BASE_RE.search(repo_id):
        return kit.base_defaults
    if _TURBO_RE.search(repo_id):
        steps = 4 if kit.family in (ImageFamily.FLUX1, ImageFamily.SDXL) else 8
        return kit.defaults.model_copy(update={"steps": steps, "cfg_scale": 1.0})
    if kit.family is ImageFamily.FLUX1 and _KONTEXT_RE.search(repo_id):
        return kit.defaults.model_copy(update={"steps": 20, "cfg_scale": 1.0})  # docs/kontext.md
    return kit.defaults
