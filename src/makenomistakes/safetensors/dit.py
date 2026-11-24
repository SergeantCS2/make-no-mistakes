"""Which transformer (DiT) family a single diffusion-model file belongs to, decided the way
stable-diffusion.cpp decides it, so a file never plans with a kit the engine will not load.

`ModelLoader::get_sd_version` (master 2988060, `src/model_loader.cpp:438-712`) walks the
tensor names in sorted order (safetensors headers are parsed into a sorted map), each name
under the prefix `--diffusion-model` gives it (`model.diffusion_model.`), and returns at the
first rule a name meets; FLUX-like files are decided after the walk from the widths of their
text and image inputs. Several families share a marker with another (Lens carries Qwen-Image's
`img_mod`, LongCat and Ovis carry FLUX.1's `double_blocks`), so the order and the extra
conditions matter: a first-match table misfiled five families.
"""

from __future__ import annotations

from makenomistakes.hub.kits import ImageFamily
from makenomistakes.safetensors.reader import TensorInfo

D = "model.diffusion_model."
# Folded into `D` by sd.cpp's `convert_tensors_name` (`name_conversion.cpp:1633-1635`).
_FOLDED = ("diffusion_model.", "transformer.", "unet.")
# The text input width of a FLUX.2 model: the text encoder's hidden size times three.
_FLUX2_TEXT_WIDTHS = {7680: ImageFamily.FLUX2_KLEIN, 12288: ImageFamily.FLUX2_KLEIN_9B}
_LONGCAT_TEXT_WIDTH = 3584
_FLUX1_VARIANT_IMAGE_WIDTHS = frozenset({384, 128, 196})  # Fill, Controls, Flex.2
_MAGE_FLOW_IMAGE_WIDTH = 128
# Video models that end the walk at a name of their own (HunyuanVideo, LingBot); Wan is a
# flag decided after the walk from its patch embedding (`VERSION_WAN2*`), MiniMax-H3 needs
# its audio input as well.
_HUNYUAN_VIDEO = f"{D}txt_in.individual_token_refiner.blocks.0.adaLN_modulation.1.weight"
_LINGBOT = f"{D}patch_embedder.weight"
_WAN = f"{D}blocks.0.cross_attn.norm_k.weight"
_WAN_PATCH = f"{D}patch_embedding.weight"
_WAN_IMG_EMB = f"{D}img_emb."  # Wan 2.1 I2V's CLIP-vision projection
_WAN_S2V = ("casual_audio_encoder.weights", "audio_injector.injector.0.q.weight")
# sd.cpp reads the patch embedding's `ne[3]`, the product of the two leading torch dims
# (output width x input channels): 5120 x 36 for the A14B I2V, 3072 x 48 for the 5B.
_WAN_PATCH_CHANNELS = {184320: ImageFamily.WAN22_A14B, 147456: ImageFamily.WAN22_TI2V}
# Every family `dit_family` can name: a file of one is a diffusion model on its own.
DIT_FAMILIES = frozenset(
    {
        ImageFamily.PID,
        ImageFamily.IDEOGRAM4,
        ImageFamily.KREA2,
        ImageFamily.CHROMA_RADIANCE,
        ImageFamily.SD3,
        ImageFamily.LENS,
        ImageFamily.MINIT2I,
        ImageFamily.QWEN_IMAGE_2_1,
        ImageFamily.MAGE_FLOW,
        ImageFamily.QWEN_IMAGE_LAYERED,
        ImageFamily.QWEN_IMAGE,
        ImageFamily.ANIMA,
        ImageFamily.SEFI,
        ImageFamily.OVIS_IMAGE,
        ImageFamily.LLADA_IMAGE,
        ImageFamily.Z_IMAGE_L2P,
        ImageFamily.Z_IMAGE,
        ImageFamily.BOOGU,
        ImageFamily.ERNIE_IMAGE,
        ImageFamily.PIXART,
        ImageFamily.IRIS,
        ImageFamily.WAN,
        ImageFamily.WAN_I2V,
        ImageFamily.WAN22_TI2V,
        ImageFamily.WAN22_A14B,
        ImageFamily.WAN22_S2V,
        ImageFamily.LINGBOT_VIDEO,
        ImageFamily.HUNYUAN_VIDEO,
        ImageFamily.LTX2,
        ImageFamily.MINIMAX_H3,
        ImageFamily.FLUX2,
        ImageFamily.FLUX2_KLEIN,
        ImageFamily.FLUX2_KLEIN_9B,
        ImageFamily.LONGCAT,
        ImageFamily.FLUX1_VARIANT,
        ImageFamily.CHROMA,
        ImageFamily.FLUX1,
    }
)


def as_loaded(name: str) -> str:
    """A tensor's name as sd.cpp sees it when the file is passed as `--diffusion-model`."""
    if name.startswith(D):
        return name
    for prefix in _FOLDED:
        if name.startswith(prefix):
            return D + name[len(prefix) :]
    return D + name


def dit_family(
    tensors: dict[str, TensorInfo], metadata: dict[str, str] | None = None
) -> ImageFamily:
    """The family, or UNKNOWN when the file is no transformer sd.cpp names."""
    named = {as_loaded(n): t for n, t in tensors.items()}
    flux = flux2 = flux2_dev = wan = False
    for name in sorted(named):
        found = _rule(name, named, metadata or {})
        if found is not None:
            return found
        flux = flux or f"{D}double_blocks." in name or f"{D}single_transformer_blocks." in name
        flux2 = flux2 or f"{D}double_stream_modulation_img.lin.weight" in name
        flux2_dev = flux2_dev or "single_blocks.47.linear1.weight" in name
        wan = wan or _WAN in name
    if wan:
        return _wan_family(named)
    if flux2:
        width = _input_width(named, "txt_in.weight")
        return (
            ImageFamily.FLUX2
            if flux2_dev
            else _FLUX2_TEXT_WIDTHS.get(width or 0, ImageFamily.FLUX2)
        )
    if flux:
        return _flux1_like(named)
    return ImageFamily.UNKNOWN


def _rule(  # noqa: PLR0911, PLR0912 - one return per sd.cpp rule, in its order
    name: str, named: dict[str, TensorInfo], metadata: dict[str, str]
) -> ImageFamily | None:
    if "net.lq_proj.latent_proj.0.weight" in name:
        return ImageFamily.PID
    if "embed_image_indicator.weight" in name:
        return ImageFamily.IDEOGRAM4
    if f"{D}txtfusion.projector.weight" in name or f"{D}text_fusion.projector.weight" in name:
        return ImageFamily.KREA2
    if f"{D}nerf_final_layer_conv." in name:
        return ImageFamily.CHROMA_RADIANCE
    if f"{D}joint_blocks." in name:
        return ImageFamily.SD3
    if f"{D}transformer_blocks.0.attn.norm_added_q.weight" in name and (
        f"{D}transformer_blocks.0.img_mlp.w1.weight" in named
    ):
        return ImageFamily.LENS
    if "net.img_embedder.proj1.weight" in name:
        return ImageFamily.MINIT2I
    if name == f"{D}txt_in.text_norm.weight":
        return ImageFamily.QWEN_IMAGE_2_1
    if f"{D}transformer_blocks.0.img_mod.1.weight" in name:
        if _input_width(named, "img_in.weight") == _MAGE_FLOW_IMAGE_WIDTH:
            return ImageFamily.MAGE_FLOW
        if f"{D}time_text_embed.addition_t_embedding.weight" in named:
            return ImageFamily.QWEN_IMAGE_LAYERED
        return ImageFamily.QWEN_IMAGE
    if "llm_adapter.blocks.0.cross_attn.q_proj.weight" in name:
        return ImageFamily.ANIMA
    if "dual_time_embed.semantic_embedder.linear_1.weight" in name:
        return ImageFamily.SEFI
    if f"{D}double_blocks.0.img_mlp.gate_proj.weight" in name:
        return ImageFamily.OVIS_IMAGE
    if f"{D}sigvq_embedder.1.weight" in name:
        return ImageFamily.LLADA_IMAGE
    if f"{D}cap_embedder.0.weight" in name:
        if f"{D}local_decoder.out_conv.weight" in named:
            return ImageFamily.Z_IMAGE_L2P
        # Ming-Image shares Z-Image's DiT; sd.cpp tells it by its text encoder, the file by
        # the config it carries (`{"transformer": {"image_model": "ming_image"}}`).
        ming = "ming_image" in metadata.get("config", "")
        return ImageFamily.UNKNOWN if ming else ImageFamily.Z_IMAGE
    if "double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight" in name:
        return ImageFamily.BOOGU
    if f"{D}layers.0.adaLN_sa_ln.weight" in name:
        return ImageFamily.ERNIE_IMAGE
    audio = f"{D}audio_patchify_proj.weight" in named  # LTX-AV shares PixArt's embeddings
    if f"{D}t_block.1.weight" in name and f"{D}x_embedder.proj.weight" in named and not audio:
        return ImageFamily.PIXART
    if f"{D}adaln_single.emb.timestep_embedder.linear_1.bias" in name:  # PixArt, or LTX
        pixart = {f"{D}pos_embed.proj.weight", f"{D}adaln_single.linear.weight"} <= set(named)
        return ImageFamily.PIXART if pixart and not audio else ImageFamily.LTX2
    if f"{D}video_patch_proj.weight" in name and f"{D}audio_patch_proj.weight" in named:
        return ImageFamily.MINIMAX_H3
    if name == f"{D}y_embedder.layer_pool.weight" and (
        f"{D}modulation_cores.adaln_img.weight" in named
    ):
        return ImageFamily.IRIS
    if _HUNYUAN_VIDEO in name:
        return ImageFamily.HUNYUAN_VIDEO
    if _LINGBOT in name:
        return ImageFamily.LINGBOT_VIDEO
    return None


def _wan_family(named: dict[str, TensorInfo]) -> ImageFamily:
    """`get_sd_version`'s Wan branch: S2V by its audio names, the Wan 2.2 5B and A14B I2V by
    their patch embedding (and no CLIP-vision projection), Wan 2.1's I2V by that projection,
    the rest (T2V 1.3B/14B, VACE, one half of the A14B T2V pair) as plain Wan."""
    if any(any(marker in name for marker in _WAN_S2V) for name in named):
        return ImageFamily.WAN22_S2V
    img_emb = any(name.startswith(_WAN_IMG_EMB) for name in named)
    patch = named.get(_WAN_PATCH)
    channels = patch.shape[0] * patch.shape[1] if patch is not None and len(patch.shape) > 1 else 0
    if not img_emb and channels in _WAN_PATCH_CHANNELS:
        return _WAN_PATCH_CHANNELS[channels]
    return ImageFamily.WAN_I2V if img_emb else ImageFamily.WAN


def _flux1_like(named: dict[str, TensorInfo]) -> ImageFamily:
    # sd.cpp keeps the last of the two in sorted order: `txt_in` when the file has both.
    text = _input_width(named, "txt_in.weight") or _input_width(named, "context_embedder.weight")
    if text == _LONGCAT_TEXT_WIDTH:
        return ImageFamily.LONGCAT
    if _input_width(named, "img_in.weight") in _FLUX1_VARIANT_IMAGE_WIDTHS:
        return ImageFamily.FLUX1_VARIANT
    # sd.cpp builds Chroma's T5-only conditioner when this tensor is there (`flux.hpp:140`).
    if f"{D}distilled_guidance_layer.in_proj.weight" in named:
        return ImageFamily.CHROMA
    return ImageFamily.FLUX1


def _input_width(named: dict[str, TensorInfo], suffix: str) -> int | None:
    """A linear layer's input width: ggml's `ne[0]`, the last dimension as stored."""
    tensor = named.get(D + suffix)
    return tensor.shape[-1] if tensor is not None and tensor.shape else None
