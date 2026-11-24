"""What a safetensors header says about a Stable Diffusion model or a LoRA.

The UNet's cross-attention width names the family whatever the file layout: every
`attn2.to_k` weight takes the text encoder's output, 768 wide for SD 1.x, 1024 for SD 2.x,
2048 for SDXL (CLIP-L + OpenCLIP-G; Illustrious, Pony and NoobAI are SDXL fine-tunes) and
1280 for the SDXL refiner. A LoRA's `attn2_to_k` down-projection has the same input
width, so a LoRA is matched to a checkpoint by the same number. Transformer (DiT) models are
recognised the way stable-diffusion.cpp itself names them (`safetensors/dit.py`),
so a single diffusion-model file runs with its family's kit.

Sizes are counted as loaded. A UNet checkpoint's float tensors take two bytes each, since
sd.cpp runs it F16 and converts F32 or BF16 with `--type f16`; a DiT file loads as stored
(sd.cpp keeps BF16 and FP8), unless it is mostly F32 (then `--type bf16`).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Literal

from pydantic import BaseModel

from makenomistakes.hub.kits import FAMILY_LABELS, ImageFamily
from makenomistakes.safetensors.dit import DIT_FAMILIES, dit_family
from makenomistakes.safetensors.reader import SafetensorsHeader, TensorInfo

_FLOATS = frozenset({"F32", "F16", "BF16"})
_CONTEXT_WIDTH: dict[int, ImageFamily] = {
    768: ImageFamily.SD1,
    1024: ImageFamily.SD2,
    2048: ImageFamily.SDXL,
    1280: ImageFamily.SDXL_REFINER,
}
# FLUX-shaped LoRAs (`double_blocks`, `single_blocks`, diffusers' `single_transformer_blocks`):
# each base's hidden width (the input of a block's down projections) and its single blocks'
# fused `linear1` (qkv + MLP in) as a multiple of it. From the recorded headers; klein 9B read
# from unsloth's GGUF.
_FLUX_GEOMETRY = {
    ImageFamily.FLUX1: (3072, 7),
    ImageFamily.CHROMA: (3072, 7),
    ImageFamily.FLUX2_KLEIN: (3072, 9),
    ImageFamily.FLUX2_KLEIN_9B: (4096, 9),
    ImageFamily.FLUX2: (6144, 9),
}
_FLUX = frozenset(_FLUX_GEOMETRY)
_FLUX_BLOCK_RE = re.compile(r"(?:double|single)_(?:transformer_)?blocks")
# Besides the diffusion model, an all-in-one file carries these (ComfyUI and Stability names).
_BUNDLED_PREFIXES = (
    "text_encoders.",
    "cond_stage_model.",
    "conditioner.",
    "first_stage_model.",
    "vae.",
)
# Single-file checkpoints (the original Stability layout).
_DIFFUSION = "model.diffusion_model."
_SD1_TEXT_ENCODER = "cond_stage_model."
_SDXL_TEXT_ENCODERS = "conditioner.embedders."  # .0. CLIP-L, .1. OpenCLIP-G
_VAE = "first_stage_model."
_STABILITY_PREFIXES = (_DIFFUSION, _SD1_TEXT_ENCODER, _SDXL_TEXT_ENCODERS, _VAE)


F8_DTYPES = frozenset({"F8_E4M3", "F8_E5M2"})  # sd.cpp widens these to F16 while loading


Q8_0_BYTES_PER_WEIGHT = 34 / 32  # a 32-weight block: 32 int8 values and one F16 scale


class ImageParts(BaseModel, ABC):
    """A model's weights, split the way the planner places them: a UNet checkpoint or
    diffusers folder (`UNetParts`), or a transformer's diffusion model on its own (`DitParts`,
    the kit brings the rest). Each answers for itself how it loads, so no caller branches on
    which it is to size or describe it."""

    family: ImageFamily  # planned when a checkpoint kit exists for it; named otherwise
    diffusion_bytes: int  # the UNet, as loaded (F16); a DiT's diffusion model as stored
    text_encoder_bytes: int  # CLIP-L (+ OpenCLIP-G), as loaded
    text_encoders: int  # how many: 1 for SD 1.x, 2 for SDXL when the model is whole
    vae_bytes: int  # the VAE baked into the file (0 when it has none)
    dtype: str  # the diffusion weights' stored type: "F16", "F32", "BF16", …
    float_dtypes: frozenset[str] = frozenset()  # every float type stored, all parts

    @property
    def label(self) -> str:
        return FAMILY_LABELS[self.family]

    @property
    def load_type(self) -> Literal["f16"] | None:
        """The `--type` the whole model loads with (None: as stored)."""
        return None

    @property
    def diffusion_type(self) -> Literal["bf16"] | None:
        """The type the diffusion model alone is converted to while loading (None: none)."""
        return None

    def loaded_diffusion_bytes(self, diffusion_type: Literal["bf16", "q8_0"] | None) -> int:
        """The diffusion model as it sits in memory under that conversion."""
        del diffusion_type
        return self.diffusion_bytes

    @abstractmethod
    def describe(self, file: str, layout: str) -> str:
        """`SDXL checkpoint x.safetensors (F16: UNet 4.8 GiB, text encoders 1.6 GiB)`."""

    def missing(self, needed: int, vae_supplied: bool) -> list[str]:
        """What a whole model lacks, given how many text encoders its family has (`needed`)
        and whether the kit brings a VAE."""
        missing = [] if self.diffusion_bytes else ["the UNet"]
        if self.text_encoders < needed:
            missing.append(
                "its text encoders"
                if not self.text_encoders
                else f"{needed - self.text_encoders} of its {needed} text encoders"
            )
        if not self.vae_bytes and not vae_supplied:
            missing.append("a VAE")
        return missing


class UNetParts(ImageParts):
    """A single-file checkpoint or a diffusers folder: UNet, text encoders and VAE."""

    @property
    def needs_f16(self) -> bool:
        """Any F32 or BF16 weights load as F16 (`--type f16`): half the memory of F32, the
        type sd.cpp's CUDA kernels are tested with, and what `loaded_bytes` counts."""
        return any(dtype != "F16" for dtype in self.float_dtypes)

    @property
    def load_type(self) -> Literal["f16"] | None:
        return "f16" if self.needs_f16 else None

    def describe(self, file: str, layout: str) -> str:
        stored = (  # a checkpoint with F8 or BF16 parts is widened to F16 while loading
            self.dtype
            if not self.needs_f16
            else f"{'/'.join(sorted(self.float_dtypes))}, loads as F16"
        )
        what = "diffusers folder" if layout == "diffusers" else f"checkpoint {file}"
        split = (
            f"UNet {self.diffusion_bytes / 2**30:.1f} GiB, text encoders "
            f"{self.text_encoder_bytes / 2**30:.1f} GiB"
        )
        return f"{self.label} {what} ({stored}: {split})"


class DitParts(ImageParts):
    """A transformer's diffusion model in one file, loaded as stored (BF16 and FP8 stay as
    they are; a mostly-F32 file loads as BF16), or converted to Q8_0 while loading."""

    diffusion_params: int  # its weight count, for load-time quantization and time/image
    bundled: bool = False  # the file also carries text encoders or a VAE

    @property
    def diffusion_type(self) -> Literal["bf16"] | None:
        """A DiT stored mostly as F32 loads its diffusion model as BF16 (F16 can overflow a
        DiT); everything else loads as stored."""
        return "bf16" if self.dtype == "F32" else None

    @property
    def loads_as(self) -> Literal["F16", "BF16"] | None:
        """The type its diffusion model has in memory when it differs from the file: BF16
        for an F32 file (`diffusion_type`), F16 for FP8 (ggml has no FP8 matmul:
        `model_loader.cpp` widens every FP8 tensor while loading). None: as stored."""
        if self.diffusion_type == "bf16":
            return "BF16"
        if self.dtype in F8_DTYPES:
            return "F16"
        return None

    def loaded_diffusion_bytes(self, diffusion_type: Literal["bf16", "q8_0"] | None) -> int:
        """As stored, or converted while loading (a CPU build loaded Anima's 3,989 MB of BF16
        as 2,136 MB of Q8_0, ratio 0.535)."""
        if diffusion_type == "q8_0":
            return int(self.diffusion_params * Q8_0_BYTES_PER_WEIGHT)
        if diffusion_type == "bf16" or self.loads_as is not None:  # BF16, or FP8 widened
            return self.diffusion_params * 2
        return self.diffusion_bytes

    def describe(self, file: str, layout: str) -> str:
        del layout
        stored = self.dtype + (f", loads as {self.loads_as}" if self.loads_as else "")
        return (
            f"{self.label} diffusion model {file} ({stored}: "
            f"{self.diffusion_bytes / 2**30:.1f} GiB)"
        )


class LoraInfo(BaseModel):
    families: frozenset[ImageFamily]  # the bases it can apply to; empty: could not tell
    label: str  # "SDXL", "FLUX", "an unrecognised model", for messages
    tensors: int


def loaded_bytes(tensor: TensorInfo) -> int:
    return tensor.elements * 2 if tensor.dtype in _FLOATS else tensor.nbytes


AnyParts = UNetParts | DitParts  # what a plan holds: one or the other, never the base


def checkpoint_parts(header: SafetensorsHeader) -> AnyParts:
    """A single-file checkpoint: UNet, text encoders and VAE under their Stability prefixes.
    A file with none of those prefixes is one component on its own (a bare UNet, a
    ControlNet), counted as diffusion weights so the missing parts can be named. A
    transformer family's file is a diffusion model on its own (`dit_parts`)."""
    tensors = header.tensors
    family = _family(tensors, header.metadata)
    if family in DIT_FAMILIES:
        return dit_parts(header, family)
    bare = not any(n.startswith(_STABILITY_PREFIXES) for n in tensors)
    diffusion = [t for n, t in tensors.items() if bare or n.startswith(_DIFFUSION)]
    sd1_encoder = [t for n, t in tensors.items() if n.startswith(_SD1_TEXT_ENCODER)]
    sdxl_encoders = {n: t for n, t in tensors.items() if n.startswith(_SDXL_TEXT_ENCODERS)}
    indices = {n[len(_SDXL_TEXT_ENCODERS) :].split(".", 1)[0] for n in sdxl_encoders}
    return UNetParts(
        family=_family(tensors),
        diffusion_bytes=_sum(diffusion),
        text_encoder_bytes=_sum(sd1_encoder) + _sum(sdxl_encoders.values()),
        text_encoders=len(indices) + (1 if sd1_encoder else 0),
        vae_bytes=_sum(t for n, t in tensors.items() if n.startswith(_VAE)),
        dtype=_dominant(diffusion),
        float_dtypes=_float_dtypes(tensors.values()),
    )


def dit_parts(header: SafetensorsHeader, family: ImageFamily) -> DitParts:
    """A transformer's diffusion model in one file, loaded as stored (BF16 and FP8 stay as
    they are; a mostly-F32 file loads as BF16). An all-in-one file that also carries text
    encoders or a VAE is `bundled`: the plan asks for the diffusion model alone."""
    tensors = header.tensors
    bundled = [n for n in tensors if n.startswith(_BUNDLED_PREFIXES)]
    prefixed = any(n.startswith(_DIFFUSION) for n in tensors)
    diffusion = [
        t
        for n, t in tensors.items()
        if (n.startswith(_DIFFUSION) if prefixed else not n.startswith(_BUNDLED_PREFIXES))
    ]
    dtype = _dominant(diffusion)
    params = sum(t.elements for t in diffusion)
    return DitParts(
        family=family,
        diffusion_bytes=params * 2 if dtype == "F32" else sum(t.nbytes for t in diffusion),
        text_encoder_bytes=0,
        text_encoders=0,
        vae_bytes=0,
        dtype=dtype,
        float_dtypes=_float_dtypes(diffusion),
        diffusion_params=params,
        bundled=bool(bundled),
    )


def diffusers_parts(
    unet: SafetensorsHeader,
    text_encoders: list[SafetensorsHeader],
    vae: SafetensorsHeader | None,
) -> AnyParts:
    """A diffusers folder: one file per component, diffusers tensor names inside each."""
    every = [unet, *text_encoders, *([vae] if vae is not None else [])]
    return UNetParts(
        family=_family(unet.tensors),
        diffusion_bytes=_sum(unet.tensors.values()),
        text_encoder_bytes=sum(_sum(h.tensors.values()) for h in text_encoders),
        text_encoders=len(text_encoders),
        vae_bytes=_sum(vae.tensors.values()) if vae is not None else 0,
        dtype=_dominant(unet.tensors.values()),
        float_dtypes=_float_dtypes(t for h in every for t in h.tensors.values()),
    )


def image_parts(
    layout: Literal["file", "diffusers"],
    files: list[str],
    header_of: Callable[[str], SafetensorsHeader],
) -> AnyParts:
    """The split of a checkpoint file or a diffusers folder's components. `header_of` reads
    one file's header: over HTTP when planning, from disk for Fits now."""
    if layout == "file":
        return checkpoint_parts(header_of(files[0]))
    by_component = {f.split("/")[-2]: f for f in files}
    return diffusers_parts(
        header_of(by_component["unet"]),
        [
            header_of(by_component[c])
            for c in ("text_encoder", "text_encoder_2")
            if c in by_component
        ],
        header_of(by_component["vae"]) if "vae" in by_component else None,
    )


def lora_info(header: SafetensorsHeader) -> LoraInfo | None:
    """None when the file holds no LoRA tensors at all (a checkpoint, an embedding, …)."""
    lora = [n for n in header.tensors if _is_lora_tensor(n)]
    if not lora:
        return None
    for name in lora:
        if "attn2" in name and "to_k" in name and _is_down_projection(name):
            shape = header.tensors[name].shape
            if len(shape) >= 2 and shape[1] in _CONTEXT_WIDTH:
                family = _CONTEXT_WIDTH[shape[1]]
                return LoraInfo(
                    families=frozenset({family}), label=FAMILY_LABELS[family], tensors=len(lora)
                )
    if any(_FLUX_BLOCK_RE.search(n) for n in lora):
        families = _flux_lora_families(header, lora)
        label = FAMILY_LABELS[next(iter(families))] if len(families) == 1 else "FLUX"
        return LoraInfo(families=families, label=label, tensors=len(lora))
    architecture = header.metadata.get("modelspec.architecture", "").lower()
    if "anima" in architecture or any(_ANIMA_LORA_RE.search(n) for n in lora):
        return LoraInfo(
            families=frozenset({ImageFamily.ANIMA}),
            label=FAMILY_LABELS[ImageFamily.ANIMA],
            tensors=len(lora),
        )
    return LoraInfo(
        families=frozenset(), label=FAMILY_LABELS[ImageFamily.UNKNOWN], tensors=len(lora)
    )


def _flux_lora_families(header: SafetensorsHeader, lora: list[str]) -> frozenset[ImageFamily]:
    """The FLUX-shaped bases a LoRA's widths allow (every one when they say nothing)."""
    widths = Counter(
        header.tensors[n].shape[1]
        for n in lora
        if _FLUX_BLOCK_RE.search(n) and _is_down_projection(n) and len(header.tensors[n].shape) == 2
    )
    if not widths:
        return _FLUX
    width = widths.most_common(1)[0][0]
    fused = [
        header.tensors[n].shape[0]
        for n in lora
        if "linear1" in n and _is_up_projection(n) and header.tensors[n].shape
    ]
    return frozenset(
        family
        for family, (hidden, times) in _FLUX_GEOMETRY.items()
        if hidden == width and (not fused or fused[0] == hidden * times)
    )


def family_of(header: SafetensorsHeader) -> ImageFamily:
    """The family a checkpoint or UNet file belongs to (UNKNOWN when nothing says)."""
    return _family(header.tensors, header.metadata)


def _family(tensors: dict[str, TensorInfo], metadata: dict[str, str] | None = None) -> ImageFamily:
    """A UNet's family from its cross-attention width, else the DiT family sd.cpp names."""
    for name, tensor in tensors.items():
        if name.endswith("attn2.to_k.weight") and len(tensor.shape) == 2:
            return _CONTEXT_WIDTH.get(tensor.shape[1], ImageFamily.UNKNOWN)
    return dit_family(tensors, metadata)


# Anima (Cosmos) LoRAs, by Cosmos's own layer names: attention projections `q_proj` …
# `output_proj` and `mlp.layer1`/`layer2`, under kohya's `lora_unet_blocks_<n>_` or `net.blocks.`,
# or the `llm_adapter`. kohya names Wan's blocks the same way, but Wan's layers are `self_attn_q`
# and `ffn_0`, so only these names say Anima (else a Wan LoRA is refused as Anima's).
_ANIMA_LORA_RE = re.compile(
    r"blocks[._]\d+[._](?:(?:self|cross)_attn[._](?:q|k|v|output)_proj|mlp[._]layer[12])"
    r"|llm_adapter"
)


def _is_lora_tensor(name: str) -> bool:
    lowered = name.lower()
    return any(tag in lowered for tag in ("lora", "hada_", "lokr_"))


def _is_down_projection(name: str) -> bool:
    return any(tag in name for tag in ("lora_down", "lora_A", "lora.down"))


def _is_up_projection(name: str) -> bool:
    return any(tag in name for tag in ("lora_up", "lora_B", "lora.up"))


def _sum(tensors: Iterable[TensorInfo]) -> int:
    return sum(loaded_bytes(t) for t in tensors)


def _float_dtypes(tensors: Iterable[TensorInfo]) -> frozenset[str]:
    return frozenset(t.dtype for t in tensors if t.dtype in _FLOATS)


def _dominant(tensors: Iterable[TensorInfo]) -> str:
    """The type most of the bytes are stored as ("F16" for an empty set)."""
    weights: Counter[str] = Counter()
    for tensor in tensors:
        weights[tensor.dtype] += tensor.nbytes
    return weights.most_common(1)[0][0] if weights else "F16"
