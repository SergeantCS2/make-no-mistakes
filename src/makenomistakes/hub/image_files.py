"""Image weights in a repo tree: checkpoint files (one per version) and diffusers folders.

A checkpoint is one large `.safetensors` file holding the UNet, text encoders and VAE (the
original Stability layout, and how fine-tunes such as waiIllustriousSDXL are published), or
a transformer's diffusion model on its own (ComfyUI's `split_files/diffusion_models/…`);
a repo may hold several versions side by side, at any depth. A diffusers folder spreads the
parts over `unet/`, `text_encoder/`, `text_encoder_2/` and `vae/`. stable-diffusion.cpp
loads both; which family a set of weights belongs to is read from its header at plan time.
Files inside a component folder (a diffusers part, or ComfyUI's `text_encoders/`, `vae/`,
`loras/`) are never versions of their own.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from makenomistakes.hub.client import TreeEntry
from makenomistakes.hub.pipelines import PIPELINE_INDEX_FILES
from makenomistakes.hub.variants import GgufVariant
from makenomistakes.hub.video_files import pair_by_noise

_VERSION_RE = re.compile(r"[vV](\d+(?:\.\d+)*)")
CHECKPOINT_MIN_BYTES = 2**30  # LoRAs, embeddings and VAEs are smaller than any checkpoint
# Microscaling formats sd.cpp has no code for: ComfyUI ships them beside bf16 and int8 builds
# of one model, so they are never offered as a version.
_UNREADABLE_FORMAT_RE = re.compile(r"nvfp4|mxfp[48]", re.I)
# A diffusers folder: these components, the fp16 variant preferred (what sd.cpp loads).
_DIFFUSERS_COMPONENTS = (
    ("unet", "diffusion_pytorch_model", True),
    ("text_encoder", "model", True),
    ("text_encoder_2", "model", False),
    ("vae", "diffusion_pytorch_model", False),
)
_COMPONENT_DIRS = frozenset(
    {
        "unet",
        "vae",
        "text_encoder",
        "text_encoder_2",
        "text_encoder_3",
        "transformer",
        "tokenizer",
        "tokenizer_2",
        "scheduler",
        "safety_checker",
        "feature_extractor",
        "image_encoder",
        "text_conditioner",  # Anima's adapter in its diffusers layout
        "llm_adapter",
        # ComfyUI's model folders that hold companions, not diffusion models.
        "text_encoders",
        "clip",
        "clip_vision",
        "loras",
        "lora",
        "controlnet",
        "upscale_models",
        "embeddings",
        "audio_encoders",  # Wan 2.2 S2V's wav2vec2: 2 GiB, never a version
        "latent_upscale_models",
        "model_patches",
        "style_models",
        "vae_approx",
    }
)


class ImageWeights(BaseModel):
    """One servable set of image weights: a single checkpoint file (one version of a
    fine-tune) or a diffusers folder (UNet, text encoders and VAE in their own files)."""

    label: str  # the file's stem, or "diffusers" (with the folder when it is not the root)
    layout: Literal["file", "diffusers"]
    files: list[str]  # the checkpoint; or unet, text_encoder[, text_encoder_2][, vae]
    size_bytes: int  # what downloading `files` costs (the high-noise half included)
    file_bytes: dict[str, int] = {}  # per file, so a dropped component can be taken off
    # Wan 2.2 A14B as two safetensors files: the low-noise one is `files`, this its first
    # stage (`--high-noise-diffusion-model`), as `GgufVariant.high_noise` for GGUF pairs.
    high_noise: str | None = None

    def as_variant(self, quant: str) -> GgufVariant:
        """These weights as the planner places them, at `quant` (how they load), the
        high-noise half travelling with the low-noise one."""
        return GgufVariant(
            quant=quant, files=self.files, size_bytes=self.size_bytes, high_noise=self.high_noise
        )

    def without(self, dropped: list[str]) -> ImageWeights:
        files = [f for f in self.files if f not in dropped]
        return self.model_copy(
            update={"files": files, "size_bytes": sum(self.file_bytes.get(f, 0) for f in files)}
        )

    @property
    def version(self) -> tuple[int, ...]:
        """The last `v<number>` in the label: `waiIllustriousSDXL_v170` -> (170,),
        `ponyDiffusionV6XL` -> (6,), `…v2.1` -> (2, 1); () when unnumbered."""
        found = _VERSION_RE.findall(self.label)
        return tuple(int(part) for part in found[-1].split(".")) if found else ()


def image_weights(files: list[TreeEntry], pasted: str | None = None) -> list[ImageWeights]:
    """Checkpoint files (each a version), else diffusers folders; oldest version first.
    `pasted` is a file the reference named outright: it is a version wherever it sits
    (LingBot-Video's diffusion model lives in its pipeline's `transformer/` folder)."""
    roots = _pipeline_roots(files)
    checkpoints = [
        ImageWeights(
            label=_basename(e.path)[: -len(".safetensors")],
            layout="file",
            files=[e.path],
            size_bytes=e.size,
            file_bytes={e.path: e.size},
        )
        for e in files
        if e.path.lower().endswith(".safetensors")
        and e.size >= CHECKPOINT_MIN_BYTES
        and (
            e.path == pasted
            or (
                not _COMPONENT_DIRS & set(e.path.split("/")[:-1])
                and not _in_component(e.path, roots)
            )
        )
        and not _UNREADABLE_FORMAT_RE.search(_basename(e.path))
    ]
    found = _paired_noise_halves(checkpoints) or _diffusers_folders(files)
    return sorted(found, key=lambda w: (w.version, -w.size_bytes, w.label))


def _paired_noise_halves(weights: list[ImageWeights]) -> list[ImageWeights]:
    """ComfyUI's `wan2.2_t2v_high_noise_14B_…` beside `…_low_noise_…`: one version, the
    low-noise file carrying its high-noise half (`video_files.pair_by_noise`, as GGUF pairs)."""

    def join(low: ImageWeights, high: ImageWeights) -> ImageWeights:
        return low.model_copy(
            update={
                "high_noise": high.files[0],
                "size_bytes": low.size_bytes + high.size_bytes,
                "file_bytes": {**low.file_bytes, **high.file_bytes},
            }
        )

    return pair_by_noise(weights, join)


def pick_label(labels: list[str], wanted: str) -> str:
    """`--version`: the label that is `wanted`, else the one label containing it
    (`turbo-v1.1` picks `anima-turbo-v1.1`); ValueError names the choices otherwise."""
    needle = wanted.strip().lower()
    exact = [label for label in labels if label.lower() == needle]
    found = exact or [label for label in labels if needle and needle in label.lower()]
    if len(found) == 1:
        return found[0]
    if not found:
        raise ValueError(f"no version '{wanted}' here (versions: {', '.join(labels) or 'none'})")
    raise ValueError(
        f"'{wanted}' matches {len(found)} versions ({', '.join(found)}); use the whole name"
    )


def _pipeline_roots(files: list[TreeEntry]) -> set[str]:
    """Folders holding a pipeline index (`""` for the repo's top)."""
    return {_dirname(e.path) for e in files if _basename(e.path) in PIPELINE_INDEX_FILES}


def _in_component(path: str, roots: set[str]) -> bool:
    """Below a pipeline index every folder is one of its components, whatever its name
    (ERNIE-Image keeps a 7 GB prompt-enhancer LLM in `pe/`): never a checkpoint version.
    A file beside the index (FLUX.2 klein's single file at the top) stays one."""
    for root in roots:
        if root and not path.startswith(f"{root}/"):
            continue
        if "/" in (path[len(root) + 1 :] if root else path):
            return True
    return False


def _diffusers_folders(files: list[TreeEntry]) -> list[ImageWeights]:
    by_path = {e.path: e for e in files}
    roots = sorted(
        {
            _dirname(_dirname(path))
            for path in by_path
            if _basename(_dirname(path)) == "unet"
            and _basename(path).startswith("diffusion_pytorch_model")
        }
    )
    folders = []
    for root in roots:
        chosen: list[TreeEntry] = []
        for component, stem, required in _DIFFUSERS_COMPONENTS:
            prefix = f"{root}/{component}/" if root else f"{component}/"
            entry = by_path.get(f"{prefix}{stem}.fp16.safetensors") or by_path.get(
                f"{prefix}{stem}.safetensors"
            )
            if entry is None and required:
                break
            if entry is not None:
                chosen.append(entry)
        else:
            folders.append(
                ImageWeights(
                    label=f"diffusers ({root})" if root else "diffusers",
                    layout="diffusers",
                    files=[e.path for e in chosen],
                    size_bytes=sum(e.size for e in chosen),
                    file_bytes={e.path: e.size for e in chosen},
                )
            )
    return folders


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _dirname(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""
