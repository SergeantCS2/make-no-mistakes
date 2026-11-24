"""ComfyUI sees mnm's image models: one copy on disk, ComfyUI in control.

ComfyUI reads extra model folders from `extra_model_paths.yaml`, one folder per kind
(checkpoints, diffusion_models, vae, text_encoders, loras, upscale_models, clip_vision). mnm's
files live in the Hugging Face cache under hashed names, so `mnm comfyui` lays out
`<models>/comfyui/<kind>/<instance>-<file>` links, relative like every link mnm makes, and
prints the block that names that folder. Run again after `mnm add` or `mnm rm`: links of
removed instances go, a file of anyone else's in those folders is never touched.

ComfyUI must see the models folder at the same path (ComfyUI installed in WSL, or its
container mounting the folder at that path): the links are relative and stay inside it.
Diffusers folders are not linked (ComfyUI loads single files); `mnm show` names their parts.
GGUF files (a Z-Image or FLUX kit's DiT and text encoder) need the ComfyUI-GGUF custom node,
whose loaders read the same diffusion_models and text_encoders folders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from makenomistakes.hub.facts import Engine
from makenomistakes.hub.kit_types import ImageFamily, Role
from makenomistakes.recipes.model import Recipe
from makenomistakes.recipes.sdcpp import sync_links

SUBDIR = "comfyui"
BLOCK_NAME = "makenomistakes"
BEGIN = "# >>> makenomistakes (written by `mnm comfyui --write`; edits between the marks go)"
END = "# <<< makenomistakes"
KINDS = (
    "checkpoints",
    "diffusion_models",
    "vae",
    "text_encoders",
    "clip_vision",
    "loras",
    "upscale_models",
)
# Families whose single file is a whole checkpoint (UNet, text encoders and VAE together).
CHECKPOINT_FAMILIES = frozenset(
    {ImageFamily.SD1, ImageFamily.SD2, ImageFamily.SDXL, ImageFamily.SDXL_REFINER}
)
_ROLE_KIND = {
    Role.VAE: "vae",
    Role.TEXT_ENCODER: "text_encoders",
    Role.CLIP_L: "text_encoders",
    Role.CLIP_G: "text_encoders",
    Role.T5XXL: "text_encoders",
    Role.CLIP_VISION: "clip_vision",
}


@dataclass(frozen=True)
class Layout:
    links: dict[str, dict[str, str]]  # kind -> link name -> target under the models folder
    skipped: list[str]  # instances with nothing ComfyUI loads as a file, and why

    @property
    def count(self) -> int:
        return sum(len(names) for names in self.links.values())


def layout(recipes: list[Recipe]) -> Layout:
    """Every downloaded image, edit and video instance's files, by the kind ComfyUI reads."""
    links: dict[str, dict[str, str]] = {kind: {} for kind in KINDS}
    skipped: list[str] = []
    for recipe in recipes:
        if recipe.engine is not Engine.SDCPP:
            continue
        if not recipe.downloaded:
            skipped.append(f"{recipe.name}: not downloaded yet")
            continue
        if recipe.layout == "diffusers":
            skipped.append(f"{recipe.name}: a diffusers folder (ComfyUI loads single files)")
        else:
            main = recipe.files[0]
            kind = (
                "checkpoints" if recipe.image_family in CHECKPOINT_FAMILIES else "diffusion_models"
            )
            _add(links[kind], recipe, main, recipe.local_paths[main])
            if recipe.high_noise and recipe.high_noise in recipe.local_paths:  # Wan 2.2 A14B
                _add(
                    links["diffusion_models"],
                    recipe,
                    recipe.high_noise,
                    recipe.local_paths[recipe.high_noise],
                )
        for companion in recipe.companions:
            role_kind = _ROLE_KIND.get(companion.role)
            if role_kind is not None and companion.key in recipe.local_paths:
                _add(links[role_kind], recipe, companion.path, recipe.local_paths[companion.key])
        for lora in recipe.loras:  # named per instance: one folder holds every instance's
            if lora.key in recipe.local_paths:
                links["loras"][f"{recipe.name}-{lora.name}.safetensors"] = recipe.local_paths[
                    lora.key
                ]
        if recipe.upscaler is not None and recipe.upscaler.key in recipe.local_paths:
            target = recipe.local_paths[recipe.upscaler.key]
            links["upscale_models"][f"{recipe.name}-{recipe.upscaler.name}"] = target
    return Layout(links, skipped)


def _add(kind: dict[str, str], recipe: Recipe, path: str, target: str) -> None:
    kind[f"{recipe.name}-{PurePosixPath(path).name}"] = target


def write_links(models_dir: Path, plan: Layout) -> None:
    """Make `<models>/comfyui/<kind>/` match the plan (only mnm's own links change)."""
    for kind, wanted in plan.links.items():
        sync_links(models_dir, f"{SUBDIR}/{kind}", wanted)


def yaml_block(models_dir: Path, extra: tuple[str, ...] = ()) -> str:
    """The `extra_model_paths.yaml` entry for the links folder (`extra`: more folders, such
    as a node pack's own)."""
    lines = [f"{BLOCK_NAME}:", f"    base_path: {json.dumps(str(models_dir / SUBDIR))}"]
    lines += [f"    {kind}: {kind}" for kind in (*KINDS, *extra)]
    return "\n".join(lines) + "\n"


def merged(existing: str | None, block: str) -> str:
    """The file with mnm's block between its marks: replaced when there, else appended. The
    rest of the user's file is kept as it is."""
    marked = f"{BEGIN}\n{block}{END}\n"
    if not existing:
        return marked
    if BEGIN in existing and END in existing:
        head, rest = existing.split(BEGIN, 1)
        tail = rest.split(END, 1)[1].lstrip("\n")
        return head + marked + tail
    if f"\n{BLOCK_NAME}:" in f"\n{existing}":
        raise ValueError(
            f"the file already has a `{BLOCK_NAME}:` entry not written by mnm; remove it, "
            "or paste the block printed by `mnm comfyui` in its place"
        )
    return existing.rstrip("\n") + "\n\n" + marked
