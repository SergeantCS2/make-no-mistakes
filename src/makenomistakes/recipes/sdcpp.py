"""stable-diffusion.cpp recipes: build one from an image profile, render it to `docker run`.

Besides the downloaded files the engine reads two folders under the models directory that
this module keeps as relative symlinks into the Hugging Face cache: `views/<instance>/`, a
diffusers folder under the names sd.cpp looks for (`unet/diffusion_pytorch_model.safetensors`
for an `….fp16.safetensors` file), and `loras/<instance>/`, the LoRAs the instance offers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from makenomistakes.engine_args import SdCppArgs
from makenomistakes.fit.profiles import Profile
from makenomistakes.hub.civitai import SUBDIR as CIVITAI_SUBDIR
from makenomistakes.hub.facts import Engine, ModelFacts
from makenomistakes.hub.kits import ImageFamily, ImageKit, Role, role_flag
from makenomistakes.recipes.cards import device_args
from makenomistakes.recipes.model import (
    CONTAINER_MODELS_DIR,
    HF_CACHE_SUBDIR,
    Recipe,
    RepoFile,
    UpscalerFile,
    container_name,
    publish,
)

SDCPP_IMAGE = "ghcr.io/leejet/stable-diffusion.cpp:master-cuda"
SDCPP_SERVED_ID = "sd-cpp-local"  # the server reports this fixed id on /v1/models
VIEWS_SUBDIR = "views"
LORAS_SUBDIR = "loras"
UPSCALERS_SUBDIR = "upscalers"  # sd-server scans a folder for upscalers (`--hires-upscalers-dir`)


def view_dir(recipe: Recipe) -> str:
    """Where a diffusers recipe's folder view lives, relative to the models directory."""
    return f"{VIEWS_SUBDIR}/{recipe.name}"


def lora_dir(recipe: Recipe) -> str:
    return f"{LORAS_SUBDIR}/{recipe.name}"


def upscaler_dir(recipe: Recipe) -> str:
    return f"{UPSCALERS_SUBDIR}/{recipe.name}"


@dataclass(frozen=True)
class CheckpointLoad:
    """How a checkpoint or diffusers plan's weights load (GGUF kits use the defaults)."""

    layout: Literal["file", "diffusers"] = "file"
    weight_type: Literal["f16"] | None = None  # F32/BF16 weights load as F16


GGUF_KIT_LOAD = CheckpointLoad()  # a GGUF diffusion file: one file, stored as it runs


def sdcpp_recipe(  # noqa: PLR0913, PLR0917 - the plan's parts, named at the one call site
    facts: ModelFacts,
    kit: ImageKit,
    sizes: dict[str, int],
    profile: Profile,
    name: str,
    port: int,
    load: CheckpointLoad = GGUF_KIT_LOAD,
    upscaler: UpscalerFile | None = None,
) -> Recipe:
    """`sizes` maps companion keys (repo:path) to byte sizes resolved from the Hub. The
    model card's example negative prompt, when there is one, is the instance default.
    `upscaler` is the folder-served ESRGAN model the user asked for (`--upscaler`)."""
    if profile.image is None:
        raise ValueError(f"{facts.repo_id}: an image recipe needs an image profile")
    args = profile.image.model_copy(
        update={"main_flag": kit.main_flag, "weight_type": load.weight_type}
    )
    if facts.negative_prompt and not args.negative_prompt:
        args = args.model_copy(update={"negative_prompt": facts.negative_prompt})
    companions = [
        RepoFile(
            role=c.role,
            repo_id=c.repo_id,
            path=c.path,
            revision=c.revision,
            size_bytes=sizes.get(c.key, 0),
        )
        for c in kit.companions
    ]
    return Recipe(
        name=name,
        display_name=facts.display_name,
        repo_id=facts.repo_id,
        revision=facts.revision,
        modality=facts.modality,
        engine=Engine.SDCPP,
        image=SDCPP_IMAGE,
        port=port,
        served_name=SDCPP_SERVED_ID,
        files=list(profile.variant.files),
        high_noise=profile.variant.high_noise,
        upscaler=upscaler,
        companions=companions,
        layout=load.layout,
        image_family=kit.family,
        args=SdCppArgs.model_validate(args.model_dump()),
        profile=profile.name,
        quant=profile.quant,
        vram_estimate_bytes=profile.fit.gpu_bytes,
        fit_level=profile.fit.level,
    )


# An SDXL folder's second text encoder (CLIP-G), passed again as `--clip_g`. sd.cpp loads
# the folder's copy as `te.1.…` and its own rename turns that into the first encoder's names
# (`cond_stage_model.transformer.1.…`), so the model it builds has one encoder: SD 1.x,
# hundreds of tensor errors. `--clip_g` loads it under
# `cond_stage_model.1.…`; the folder's copy must stay too, or the UNet is renamed by SD 1.x
# rules (6 tensors missing). Reproduced and verified on a CPU build of master 2988060.
SECOND_ENCODER = "text_encoder_2/model.safetensors"


def sdcpp_run_argv(recipe: Recipe, models_dir: Path) -> list[str]:
    if not recipe.downloaded:
        raise ValueError(f"{recipe.name} is not fully downloaded")
    argv = [
        "docker",
        "run",
        "--detach",
        "--init",
        "--name",
        container_name(recipe.name),
        *device_args(recipe),
        "--publish",
        publish(recipe),
        "--volume",
        f"{models_dir}:{CONTAINER_MODELS_DIR}:ro",
        "--restart",
        "no",
        "--label",
        f"mnm.instance={recipe.name}",
        "--entrypoint",
        "/sd-server",
        recipe.image,
        "--listen-ip",
        "0.0.0.0",
        "--listen-port",
        str(recipe.port),
        recipe.sd_args.main_flag,
        (
            f"{CONTAINER_MODELS_DIR}/{view_dir(recipe)}"
            if recipe.layout == "diffusers"
            else recipe.container_path(recipe.files[0])
        ),
    ]
    if recipe.layout == "diffusers" and recipe.image_family is ImageFamily.SDXL:
        argv += [
            role_flag(Role.CLIP_G),
            f"{CONTAINER_MODELS_DIR}/{view_dir(recipe)}/{SECOND_ENCODER}",
        ]
    if recipe.high_noise is not None:  # Wan 2.2 A14B's first stage
        argv += [role_flag(Role.HIGH_NOISE), recipe.container_path(recipe.high_noise)]
    for companion in recipe.companions:
        argv += [role_flag(companion.role), recipe.container_path(companion.key)]
    argv += ["--lora-model-dir", f"{CONTAINER_MODELS_DIR}/{lora_dir(recipe)}"]
    if recipe.upscaler is not None:
        # The server lists the folder's models for `/sdcpp/v1/upscale` and the hires step
        # (`--upscale-model` is sd-cli's, and would upscale every output).
        argv += ["--hires-upscalers-dir", f"{CONTAINER_MODELS_DIR}/{upscaler_dir(recipe)}"]
    always = [f"<lora:{lora.name}:{lora.weight:g}>" for lora in recipe.loras if lora.always_on]
    if always:  # the startup default prompt: every request inherits these (verified, f89d9b1)
        argv += ["-p", " ".join(always)]
    return argv + recipe.sd_args.to_cli()


def diffusers_name(repo_file: str) -> str:
    """The name sd.cpp loads a diffusers component under: `unet/…model.fp16.safetensors` ->
    `unet/…model.safetensors` (it only looks for the plain name)."""
    component, name = Path(repo_file).parent.name, Path(repo_file).name
    return f"{component}/{name.replace('.fp16.safetensors', '.safetensors')}"


def link_image_files(recipe: Recipe, models_dir: Path) -> None:
    """Make the folders the engine reads match the recipe: idempotent, run after a download
    and before every start. Only links into the download cache (the ones this tool makes)
    are ever replaced or removed there."""
    if recipe.engine is not Engine.SDCPP:
        return
    if recipe.layout == "diffusers":
        sync_links(
            models_dir,
            view_dir(recipe),
            {diffusers_name(f): recipe.local_paths[f] for f in recipe.files},
        )
    sync_links(
        models_dir,
        lora_dir(recipe),
        {f"{lora.name}.safetensors": recipe.local_paths[lora.key] for lora in recipe.loras},
    )
    if recipe.upscaler is not None:
        sync_links(
            models_dir,
            upscaler_dir(recipe),
            {recipe.upscaler.name: recipe.local_paths[recipe.upscaler.key]},
        )


def remove_image_links(recipe: Recipe, models_dir: Path) -> None:
    """`mnm rm`: take the instance's links away, and its folders when nothing else is in
    them (the downloaded files stay in the cache)."""
    for folder in (view_dir(recipe), lora_dir(recipe), upscaler_dir(recipe)):
        base = models_dir / folder
        if not base.is_dir():
            continue
        sync_links(models_dir, folder, {})
        for path in sorted((p for p in base.rglob("*") if p.is_dir()), reverse=True):
            if not any(path.iterdir()):
                path.rmdir()
        if not any(base.iterdir()):
            base.rmdir()


def _ours(link: Path, models_dir: Path) -> bool:
    """A link this tool made: it points into the download cache (or Civitai's files)."""
    target = (link.parent / os.readlink(link)).resolve()
    return any(
        target.is_relative_to((models_dir / sub).resolve())
        for sub in (HF_CACHE_SUBDIR, CIVITAI_SUBDIR)
    )


def sync_links(models_dir: Path, folder: str, wanted: dict[str, str]) -> None:
    """`wanted`: link name inside `folder` -> target relative to the models directory."""
    base = models_dir / folder
    base.mkdir(parents=True, exist_ok=True)
    for name, target in wanted.items():
        link = base / name
        link.parent.mkdir(parents=True, exist_ok=True)
        relative = os.path.relpath(models_dir / target, link.parent)
        if link.is_symlink() and os.readlink(link) == relative:
            continue
        if link.is_symlink() and _ours(link, models_dir):
            link.unlink()
        elif link.exists() or link.is_symlink():
            raise ValueError(f"{link} is not a link this tool made; move it away")
        link.symlink_to(relative)
    for path in sorted(base.rglob("*"), reverse=True):
        unwanted = path.relative_to(base).as_posix() not in wanted
        if path.is_symlink() and unwanted and _ours(path, models_dir):
            path.unlink()
