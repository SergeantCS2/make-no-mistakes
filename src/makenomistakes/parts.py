"""What an instance is made of: each file, the part it plays, its size, and where the engine
keeps it (GPU or RAM), as `mnm show` and the instance page list it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from makenomistakes.engine_args import AudioCppArgs, LlamaCppArgs, SdCppArgs
from makenomistakes.hub.kits import Role
from makenomistakes.model_files import read_model_files
from makenomistakes.recipes import Recipe

_ROLES = {
    Role.VAE: "VAE",
    Role.TEXT_ENCODER: "text encoder",
    Role.CLIP_L: "text encoder (CLIP-L)",
    Role.CLIP_G: "text encoder (CLIP-G)",
    Role.T5XXL: "text encoder (T5-XXL)",
    Role.CLIP_VISION: "image encoder (CLIP-vision)",
    Role.HIGH_NOISE: "high-noise diffusion model",
    Role.AUDIO_VAE: "audio VAE",
    Role.EMBEDDINGS_CONNECTORS: "text embeddings connectors",
}
ENGRAM_PLACE = "disk, read as needed"  # An Engram table (`--lazy-mode on`)
_STREAMED = "RAM, copied to the GPU while in use"  # sd.cpp --offload-to-cpu


@dataclass(frozen=True)
class Part:
    role: str
    file: str  # the file's name; a diffusers folder is named by its components
    repo_id: str
    size_bytes: int | None  # None: not downloaded and not recorded
    place: str


def instance_parts(recipe: Recipe, models_dir: Path) -> list[Part]:
    """Every file the next start loads, in the order the engine takes them."""
    if isinstance(recipe.args, SdCppArgs):
        return _image_parts(recipe, recipe.args, models_dir)
    if isinstance(recipe.args, AudioCppArgs):
        return _voice_parts(recipe, models_dir)
    place = "GPU"
    if isinstance(recipe.args, LlamaCppArgs):
        place = _llama_place(recipe, recipe.args, models_dir)
    parts = [
        Part(
            "model",
            _name(f),
            recipe.repo_id,
            _size(recipe, f, models_dir),
            ENGRAM_PLACE if f == recipe.engram else place,
        )
        for f in recipe.files
    ]
    if recipe.projector:
        size = _size(recipe, recipe.projector, models_dir)
        parts.append(Part("vision projector", _name(recipe.projector), recipe.repo_id, size, "GPU"))
    if recipe.mtp and not recipe.drafts_with_mtp:  # Downloaded, loaded only as a draft
        size = _size(recipe, recipe.mtp, models_dir)
        role = f"MTP head (unused: `mnm set {recipe.name} draft=mtp` drafts with it)"
        parts.append(Part(role, _name(recipe.mtp), recipe.repo_id, size, "disk only"))
    if recipe.draft is not None:  # speculative decoding: every layer of it on the GPU
        draft = recipe.draft
        on_disk = (models_dir / draft.path).is_file()  # gone: the next start says so
        role = f"draft model ({draft.source}, up to {recipe.llama_args.draft_max} a step)"
        size = draft.size_bytes if on_disk else None
        parts.append(Part(role, _name(draft.file), draft.repo_id, size, "GPU (every layer)"))
    for head in recipe.draft_heads:  # Listed, never downloaded or run
        role = f"{head.kind} draft head (unused, not downloaded)"
        parts.append(Part(role, _name(head.file), recipe.repo_id, head.size_bytes, "on the Hub"))
    return parts


def _voice_parts(recipe: Recipe, models_dir: Path) -> list[Part]:
    """The voice model, then its built-in voices (PocketTTS keeps one small file a voice)."""
    model, *voices = recipe.files
    music = isinstance(recipe.args, AudioCppArgs) and recipe.args.music
    role = f"{recipe.modality.label} model" if music else "voice model"
    parts = [Part(role, _name(model), recipe.repo_id, _size(recipe, model, models_dir), "GPU")]
    if voices:
        size = sum(_size(recipe, f, models_dir) or 0 for f in voices) or None
        parts.append(
            Part(f"voices ({len(voices)} embeddings)", "embeddings/", recipe.repo_id, size, "GPU")
        )
    return parts


def _llama_place(recipe: Recipe, args: LlamaCppArgs, models_dir: Path) -> str:
    if args.gpu_layers == 0:  # The CPU backend's only plan
        return "RAM: every layer (the CPU backend)"
    files = read_model_files(recipe, models_dir)
    blocks = files.header.block_count if files and files.header else None
    if blocks is not None and args.gpu_layers > blocks:
        return "GPU (every layer" + _experts_in_ram(args, blocks) + ")"
    if blocks is not None:
        return f"GPU: {args.gpu_layers} of {blocks + 1} layers, the rest in RAM"
    return f"GPU: {args.gpu_layers} layers (not downloaded: the model's count is unknown)"


def _experts_in_ram(args: LlamaCppArgs, blocks: int) -> str:
    """`; the experts of 30 of 40 layers in RAM` for a MoE kept partly in RAM (`--n-cpu-moe`),
    else nothing."""
    held = args.cpu_moe_layers or 0
    if held <= 0:
        return ""
    return f"; the experts of {min(held, blocks)} of {blocks} layers in RAM"


def _image_parts(recipe: Recipe, args: SdCppArgs, models_dir: Path) -> list[Part]:
    def where(on_cpu: bool) -> str:
        if on_cpu:
            return "RAM"
        return _STREAMED if args.offload_to_cpu else "GPU"

    own = _STREAMED if args.offload_to_cpu else "GPU"
    if args.main_flag == "--model":  # a checkpoint or diffusers folder carries its own parts
        inside = "UNet, text encoders and VAE" if recipe.layout == "file" else "diffusers folder"
        if args.text_encoder_on_cpu:
            own += "; its text encoders in RAM"
        role = f"checkpoint ({inside})"
    else:
        role = "diffusion model"
        if args.diffusion_type is not None:
            role += f" (converted to {args.diffusion_type.upper()} while loading)"
    if recipe.layout == "diffusers":
        size = sum(_size(recipe, f, models_dir) or 0 for f in recipe.files) or None
        parts = [Part(role, f"{len(recipe.files)} files", recipe.repo_id, size, own)]
    else:
        parts = [
            Part(role, _name(f), recipe.repo_id, _size(recipe, f, models_dir), own)
            for f in recipe.files
        ]
    if recipe.high_noise is not None:  # Wan 2.2 A14B: the first stage, loaded beside the model
        size = _size(recipe, recipe.high_noise, models_dir)
        parts.append(
            Part(_ROLES[Role.HIGH_NOISE], _name(recipe.high_noise), recipe.repo_id, size, own)
        )
    for companion in recipe.companions:
        on_cpu = args.vae_on_cpu if companion.role is Role.VAE else args.text_encoder_on_cpu
        size = _size(recipe, companion.key, models_dir) or companion.size_bytes or None
        label = _ROLES[companion.role]
        if companion.role is Role.VAE and args.main_flag == "--model":
            label += " (replaces the checkpoint's own)"
        parts.append(Part(label, _name(companion.path), companion.repo_id, size, where(on_cpu)))
    if recipe.upscaler is not None:
        up = recipe.upscaler
        size = _size(recipe, up.key, models_dir) or up.size_bytes or None
        parts.append(Part("upscaler (ESRGAN)", up.name, up.repo_id, size, "loaded per request"))
    for lora in recipe.loras:
        how = f"weight {lora.weight:g}, " + ("always on" if lora.always_on else "picked per image")
        size = _size(recipe, lora.key, models_dir) or lora.size_bytes or None
        parts.append(
            Part(
                f"LoRA ({how})",
                _name(lora.path),
                lora.repo_id,
                size,
                "applied to the diffusion model",
            )
        )
    return parts


def _size(recipe: Recipe, key: str, models_dir: Path) -> int | None:
    rel = recipe.local_paths.get(key)
    if rel is None:
        return None
    try:
        return (models_dir / rel).stat().st_size
    except OSError:
        return None


def _name(path: str) -> str:
    return path.rsplit("/", 1)[-1]
