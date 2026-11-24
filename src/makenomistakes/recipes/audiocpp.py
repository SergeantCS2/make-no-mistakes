"""audio.cpp recipes: text-to-speech and the studio's music models
(a song, stems, a conversion, a score), from its official images.

The server reads a JSON config, not flags, so a start renders `<models>/configs/<name>.json`
from the typed recipe (`write_config`, the registry's prepare step) and mounts the models
folder read-only as every engine does; the config names the one model under `/models`.
Verified on a CPU build of 2026-10-15: `/health` answers `{"status":"ok","models":1}` only
once the model is loaded with `lazy_load` off, `/v1/audio/speech` returns `audio/wav`, and
the engine canonicalises every path it reads, so a GGUF reached through the cache's
`snapshots/<rev>/` link fails on the blob's missing `.gguf` extension (`unsupported tensor
source format`): voice packages download as plain files under `voices/` (`download.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

from makenomistakes.engine_args_music import (
    STYLE_ADAPTER,
    STYLES_FOLDER,
    AceStepOptions,
    ConvertOptions,
    SeparatorOptions,
)
from makenomistakes.fit.cpu_rates import CpuSide
from makenomistakes.fit.profiles import Profile
from makenomistakes.hub.converted import converted_by_file, folder_of, kit_of
from makenomistakes.hub.facts import Engine, ModelFacts
from makenomistakes.hub.voices import VoiceFamily, kit_for
from makenomistakes.recipes.cards import device_args
from makenomistakes.recipes.model import (
    CONTAINER_MODELS_DIR,
    Recipe,
    container_name,
    publish,
)

AUDIOCPP_IMAGE = "ghcr.io/0xshug0/audio.cpp:full-cuda12"
CONFIGS_SUBDIR = "configs"
LORA_OPTION = "ace_step.lora"  # The session option naming a style's PEFT folder


def audiocpp_recipe(
    facts: ModelFacts,
    profile: Profile,
    name: str,
    port: int,
    image: str = AUDIOCPP_IMAGE,
    cpu: CpuSide | None = None,
) -> Recipe:
    """`cpu`: a CPU plan (no card): the physical cores as the config's `threads` (left
    unset, audio.cpp ran a separator on one core and outlasted the Song's wait)."""
    if profile.voice is None or (facts.voice_family is None and facts.music_family is None):
        raise ValueError(f"{facts.repo_id}: an audio.cpp recipe needs a voice or music profile")
    args = profile.voice
    if cpu is not None and cpu.threads and args.threads is None:
        args = args.model_copy(update={"threads": cpu.threads})
    if facts.music_family is not None:  # The config's id, never a chat client's model
        served = kit_of(facts.music_family, profile.variant.files[0]).served_id
    else:
        assert facts.voice_family is not None
        served = kit_for(facts.voice_family).served_id  # carries `tts` for Odysseus
    return Recipe(
        name=name,
        display_name=facts.display_name,
        repo_id=facts.repo_id,
        revision=facts.revision,
        modality=facts.modality,
        engine=Engine.AUDIOCPP,
        image=image,
        port=port,
        served_name=served,
        files=[*profile.variant.files, *profile.variant.voice_files],
        args=args,
        profile=profile.name,
        quant=profile.quant,
        vram_estimate_bytes=profile.fit.gpu_bytes,
        fit_level=profile.fit.level,
    )


def config_path(recipe: Recipe) -> str:
    """Where the rendered config lives, relative to the models directory."""
    return f"{CONFIGS_SUBDIR}/{recipe.name}.json"


# The config's `backend` per image (audio.cpp names it; the CPU image refuses `cuda`). A CUDA
# tag keeps the `cuda` every earlier config named.
_CONFIG_BACKENDS = (("full-cpu", "cpu"), ("full-vulkan", "vulkan"))


def config_backend(recipe: Recipe) -> str:
    tag = recipe.image.rsplit(":", 1)[-1]
    return next((name for prefix, name in _CONFIG_BACKENDS if tag.startswith(prefix)), "cuda")


def music_model(recipe: Recipe) -> dict[str, object]:
    """A studio instance's model entry: its task and the options audio.cpp reads at
    load (`session_options`) and per request (`default_request_options`; a request's own
    `options` override them)."""
    args = recipe.voice_args
    family, options = args.family, args.options
    session: dict[str, object] = {}
    defaults: dict[str, object] = {}
    if args.seed is not None:
        defaults["seed"] = args.seed
    if isinstance(options, AceStepOptions):
        defaults.update(route=options.route, duration_sec=options.duration)
        if options.steps is not None:
            defaults["num_inference_steps"] = options.steps
        if options.guidance is not None:
            defaults["guidance_scale"] = options.guidance
        if options.mem_saver:
            session["ace_step.mem_saver"] = True
        if options.style:  # The style's PEFT folder, merged into the DiT at load
            session[LORA_OPTION] = f"{CONTAINER_MODELS_DIR}/{STYLES_FOLDER}/{options.style}"
            session["ace_step.lora_scale"] = options.style_scale
    elif isinstance(options, SeparatorOptions) and options.num_overlap is not None:
        session[f"{family}.num_overlap"] = options.num_overlap
    elif isinstance(options, ConvertOptions):
        defaults.update(
            semitone_shift=options.semitone_shift, retrieval_blend=options.retrieval_blend
        )
    model: dict[str, object] = {
        "id": recipe.served_name,
        "family": family,
        "path": _model_path(recipe),
        "task": args.task,
        "mode": "offline",
        "default_request_options": defaults,
    }
    if session:
        model["session_options"] = session
    return model


def _model_path(recipe: Recipe) -> str:
    """The model file, or for a checkpoint mnm converts the folder made beside it."""
    if converted_by_file(recipe.files[0]) is not None:
        return f"{CONTAINER_MODELS_DIR}/{folder_of(recipe.local_paths[recipe.files[0]])}"
    return recipe.container_path(recipe.files[0])


def render_config(recipe: Recipe) -> dict[str, object]:
    """audio.cpp's `server.json` for this instance, from the typed recipe alone."""
    args = recipe.voice_args
    if args.music:
        return _server(recipe, music_model(recipe))
    model: dict[str, object] = {
        "id": recipe.served_name,
        "family": args.family,
        "path": recipe.container_path(recipe.files[0]),
        "task": args.task,
        "mode": "offline",
        "default_voice_preset": {"voice_id": args.voice},
        # every built-in voice as a preset: a request's `voice` names one of these (the
        # package's own voice files when it ships them, else the kit's list)
        "voice_presets": {v: {"voice_id": v} for v in voice_names(recipe)},
        "default_request_options": {
            **({"speed": args.speed} if args.paced else {}),
            **({"seed": args.seed} if args.seed is not None else {}),
        },
    }
    if args.language is not None:
        model["load_options"] = {"language": args.language}
    return _server(recipe, model)


def _server(recipe: Recipe, model: dict[str, object]) -> dict[str, object]:
    args = recipe.voice_args
    config: dict[str, object] = {
        "host": "0.0.0.0",
        "port": recipe.port,
        "backend": config_backend(recipe),
        "device": 0,
        "lazy_load": False,  # loaded before the port opens: /health 200 means ready
        "max_loaded_models": 1,
        "ui": True,
        "ui_management": False,
        "models": [model],
    }
    if args.threads is not None:
        config["threads"] = args.threads
    return config


def shipped_voices(recipe: Recipe) -> list[str]:
    """The voices the package ships as files, one embeddings file each after the model
    (PocketTTS); empty when the family's voices are built in (Kokoro, Qwen3)."""
    return [Path(f).stem for f in recipe.files[1:]]


def known_voices(recipe: Recipe) -> list[str]:
    """The voices this package has: its own embeddings when it ships them, else the kit's
    list; none for a studio instance, which reads no voice."""
    if recipe.voice_args.music:
        return []
    return shipped_voices(recipe) or list(kit_for(VoiceFamily(recipe.voice_args.family)).voices)


def voices_are_files(recipe: Recipe) -> bool:
    """A request naming a voice the package has no file for is a 500 naming the missing file.
    A kit's list is not the whole story: clones from the engine page are voices too."""
    return bool(shipped_voices(recipe))


def voice_names(recipe: Recipe) -> list[str]:
    """The voices a request may name, the instance's default first (when it is one of them)."""
    known = known_voices(recipe)
    voice = recipe.voice_args.voice
    return [voice, *(v for v in known if v != voice)] if voice in known else known


def write_config(recipe: Recipe, models_dir: Path) -> None:
    """Render the config under the models folder before a start (the registry's prepare step);
    the engine reads it through the read-only models mount. A style whose adapter is gone (a
    restore brings the library's rows, not its files) is refused here, in words, rather than
    as the engine's load failure."""
    if not recipe.downloaded:
        raise ValueError(f"{recipe.name} is not fully downloaded")
    options = recipe.voice_args.options
    if isinstance(options, AceStepOptions) and options.style:
        folder = models_dir / STYLES_FOLDER / options.style
        if not (folder / STYLE_ADAPTER).is_file():
            raise ValueError(
                f"its style '{options.style}' has no {STYLE_ADAPTER} in {folder} (a restore "
                f"brings the library's rows, not its files): `mnm set {recipe.name} style=` "
                "plays without it"
            )
    path = models_dir / config_path(recipe)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(render_config(recipe), indent=2) + "\n", encoding="utf-8")


def remove_config(recipe: Recipe, models_dir: Path) -> None:
    (models_dir / config_path(recipe)).unlink(missing_ok=True)


def audiocpp_run_argv(recipe: Recipe, models_dir: Path) -> list[str]:
    """The engine publishes the instance's port on 127.0.0.1, where the router reaches it; the
    config's `port` is the same one, inside the container."""
    if not recipe.downloaded:
        raise ValueError(f"{recipe.name} is not fully downloaded")
    return [
        "docker",
        "run",
        "--detach",
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
        recipe.image,
        "server",  # the image's entrypoint takes the tool first (`cli` or `server`)
        "--config",
        f"{CONTAINER_MODELS_DIR}/{config_path(recipe)}",
    ]
