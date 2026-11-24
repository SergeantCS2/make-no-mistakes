"""vLLM recipes: a safetensors model folder served by `vllm serve` from the official image.

The model folder is the Hugging Face cache snapshot the files landed in, mounted read-only
like every engine's. vLLM writes compile caches under `~/.cache`: a Docker volume holds them
(`mnm-vllm-cache`), so a second start is faster and nothing root-owned lands in the models
folder. Usage statistics are switched off (`VLLM_NO_USAGE_STATS`, `DO_NOT_TRACK`) and the Hub
is never asked at run time (`HF_HUB_OFFLINE`): the files are all on disk already.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from makenomistakes.backend import Backend
from makenomistakes.backend_now import recipe_backend
from makenomistakes.fit.profiles import Profile
from makenomistakes.fit.vllm import BATCHED_TOKENS
from makenomistakes.hub.facts import Engine, ModelFacts
from makenomistakes.hub.vllm_files import config_path
from makenomistakes.recipes.cards import device_args
from makenomistakes.recipes.llamacpp import served_name
from makenomistakes.recipes.model import CONTAINER_MODELS_DIR, Recipe, container_name, publish

VLLM_IMAGE = "vllm/vllm-openai:latest"
CACHE_VOLUME = "mnm-vllm-cache"
PRIVATE_ENV = ("HF_HUB_OFFLINE=1", "VLLM_NO_USAGE_STATS=1", "DO_NOT_TRACK=1")


def vllm_recipe(
    facts: ModelFacts, profile: Profile, name: str, port: int, image: str = VLLM_IMAGE
) -> Recipe:
    if profile.vllm is None:
        raise ValueError(f"{facts.repo_id}: a vLLM recipe needs a vLLM profile")
    return Recipe(
        name=name,
        display_name=facts.display_name,
        repo_id=facts.repo_id,
        revision=facts.revision,
        modality=facts.modality,
        engine=Engine.VLLM,
        image=image,
        port=port,
        served_name=vllm_served_name(facts, profile.quant),
        files=list(profile.variant.files),
        args=profile.vllm,
        profile=profile.name,
        quant=profile.quant,
        vram_estimate_bytes=profile.fit.gpu_bytes,
        fit_level=profile.fit.level,
    )


def vllm_served_name(facts: ModelFacts, quant: str) -> str:
    """llama.cpp's id (the family name, `-vision` where Odysseus needs it), with the quant
    only when the repo's name does not say it already (`Qwen3-8B-AWQ`, not `…-awq-awq-int4`)."""
    named = served_name(facts, quant)
    method = quant.split("-", maxsplit=1)[0].lower()
    base = named.removesuffix(f"-{quant.lower()}")
    return base if method in base.split("-") else named


def model_folder(recipe: Recipe) -> str:
    """The snapshot folder that holds `config.json`, as the container sees it."""
    return str(PurePosixPath(recipe.container_path(config_path(recipe.files))).parent)


ROCM_SHM = "32g"  # the R9700 guides' (Radiance, tcclaviger); planned


def shm_argv(recipe: Recipe) -> list[str]:
    """vLLM on ROCm (planned, unverified): `--shm-size 32g` beside `--ipc host`, as the
    R9700 guides run it; nothing on CUDA."""
    return ["--shm-size", ROCM_SHM] if recipe_backend(recipe) is Backend.ROCM else []


def vllm_run_argv(recipe: Recipe, models_dir: Path) -> list[str]:
    if not recipe.downloaded:
        raise ValueError(f"{recipe.name} is not fully downloaded")
    env = [part for value in PRIVATE_ENV for part in ("--env", value)]
    return [
        "docker",
        "run",
        "--detach",
        "--name",
        container_name(recipe.name),
        *device_args(recipe),
        "--ipc",
        "host",  # PyTorch's shared memory between the server's processes
        *shm_argv(recipe),
        "--publish",
        publish(recipe),
        "--volume",
        f"{models_dir}:{CONTAINER_MODELS_DIR}:ro",
        "--volume",
        f"{CACHE_VOLUME}:/root/.cache",
        *env,
        "--restart",
        "no",
        "--label",
        f"mnm.instance={recipe.name}",
        "--entrypoint",
        "vllm",
        recipe.image,
        "serve",
        model_folder(recipe),
        "--served-model-name",
        recipe.served_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(recipe.port),
        "--max-num-batched-tokens",
        str(BATCHED_TOKENS),
        *recipe.vllm_args.to_cli(),
    ]
