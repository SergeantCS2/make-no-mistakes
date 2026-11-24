"""Speech-to-text recipes: whisper.cpp's `whisper-server` from the official CUDA image.

Verified on a CPU build of 2026-10-10: with `--inference-path /v1/audio/transcriptions` the
server answers the OpenAI transcription contract (`multipart file=…` → `{"text": …}`,
`response_format=verbose_json` adds segments), `/health` is 200 once loaded, and `--convert`
(ffmpeg is in the image) accepts webm/mp3 uploads such as Odysseus's. The image's entrypoint
is `bash -c`, so the recipe overrides it with the binary, which is on the image's PATH.
"""

from __future__ import annotations

from pathlib import Path

from makenomistakes.fit.profiles import Profile
from makenomistakes.hub.facts import Engine, ModelFacts
from makenomistakes.recipes.cards import device_args
from makenomistakes.recipes.model import CONTAINER_MODELS_DIR, Recipe, container_name, publish

WHISPERCPP_IMAGE = "ghcr.io/ggml-org/whisper.cpp:main-cuda"
WHISPERCPP_SERVED_ID = "whisper-1"  # what OpenAI clients send; the server holds one model
TRANSCRIPTIONS_PATH = "/v1/audio/transcriptions"


def whispercpp_recipe(
    facts: ModelFacts, profile: Profile, name: str, port: int, image: str = WHISPERCPP_IMAGE
) -> Recipe:
    if profile.audio is None:
        raise ValueError(f"{facts.repo_id}: a speech-to-text recipe needs an audio profile")
    return Recipe(
        name=name,
        display_name=facts.display_name,
        repo_id=facts.repo_id,
        revision=facts.revision,
        modality=facts.modality,
        engine=Engine.WHISPERCPP,
        image=image,
        port=port,
        served_name=WHISPERCPP_SERVED_ID,
        files=list(profile.variant.files),
        args=profile.audio,
        profile=profile.name,
        quant=profile.quant,
        vram_estimate_bytes=profile.fit.gpu_bytes,
        fit_level=profile.fit.level,
    )


def whispercpp_run_argv(recipe: Recipe, models_dir: Path) -> list[str]:
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
        "--entrypoint",
        "whisper-server",  # the image's entrypoint is `bash -c`; the binary is on its PATH
        recipe.image,
        "-m",
        recipe.container_path(recipe.files[0]),
        "--host",
        "0.0.0.0",
        "--port",
        str(recipe.port),
        "--inference-path",
        TRANSCRIPTIONS_PATH,
        *recipe.whisper_args.to_cli(),
    ]
