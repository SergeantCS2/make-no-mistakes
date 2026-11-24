"""One row per engine: what every module that used to test `recipe.engine is Engine.X` or
`isinstance(recipe.args, …)` needs to know.

A new engine (audio.cpp's text-to-speech) or a new mode of one (sd.cpp's `vid_gen`) is a row
here, not a branch in `recipes/__init__`, `engines`, `plan_types`, `download` and `autofix`.
The planner, the recipe builders and the per-engine fit models still dispatch where they
live: they take different inputs per engine (a GGUF header, an image kit, a whisper size)
and move into the registry together with the per-engine estimate type.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from makenomistakes.backend import Backend
from makenomistakes.engine_images import IMAGES, EngineImages, NoImage
from makenomistakes.hub.facts import Engine
from makenomistakes.recipes.audiocpp import (
    AUDIOCPP_IMAGE,
    audiocpp_run_argv,
    remove_config,
    write_config,
)
from makenomistakes.recipes.llamacpp import LLAMACPP_IMAGE, llamacpp_run_argv
from makenomistakes.recipes.llamacpp_cache import make_cache_folder, remove_cache_folder
from makenomistakes.recipes.model import Recipe
from makenomistakes.recipes.sdcpp import (
    SDCPP_IMAGE,
    link_image_files,
    remove_image_links,
    sdcpp_run_argv,
)
from makenomistakes.recipes.vllm import VLLM_IMAGE, vllm_run_argv
from makenomistakes.recipes.whispercpp import WHISPERCPP_IMAGE, whispercpp_run_argv

RunArgv = Callable[[Recipe, Path], list[str]]
Prepare = Callable[[Recipe, Path], None]


@dataclass(frozen=True)
class EngineSpec:
    engine: Engine
    label: str  # what the user reads: "chat", "image", "speech-to-text"
    image: str  # the default tag (CUDA 12 on amd64): `images.default`
    health_path: str  # GET here answers 200 once the engine is ready
    run_argv: RunArgv  # the exact `docker run`, from the typed recipe
    # Engines whose settings are not context and KV: where their own knobs live instead.
    knobs: str | None
    autofix_ladder: bool  # an out-of-memory start has settings to step down
    # What a start lays out under the models folder before `docker run`: sd.cpp's link
    # folders, audio.cpp's rendered config; None for an engine that reads the files as they are.
    prepare: Prepare | None = None
    cleanup: Prepare | None = None  # `mnm rm`: what prepare laid out goes (the files stay)
    # Where the files live: huggingface_hub's cache (`hf/`, blob links by hash) or plain
    # files under `voices/` for an engine that reads a file's format from its name.
    plain_files: bool = False
    # When what it holds on the GPU can be measured: at ready for an engine that
    # allocates while loading; at the first request for one that loads its weights then
    # (sd-server holds 40-200 MiB when ready, measured on an RTX 4070).
    resident_at: Literal["load", "request"] = "load"
    # Its server answers `GET /v1/models`; whisper-server's is a 404, so the router answers
    # `/upstream/<name>/v1/models` for it, loaded or parked (one model, the instance).
    lists_models: bool = True

    @property
    def images(self) -> EngineImages:
        """Its image per backend and architecture (`engine_images`)."""
        return IMAGES[self.engine]

    def image_for(self, backend: Backend, arch: str) -> str | NoImage:
        """The tag this box runs the engine from, or why it has none."""
        return self.images.image_for(backend, arch)


SPECS: dict[Engine, EngineSpec] = {
    Engine.LLAMACPP: EngineSpec(
        engine=Engine.LLAMACPP,
        label="chat",
        image=LLAMACPP_IMAGE,
        health_path="/health",  # 503 while loading, 200 when ready
        run_argv=llamacpp_run_argv,
        knobs=None,
        autofix_ladder=True,
        prepare=make_cache_folder,  # The folder its saved slots go to, `cache=on`
        cleanup=remove_cache_folder,
    ),
    Engine.SDCPP: EngineSpec(
        engine=Engine.SDCPP,
        label="image",
        image=SDCPP_IMAGE,
        health_path="/v1/models",  # no /health; the port opens once its files are planned
        run_argv=sdcpp_run_argv,
        knobs="size, steps and guidance are settings on the instance (`mnm set <name> width=…`)",
        autofix_ladder=True,
        prepare=link_image_files,
        cleanup=remove_image_links,
        resident_at="request",
    ),
    Engine.WHISPERCPP: EngineSpec(
        engine=Engine.WHISPERCPP,
        label="speech-to-text",
        image=WHISPERCPP_IMAGE,
        health_path="/health",  # 200 once the model is loaded (verified on the CPU build)
        run_argv=whispercpp_run_argv,
        knobs="language and threads are settings on the instance (`mnm set <name> language=en`)",
        autofix_ladder=False,  # one small model: its diagnosis says what to stop instead
        lists_models=False,
    ),
    Engine.AUDIOCPP: EngineSpec(
        engine=Engine.AUDIOCPP,
        label="text-to-speech",
        image=AUDIOCPP_IMAGE,
        health_path="/health",  # answers only once the model is loaded (lazy_load off)
        run_argv=audiocpp_run_argv,
        knobs="voice and seed (and speed, on Kokoro) are settings on the instance "
        "(`mnm set <name> voice=…`)",
        autofix_ladder=False,  # one small model
        plain_files=True,
        prepare=write_config,
        cleanup=remove_config,
    ),
    Engine.VLLM: EngineSpec(
        engine=Engine.VLLM,
        label="chat",
        image=VLLM_IMAGE,
        health_path="/health",  # 200 once the model is loaded and the KV cache allocated
        run_argv=vllm_run_argv,
        knobs=None,  # context and KV precision plan like llama.cpp's (`Plan._choose_vllm`)
        autofix_ladder=False,  # its own diagnosis names the share and the context to lower
    ),
}


def spec(engine: Engine) -> EngineSpec:
    return SPECS[engine]


def spec_for(recipe: Recipe) -> EngineSpec:
    return SPECS[recipe.engine]
