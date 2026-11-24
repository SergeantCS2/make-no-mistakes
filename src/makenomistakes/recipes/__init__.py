"""Typed descriptions of runnable instances and how to render them for an engine."""

from pathlib import Path

from makenomistakes.engine_args import AudioCppArgs, LlamaCppArgs, SdCppArgs
from makenomistakes.engine_args_whisper import WhisperArgs
from makenomistakes.hub.facts import Engine
from makenomistakes.recipes.audiocpp import AUDIOCPP_IMAGE, audiocpp_recipe, audiocpp_run_argv
from makenomistakes.recipes.llamacpp import LLAMACPP_IMAGE, llamacpp_recipe, llamacpp_run_argv
from makenomistakes.recipes.model import (
    CONTAINER_MODELS_DIR,
    Recipe,
    RepoFile,
    UpscalerFile,
    companion_key,
    container_name,
    validate_instance_name,
)
from makenomistakes.recipes.registry import SPECS, EngineSpec, spec, spec_for
from makenomistakes.recipes.sdcpp import SDCPP_IMAGE, SDCPP_SERVED_ID, sdcpp_recipe, sdcpp_run_argv
from makenomistakes.recipes.whispercpp import (
    WHISPERCPP_IMAGE,
    WHISPERCPP_SERVED_ID,
    whispercpp_recipe,
    whispercpp_run_argv,
)


def docker_run_argv(recipe: Recipe, models_dir: Path) -> list[str]:
    """The exact command the supervisor runs, derived from the typed recipe."""
    return spec_for(recipe).run_argv(recipe, models_dir)


def health_path(recipe: Recipe) -> str:
    return spec_for(recipe).health_path


__all__ = [
    "AUDIOCPP_IMAGE",
    "CONTAINER_MODELS_DIR",
    "LLAMACPP_IMAGE",
    "SDCPP_IMAGE",
    "SDCPP_SERVED_ID",
    "SPECS",
    "WHISPERCPP_IMAGE",
    "WHISPERCPP_SERVED_ID",
    "AudioCppArgs",
    "Engine",
    "EngineSpec",
    "LlamaCppArgs",
    "Recipe",
    "RepoFile",
    "SdCppArgs",
    "UpscalerFile",
    "WhisperArgs",
    "audiocpp_recipe",
    "audiocpp_run_argv",
    "companion_key",
    "container_name",
    "docker_run_argv",
    "health_path",
    "llamacpp_recipe",
    "llamacpp_run_argv",
    "sdcpp_recipe",
    "sdcpp_run_argv",
    "spec",
    "spec_for",
    "validate_instance_name",
    "whispercpp_recipe",
    "whispercpp_run_argv",
]
