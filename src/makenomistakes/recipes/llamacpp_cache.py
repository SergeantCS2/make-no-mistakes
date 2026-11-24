"""The folder a llama.cpp instance's saved slots live in, `<models>/prompt-cache/<name>/`,
mounted read-write at `SLOT_CACHE_DIR` in its container while it keeps its slots.

The engine runs as root and writes its files there; the folder is the user's (made here, mode
0700, before `docker run`: Docker would make a missing one as root), so the user's processes
delete them (removing a file needs the folder's permission, not the file's) and nobody else
reads them: a slot holds a conversation's prompt."""

from __future__ import annotations

import shutil
from pathlib import Path

from makenomistakes.engine_args import SLOT_CACHE_DIR, LlamaCppArgs
from makenomistakes.hub.facts import Modality
from makenomistakes.recipes.model import Recipe

CACHE_FOLDER = "prompt-cache"  # under the models folder; `uninstall.sh` clears it


def cache_folder(models_dir: Path, name: str) -> Path:
    return models_dir / CACHE_FOLDER / name


def keeps_slots(recipe: Recipe) -> bool:
    """A llama.cpp instance whose slots are saved across unloads: `cache=on`, or `cache` left
    to auto on a coding instance (an agent's long prompt is what a reload reads again)."""
    if not isinstance(recipe.args, LlamaCppArgs):
        return False
    chosen = recipe.args.slot_cache
    return chosen if chosen is not None else recipe.modality is Modality.CODING


def cache_volume(recipe: Recipe, models_dir: Path) -> list[str]:
    """The folder's mount, beside the read-only models mount (empty when it keeps none)."""
    if not keeps_slots(recipe):
        return []
    return ["--volume", f"{cache_folder(models_dir, recipe.name)}:{SLOT_CACHE_DIR}"]


def slot_dir(recipe: Recipe) -> str | None:
    """The folder the engine's `--slot-save-path` names, with the mount or not at all."""
    return SLOT_CACHE_DIR if keeps_slots(recipe) else None


def make_cache_folder(recipe: Recipe, models_dir: Path) -> None:
    """Before every start (the registry's prepare step): the folder, the user's alone, while
    the instance keeps its slots; gone with what it held when it no longer does, however the
    setting changed (a restored backup, an edit by hand)."""
    folder = cache_folder(models_dir, recipe.name)
    if not keeps_slots(recipe):
        remove_cache_folder(recipe, models_dir)
        return
    folder.mkdir(parents=True, exist_ok=True)
    folder.chmod(0o700)


def remove_cache_folder(recipe: Recipe, models_dir: Path) -> None:
    """`mnm rm`, `cache=off`: the saved slots go (a cache, and a conversation's prompt). A
    delete that fails raises (`OSError`): the caller says so, a conversation is never left on
    disk without a word."""
    folder = cache_folder(models_dir, recipe.name)
    if folder.is_dir() and not folder.is_symlink():
        shutil.rmtree(folder)


def without_slots(recipe: Recipe) -> Recipe:
    """A copy's recipe (a tune's trial, an engine self-test): no saved slots, so it mounts no
    folder of the instance's and its stop saves nothing under the instance's name."""
    if not keeps_slots(recipe):
        return recipe
    args = recipe.args
    assert isinstance(args, LlamaCppArgs)  # `keeps_slots` said so
    return recipe.model_copy(update={"args": args.model_copy(update={"slot_cache": False})})
