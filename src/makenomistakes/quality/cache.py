"""Where a quality tune's quant files sit in the models folder's Hub cache: a recipe with
the files the cache holds filled in, read from disk, never the Hub. The plan finds a quant that
is here already (another instance's, a kept candidate's); the end finds what a download put
there, a download cut short included, before it deletes it."""

from __future__ import annotations

from pathlib import Path

from makenomistakes.download import hub_cache_lookup, wanted_files
from makenomistakes.recipes import Recipe


def found_here(models_dir: Path, recipe: Recipe) -> Recipe:
    """`recipe` with each file the cache holds recorded in its `local_paths`."""
    cache = models_dir / "hf"
    paths = dict(recipe.local_paths)
    for key, repo, path, revision in wanted_files(recipe):
        if key in paths:
            continue
        hit = hub_cache_lookup(repo, path, revision, cache)
        if hit is not None:
            paths[key] = str(hit.relative_to(models_dir))
    return recipe.model_copy(update={"local_paths": paths})
