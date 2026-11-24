"""A downloaded recipe's files as the fit models need them: sizes, the GGUF header, and a
checkpoint split the way the plan placed it (Fits now, the start-time upgrade, auto-fix)."""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

from pydantic import ValidationError

from makenomistakes.fit.image_memory import checkpoint_sizes, kit_sizes
from makenomistakes.gguf.reader import FileSource, GgufError, read_header
from makenomistakes.gguf.tensors import model_tensors, needs_tensors, read_tensors
from makenomistakes.hub.facts import Engine
from makenomistakes.hub.hf_config import HfConfig
from makenomistakes.hub.vllm_files import config_path
from makenomistakes.recipes import Recipe
from makenomistakes.safetensors.reader import SafetensorsError, read_safetensors_header
from makenomistakes.safetensors.sd import image_parts
from makenomistakes.settings_estimates import ModelFiles

# What was read, by the recipe and its files' (path, mtime, size): the web bar every 10 s,
# `mnm ls` and each router load's room check re-read every GGUF header (and a sharded MoE's
# tensor tables) otherwise. A file rewritten or a recipe changed reads anew.
_KEEP = 64
_read: OrderedDict[tuple[object, ...], ModelFiles | None] = OrderedDict()
_lock = threading.Lock()


def read_model_files(recipe: Recipe, models_dir: Path) -> ModelFiles | None:
    """Sizes of a downloaded recipe's files (and the GGUF header for llama.cpp), each file
    read once while it stays as it is; None when anything is missing or unreadable."""
    if not recipe.downloaded:
        return None
    try:
        key = _key(recipe, models_dir)
    except OSError:  # a file gone: read (and refused) as before, nothing kept
        return _read_files(recipe, models_dir)
    with _lock:
        if key in _read:
            _read.move_to_end(key)
            return _read[key]
    found = _read_files(recipe, models_dir)
    with _lock:
        _read[key] = found
        while len(_read) > _KEEP:
            _read.popitem(last=False)
    return found


def _key(recipe: Recipe, models_dir: Path) -> tuple[object, ...]:
    rels = [*recipe.local_paths.values(), *([recipe.draft.path] if recipe.draft else [])]
    stats = []
    for rel in rels:
        info = (models_dir / rel).stat()
        stats.append((rel, info.st_mtime_ns, info.st_size))
    return (str(models_dir), recipe.model_dump_json(), tuple(stats))


def _read_files(recipe: Recipe, models_dir: Path) -> ModelFiles | None:
    paths = {key: models_dir / rel for key, rel in recipe.local_paths.items()}
    try:
        halves = [*recipe.files, *([recipe.high_noise] if recipe.high_noise else [])]
        weights = sum(paths[f].stat().st_size for f in halves)
        if recipe.engram is not None:  # its Engram table stays on disk
            weights -= paths[recipe.engram].stat().st_size
        projector = paths[recipe.projector].stat().st_size if recipe.projector else 0
        sizes = {c.key: paths[c.key].stat().st_size for c in recipe.companions}
        header = (
            read_header(FileSource(paths[recipe.files[0]])).own_quant(recipe.quant)
            if recipe.engine is Engine.LLAMACPP
            else None
        )
        if header is not None and needs_tensors(header):  # each block's own experts
            found = [t for f in recipe.files for t in read_tensors(FileSource(paths[f])).tensors]
            header = header.with_tensors(model_tensors(found, header.block_count or 0))
        kit = kit_sizes(recipe.companions, sizes)
        if recipe.safetensors_weights:  # split the way the plan placed it
            parts = image_parts(
                recipe.layout,
                recipe.files,
                lambda f: read_safetensors_header(FileSource(paths[f])),
            )
            checkpoint = checkpoint_sizes(parts, kit, recipe.sd_args.diffusion_type)
            weights, kit = checkpoint.diffusion_bytes, checkpoint.sizes
        config = (
            HfConfig.model_validate_json(paths[config_path(recipe.files)].read_text())
            if recipe.engine is Engine.VLLM
            else None
        )
        draft = recipe.draft
        draft_header = (
            read_header(FileSource(models_dir / draft.path)).own_quant(draft.quant)
            if draft is not None and recipe.engine is Engine.LLAMACPP
            else None
        )
    except (GgufError, SafetensorsError, OSError, KeyError, ValidationError):
        return None
    draft_bytes = draft.size_bytes if draft is not None else 0
    return ModelFiles(weights, projector, kit, header, config, draft_header, draft_bytes)
