"""A converted checkpoint's folder, made after its download (the rows are
`hub/converted.py`): the checkpoint's tensors as `model.safetensors`
(`safetensors/torch_ckpt.py`) and its MSST YAML as audio.cpp's `config.json`, in
`converted/<checkpoint stem>/` beside the checkpoint, so `mnm rm --files` and the uninstall
find it with the download. A folder already made from the same checkpoint (its size and time
recorded beside it) is kept; anything else is made again.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from makenomistakes.hub.converted import ConvertedKit, converted_by_file, folder_of
from makenomistakes.hub.music import MusicFamily
from makenomistakes.recipes.model import Recipe
from makenomistakes.safetensors.torch_ckpt import CheckpointError, write_safetensors

WEIGHTS = "model.safetensors"
CONFIG = "config.json"
STAMP = ".mnm-converted.json"  # the checkpoint the folder was made from: its size and time
# Bumped with any change to what a converted folder holds (`torch_ckpt`, `split_qkv`,
# `audiocpp_config`): a folder another converter made is made again. 2: the stamp says it.
CONVERTER = 2


class ConversionError(ValueError):
    """A checkpoint mnm could not make into audio.cpp's folder, worded for the user."""


class _Loader(yaml.SafeLoader):
    """MSST's YAML writes some lists as `!!python/tuple`: read as lists, nothing constructed."""


def _as_list(loader: yaml.SafeLoader, node: yaml.Node) -> list[Any]:
    if not isinstance(node, yaml.SequenceNode):
        raise yaml.constructor.ConstructorError(
            None, None, "a tuple that is not a list", node.start_mark
        )
    return loader.construct_sequence(node)


_Loader.add_constructor("tag:yaml.org,2002:python/tuple", _as_list)


class _Loose(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _Audio(_Loose):
    sample_rate: int
    chunk_size: int


class _Model(_Loose):
    dim: int
    depth: int
    stereo: bool = True
    num_stems: int
    time_transformer_depth: int = 1
    freq_transformer_depth: int = 1
    linear_transformer_depth: int = 0
    freqs_per_bands: list[int] | None = None  # BS-RoFormer
    num_bands: int | None = None  # Mel-Band RoFormer
    dim_head: int
    heads: int
    stft_n_fft: int
    stft_hop_length: int
    stft_win_length: int
    stft_normalized: bool = False
    mask_estimator_depth: int
    mlp_expansion_factor: int = 4
    skip_connection: bool = False


class _Inference(_Loose):
    batch_size: int = 1
    num_overlap: int = 4


class MsstConfig(_Loose):
    audio: _Audio
    model: _Model
    inference: _Inference = _Inference()


def audiocpp_config(text: str, family: MusicFamily) -> dict[str, Any]:
    """audio.cpp's `config.json` from an MSST YAML (the fields its RoFormer loader reads)."""
    try:
        parsed = MsstConfig.model_validate(yaml.load(text, Loader=_Loader))
    except (yaml.YAMLError, ValidationError) as exc:
        raise ConversionError(f"its YAML is not an MSST RoFormer config: {exc}") from exc
    model = parsed.model
    if model.linear_transformer_depth or model.skip_connection:
        raise ConversionError(
            "its YAML asks for a linear transformer or skip connections, which audio.cpp's "
            "RoFormer runtime does not run"
        )
    bands: dict[str, Any]
    if family is MusicFamily.BS_ROFORMER:
        if not model.freqs_per_bands:
            raise ConversionError("its YAML names no `freqs_per_bands` for a BS-RoFormer")
        bands = {"freqs_per_bands": model.freqs_per_bands}
    else:
        if not model.num_bands:
            raise ConversionError("its YAML names no `num_bands` for a Mel-Band RoFormer")
        bands = {"num_bands": model.num_bands}
    return {
        "model_type": family.value,
        "sample_rate": parsed.audio.sample_rate,
        "stereo": model.stereo,
        "chunk_size": parsed.audio.chunk_size,
        "batch_size": parsed.inference.batch_size,
        "num_overlap": parsed.inference.num_overlap,
        "normalize": False,
        "dim": model.dim,
        "depth": model.depth,
        "num_stems": model.num_stems,
        "time_transformer_depth": model.time_transformer_depth,
        "freq_transformer_depth": model.freq_transformer_depth,
        "linear_transformer_depth": 0,
        **bands,
        "dim_head": model.dim_head,
        "heads": model.heads,
        "n_fft": model.stft_n_fft,
        "hop_length": model.stft_hop_length,
        "win_length": model.stft_win_length,
        "stft_normalized": model.stft_normalized,
        "mask_estimator_depth": model.mask_estimator_depth,
        "mlp_expansion_factor": model.mlp_expansion_factor,
        "skip_connection": False,
    }


def _stamp(*files: Path) -> dict[str, object]:
    """What the folder was made from: each file's size and time, and the converter that made
    it (a fix to the reader or the config makes every folder again: a folder kept by its files'
    times alone would outlive the fix)."""
    files_made = {f.name: [f.stat().st_size, f.stat().st_mtime_ns] for f in files}
    return {"converter": CONVERTER, "files": files_made}


def _made_from(stamp: Path) -> object:
    try:
        return json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None  # missing or damaged: made again


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked(row: ConvertedKit, checkpoint: Path, config: Path) -> None:
    """The files are the ones mnm checked against their author's output, or a refusal."""
    for path, wanted in ((checkpoint, row.checkpoint_sha256), (config, row.config_sha256)):
        found = _sha256(path)
        if found != wanted:
            raise ConversionError(
                f"{row.repo_id}'s {path.name} is not the file mnm checked (sha256 {found[:12]}…, "
                f"checked {wanted[:12]}…): it changed on the Hub since, and mnm converts only "
                "what it checked; an mnm update that checks the new one will take it."
            )


def converted_row(recipe: Recipe) -> ConvertedKit | None:
    return converted_by_file(recipe.files[0]) if recipe.files else None


def converted_files(recipe: Recipe, models_dir: Path) -> list[Path]:
    """The files of a recipe's converted folder, as `mnm disk` lists them (empty for any other)."""
    row = converted_row(recipe)
    held = recipe.local_paths.get(row.checkpoint) if row is not None else None
    if held is None:
        return []
    folder = models_dir / folder_of(held)
    return sorted(f for f in folder.rglob("*") if f.is_file()) if folder.is_dir() else []


def ensure_converted(
    recipe: Recipe, models_dir: Path, say: Callable[[str], None] | None = None
) -> Path | None:
    """Make (or keep) the folder audio.cpp loads, for a recipe of a converted checkpoint."""
    row = converted_row(recipe)
    if row is None:
        return None
    try:
        checkpoint = models_dir / recipe.local_paths[row.checkpoint]
        config = models_dir / recipe.local_paths[row.config]
    except KeyError as exc:
        raise ConversionError(f"{recipe.name} is not fully downloaded: {exc} is missing") from exc
    out = models_dir / folder_of(recipe.local_paths[row.checkpoint])
    stamp = out / STAMP
    done = (out / WEIGHTS).exists() and (out / CONFIG).exists()
    if done and _made_from(stamp) == _stamp(checkpoint, config):
        return out
    _checked(row, checkpoint, config)
    try:
        body = audiocpp_config(config.read_text(encoding="utf-8"), row.kit.family)
    except (ConversionError, OSError) as exc:
        raise ConversionError(f"{row.repo_id}'s {row.config}: {exc}") from exc
    if say:
        say(f"Converting {row.checkpoint} for audio.cpp (once, beside the download)")
    out.mkdir(parents=True, exist_ok=True)
    try:
        write_safetensors(checkpoint, out / WEIGHTS, split_qkv=row.split_qkv)
    except (CheckpointError, OSError) as exc:
        raise ConversionError(f"{row.checkpoint} could not be converted: {exc}") from exc
    (out / CONFIG).write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    stamp.write_text(json.dumps(_stamp(checkpoint, config)), encoding="utf-8")
    return out
