"""The files vLLM needs from a transformers repo: the weights and the small files beside them.

vLLM loads a model folder: `config.json`, the tokenizer files, the generation config and the
`*.safetensors` weights (sharded with an index, or one file). Mistral repos ship the same
weights twice, as `consolidated.safetensors` and as sharded `model-*.safetensors`: the sharded
set is what transformers' loader reads. Subfolders (`original/` .pth files, GGUF exports,
ONNX) are never fetched.

A model folder can sit anywhere in a repo: a pasted folder is used as it
is; otherwise the root, else every folder holding a `config.json` beside safetensors weights
is a version, the largest one planned first (`--version <folder>` picks another).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Protocol

SUPPORT_SUFFIXES = (".json", ".model", ".tiktoken", ".txt", ".jinja")
CONFIG_FILE = "config.json"


class _File(Protocol):
    @property
    def path(self) -> str: ...

    @property
    def size(self) -> int: ...


def vllm_files(files: Sequence[_File], folder: str | None) -> list[str]:
    """The files of one model folder (the repo root, or the pasted folder) vLLM loads; empty
    when the folder has no `config.json` or no safetensors weights."""
    prefix = f"{folder.strip('/')}/" if folder else ""
    here = [f for f in files if f.path.startswith(prefix) and "/" not in f.path[len(prefix) :]]
    weights = [f for f in here if f.path.endswith(".safetensors")]
    sharded = [f for f in weights if not PurePosixPath(f.path).name.startswith("consolidated")]
    weights = sharded or weights
    support = [f for f in here if f.path.endswith(SUPPORT_SUFFIXES)]
    if not weights or prefix + CONFIG_FILE not in {f.path for f in support}:
        return []
    return sorted(f.path for f in weights) + sorted(f.path for f in support)


def model_folders(files: Sequence[_File]) -> list[str]:
    """Every folder vLLM could load, largest weights first ("" is the repo root)."""
    sizes: dict[str, int] = {}
    configs = {str(PurePosixPath(f.path).parent) for f in files if _name(f) == CONFIG_FILE}
    for f in files:
        folder = str(PurePosixPath(f.path).parent)
        if f.path.endswith(".safetensors") and folder in configs:
            sizes[folder] = sizes.get(folder, 0) + f.size
    ordered = sorted(sizes, key=lambda folder: (-sizes[folder], folder))
    return ["" if folder == "." else folder for folder in ordered]


def pick_folder(files: Sequence[_File], pasted: str | None) -> tuple[list[str], list[str]]:
    """The served files and, when the repo holds several model folders and none was pasted,
    every folder's label (largest first, the one served first)."""
    if pasted or vllm_files(files, None):
        return vllm_files(files, pasted), []
    folders = model_folders(files)
    if not folders:
        return [], []
    return vllm_files(files, folders[0]), folders if len(folders) > 1 else []


def _name(f: _File) -> str:
    return PurePosixPath(f.path).name


def config_path(files: list[str]) -> str:
    """The folder's `config.json`, which also names the folder vLLM is pointed at."""
    return next(f for f in files if PurePosixPath(f).name == CONFIG_FILE)
