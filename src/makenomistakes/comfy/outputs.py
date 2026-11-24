"""What a workflow run leaves: every save node pointed under the run's own folder
in ComfyUI's output, the files ComfyUI's history names for a prompt, and each one moved from
ComfyUI's (mounted) folder into the run's, one copy."""

from __future__ import annotations

import contextlib
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

from makenomistakes.comfy.answers import HISTORY, OutputFile


def history_outputs(http: httpx.Client, address: str, prompt_id: str) -> list[OutputFile]:
    """The files ComfyUI's history names for this prompt (none when it has no entry)."""
    response = http.get(f"{address}/history/{prompt_id}")
    history = HISTORY.validate_json(response.content or b"{}")
    entry = history.get(prompt_id)
    if entry is None:
        return []
    return [
        item
        for node in entry.outputs.values()
        for item in (*node.images, *node.gifs, *node.videos, *node.audio)
    ]


def collect(base: Path, folder: Path, run_id: str, files: list[OutputFile]) -> list[Path]:
    """Each output moved from ComfyUI's folder under `base` (mounted: no copy over HTTP) into
    the run's `folder`; a preview in its temp folder is copied."""
    made = []
    for item in files:
        kind = "output" if item.type == "output" else "temp"
        where = base / kind
        if where.is_symlink():
            continue  # ComfyUI's folder itself replaced by a link: nothing is followed
        root = base.resolve() / kind
        source = (where / item.subfolder / item.filename).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            continue  # a name that leaves ComfyUI's folder is never followed
        target = folder / _free(folder, PurePosixPath(item.filename).name)
        if kind == "output":
            shutil.move(str(source), target)
        else:
            shutil.copy2(source, target)
        made.append(target)
    _prune(base / "output" / run_id)
    return made


def own_prefixes(prompt: dict[str, dict[str, Any]], run_id: str) -> list[str]:
    """Every save node writes under the run's own folder (`<run id>/<its prefix>`): an
    identical second run then still saves (ComfyUI's cache served the whole prompt, and the
    files it named had been moved into the first run: "frames: no files"), and every file in
    ComfyUI's output folder says which run made it. Any node with a `filename_prefix` it was
    given as text (SaveImage, SaveAnimatedWEBP/PNG, VHS_VideoCombine, SaveVideo, …), its `..`
    segments made `_` and a leading `/` dropped, so every file lands under the run's folder. A
    prefix wired from another node is left alone: the ids of those nodes come back, for the run
    to say so."""
    wired = []
    for node, entry in prompt.items():
        inputs = entry.get("inputs", {})
        given = inputs.get("filename_prefix")
        if isinstance(given, str):
            inputs["filename_prefix"] = f"{run_id}/{_inside(given)}"
        elif given is not None:
            wired.append(node)
    return wired


def _inside(prefix: str) -> str:
    """A prefix that stays under the run's folder: `..` segments become `_`, empty and `.`
    ones go (a leading `/`, `a//b`); nothing left is ComfyUI's own default."""
    parts = [p for p in prefix.replace("\\", "/").split("/") if p not in ("", ".")]
    return "/".join("_" if p == ".." else p for p in parts) or "ComfyUI"


def wired_words(nodes: list[str]) -> str:
    """The run's one line about save nodes whose file name another node sets."""
    which = (
        f"Node {nodes[0]}'s file name is"
        if len(nodes) == 1
        else f"Nodes {', '.join(nodes)} have their file names"
    )
    return f"{which} set by another node: an identical run again may make no files."


def _prune(folder: Path) -> None:
    """The run's folder under ComfyUI's output, once its files are moved: removed when
    empty, its own empty subfolders first (a prefix may name one: `video/ComfyUI`)."""
    if folder.is_symlink() or not folder.is_dir():
        return
    for inner in sorted(folder.rglob("*"), key=lambda p: -len(p.parts)):
        if inner.is_dir() and not inner.is_symlink():
            with contextlib.suppress(OSError):
                inner.rmdir()
    with contextlib.suppress(OSError):
        folder.rmdir()


def _free(folder: Path, name: str) -> str:
    stem, suffix = PurePosixPath(name).stem, PurePosixPath(name).suffix
    candidate, number = name, 2
    while (folder / candidate).exists():
        candidate = f"{stem}-{number}{suffix}"
        number += 1
    return candidate
