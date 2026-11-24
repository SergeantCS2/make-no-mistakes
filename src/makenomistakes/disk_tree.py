"""The models folder as `mnm disk` reads it: folder sizes with links never followed and
a hard-linked file counted once, and the *unlisted* files: files in the download areas
(`model_cleanup.download_areas`: hf/, voices/, civitai/, ollama/, modelscope/) that no recipe,
workflow, LoRA or draft lists, by the same rules `mnm rm --files` uses to decide what is
another's. Nothing here deletes anything."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from makenomistakes.comfyui import SUBDIR as COMFY_SUBDIR
from makenomistakes.converted_models import converted_files
from makenomistakes.disk_types import FolderSize, UnlistedFile
from makenomistakes.download import VOICES_SUBDIR, present_files
from makenomistakes.hub.civitai import SUBDIR as CIVITAI_SUBDIR
from makenomistakes.hub.elsewhere import SOURCES
from makenomistakes.locking import LOCK_FILE
from makenomistakes.model_cleanup import download_areas, file_behind
from makenomistakes.music_api import VOICE_FOLDER
from makenomistakes.quality import WORK_FOLDER as QUALITY_FOLDER
from makenomistakes.recipes import Recipe
from makenomistakes.recipes.audiocpp import CONFIGS_SUBDIR
from makenomistakes.recipes.llamacpp_cache import CACHE_FOLDER
from makenomistakes.recipes.model import HF_CACHE_SUBDIR

# Under the models folder: where mnm downloads, then the folders of links it keeps (and the
# voice configs it renders, the voice library's recordings: never a download, never
# evicted, the saved prompts, and a quality tune's work). Anything else there is the user's.
DOWNLOAD_FOLDERS = (
    HF_CACHE_SUBDIR,
    CIVITAI_SUBDIR,
    *(source.subdir for source in SOURCES),
    VOICES_SUBDIR,
)
LINK_FOLDERS = ("views", "loras", "upscalers", COMFY_SUBDIR)
MNM_FOLDERS = (
    *DOWNLOAD_FOLDERS,
    *LINK_FOLDERS,
    CONFIGS_SUBDIR,
    VOICE_FOLDER,
    CACHE_FOLDER,
    QUALITY_FOLDER,
)
# The Hub cache's and its local-dir downloads' own bookkeeping, never a model file: its
# folders, and huggingface_hub 2.x's shared blob store's tag, marker, manifests and locks
# (`<hf>/blobs/<2 hex>/<xet hash>.refs`, `.lock`)
_BOOKKEEPING = frozenset({".locks", "refs", ".cache", ".no_exist"})
_BOOKKEEPING_FILES = frozenset({LOCK_FILE, "CACHEDIR.TAG", ".huggingface-shared-blobs"})
_BOOKKEEPING_SUFFIXES = (".refs", ".lock", ".tmp")
UNFINISHED = (".incomplete", ".part")


def tree_size(path: Path, seen: set[tuple[int, int]] | None = None) -> tuple[int, int]:
    """Bytes and regular files under `path` (or of the file itself), links not followed, a
    file reached twice (a hard link) counted once in `seen`; what cannot be read counts 0."""
    seen = set() if seen is None else seen
    size = files = 0
    for entry in _files(path):
        try:
            info = entry.lstat()
        except OSError:
            continue
        key = (info.st_dev, info.st_ino)
        if key in seen:
            continue
        seen.add(key)
        size += info.st_size
        files += 1
    return size, files


def _files(path: Path) -> Iterator[Path]:
    """Every regular file under `path` (itself when it is one), links not followed."""
    if path.is_symlink():
        return
    if path.is_file():
        yield path
        return
    for folder, _dirs, names in os.walk(path, followlinks=False, onerror=lambda _e: None):
        for name in names:
            entry = Path(folder, name)
            if not entry.is_symlink() and entry.is_file():
                yield entry


def folder_size(label: str, path: Path, seen: set[tuple[int, int]] | None = None) -> FolderSize:
    if not path.exists():
        return FolderSize(label=label, path=str(path), exists=False)
    size, files = tree_size(path, seen)
    return FolderSize(label=label, path=str(path), exists=True, size_bytes=size, files=files)


def models_folders(models_dir: Path) -> tuple[list[FolderSize], list[FolderSize]]:
    """mnm's folders under the models folder that exist, then every other entry there."""
    seen: set[tuple[int, int]] = set()
    ours = [
        folder_size(f"{name}/", models_dir / name, seen)
        for name in MNM_FOLDERS
        if (models_dir / name).exists()
    ]
    try:
        entries = sorted(models_dir.iterdir())
    except OSError:
        entries = []
    others = [
        folder_size(entry.name + ("/" if entry.is_dir() else ""), entry, seen)
        for entry in entries
        if entry.name not in MNM_FOLDERS and entry.name != LOCK_FILE
    ]
    return ours, others


def listed_targets(
    recipes: list[Recipe], linked: Iterable[tuple[str, str]], models_dir: Path
) -> dict[Path, list[str]]:
    """Every file something lists, resolved, with who lists it: each recipe's downloads (and a
    file it awaits that the store already holds), its draft's file, every workflow link
    (`linked`: the target, relative to the models folder, and who uses it)."""
    found: dict[Path, list[str]] = {}

    def add(target: Path | None, who: str) -> None:
        if target is not None and who not in found.setdefault(target, []):
            found[target].append(who)

    for recipe in recipes:
        for rel in recipe.local_paths.values():
            add(file_behind(models_dir / rel), recipe.name)
        for made in converted_files(recipe, models_dir):  # Its converted folder
            add(made.resolve(), recipe.name)
        if recipe.draft is not None:
            add(file_behind(models_dir / recipe.draft.path), recipe.name)
        for key, path in present_files(recipe, models_dir).items():
            if key not in recipe.local_paths and path is not None:
                add(file_behind(path), recipe.name)
    for rel, who in linked:
        add(file_behind(models_dir / rel), who)
    return found


def unlisted_files(models_dir: Path, listed: Iterable[Path]) -> list[UnlistedFile]:
    """Files in the download areas nothing lists, largest first: what `mnm rm` without
    `--files` left, a download cut short, a file copied in by hand."""
    known = set(listed)
    found = []
    for area in download_areas(models_dir):
        for entry in _files(area):
            relative = entry.relative_to(area)
            if _bookkeeping(entry, relative):
                continue
            if entry.resolve() in known:
                continue
            unfinished = entry.name.endswith(UNFINISHED)
            why = "a download that never finished" if unfinished else "no instance or workflow"
            try:
                size = entry.lstat().st_size
            except OSError:
                continue
            found.append(UnlistedFile(path=_shown(entry, models_dir), size_bytes=size, why=why))
    return sorted(found, key=lambda f: (-f.size_bytes, f.path))


def _shown(entry: Path, models_dir: Path) -> str:
    """Relative to the models folder when it is inside it (an area may be a link elsewhere)."""
    try:
        return str(entry.relative_to(models_dir.resolve()))
    except ValueError:
        return str(entry)


def _bookkeeping(entry: Path, relative: Path) -> bool:
    if entry.name in _BOOKKEEPING_FILES or entry.name.endswith(_BOOKKEEPING_SUFFIXES):
        return True
    return bool(_BOOKKEEPING & set(relative.parts[:-1]))
