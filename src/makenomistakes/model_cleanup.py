"""`mnm rm <name> --files`: delete the downloads only this instance uses.

Instances share files: two profiles of one model, every SDXL checkpoint's fp16-fix VAE, the T5
encoder of Wan and FLUX, the upscaler. A file another recipe lists (by its path, or by the
cache blob its link points at) stays. Only files inside the models folder's download areas
are touched: the Hugging Face cache (`hf/`, a snapshot link and the blob behind it), the
voice packages (`voices/`) and Civitai's, Ollama's and ModelScope's files, plain files (a
cut download's `.part` too). Anything that resolves elsewhere is left alone. With huggingface_hub
2.x's shared blob store the file sits behind two links (`<repo>/blobs/<etag>` → `<hf>/blobs/<2
hex>/<xet hash>`): both links go with it, and the store's `.refs` and `.lock` beside it.
"""

from __future__ import annotations

import contextlib
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from makenomistakes.download import VOICES_SUBDIR, wanted_files
from makenomistakes.download_civitai import civitai_path
from makenomistakes.hub.civitai import SUBDIR as CIVITAI_SUBDIR
from makenomistakes.hub.civitai import is_civitai
from makenomistakes.hub.converted import converted_by_file, folder_of
from makenomistakes.hub.elsewhere import SOURCES, elsewhere
from makenomistakes.locking import LOCK_FILE
from makenomistakes.recipes import Recipe
from makenomistakes.recipes.model import HF_CACHE_SUBDIR

# huggingface_hub 2.x's shared blob store keeps these beside each blob (`utils/_shared_blobs`)
SHARED_MANIFEST = ".refs"
SHARED_LOCK = ".lock"
MAX_HOPS = 8


@dataclass
class Cleanup:
    """What `delete_own_files` did: bytes freed, files deleted, files kept because shared."""

    freed_bytes: int = 0
    deleted: list[str] = field(default_factory=list)
    shared: list[str] = field(default_factory=list)


def delete_own_files(
    recipe: Recipe,
    others: list[Recipe],
    models_dir: Path,
    linked: frozenset[str] = frozenset(),
    dry_run: bool = False,
) -> Cleanup:
    """`others` is every other saved recipe, read before `recipe` left the store. A file is
    another's when that recipe wants the same repo file (downloaded or not yet) or lists a
    path whose link resolves to the same blob. `linked`: files ComfyUI workflows use
    (relative to the models folder), kept the same way. `dry_run` (the disk policy)
    says what would go and frees nothing."""
    areas = _areas(models_dir)
    wanted = {
        _file(repo, path, rev) for other in others for _k, repo, path, rev in wanted_files(other)
    }
    in_use = {rel for other in others for rel in other.local_paths.values()} | set(linked)
    in_use |= {other.draft.path for other in others if other.draft is not None}
    in_use_targets = {_target(models_dir / rel) for rel in in_use} - {None}
    done = Cleanup()
    counted: set[Path] = set()  # one file under two keys counts once (a dry run sees it twice)
    for key, repo, path, revision in wanted_files(recipe):
        rel = recipe.local_paths.get(key)
        if rel is None:  # never finished: a cut download's `.part` goes
            if _file(repo, path, revision) not in wanted and not dry_run:
                _delete_part(models_dir, repo, path, revision, areas, done)
            continue
        link = models_dir / rel
        target = _target(link)
        if target is None or target in counted:
            continue  # gone already
        counted.add(target)
        if _file(repo, path, revision) in wanted or rel in in_use or target in in_use_targets:
            done.shared.append(rel)
            continue
        _delete(rel, link, target, areas, done, dry_run)
        if converted_by_file(path) is not None:  # The folder made from it goes too
            _delete_converted(models_dir, folder_of(rel), areas, done, dry_run)
    return done


def _delete_converted(
    models_dir: Path, rel: PurePosixPath, areas: list[Path], done: Cleanup, dry_run: bool
) -> None:
    folder = models_dir / rel
    if not folder.is_dir() or folder.is_symlink():
        return
    if not any(folder.resolve().is_relative_to(area) for area in areas):
        return  # not a folder this runner made
    done.freed_bytes += sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
    done.deleted.append(f"{rel}/")
    if dry_run:
        return
    shutil.rmtree(folder)
    _prune(folder.parent, areas)  # `converted/`, then the package folder when it empties


def _delete(
    rel: str, link: Path, target: Path, areas: list[Path], done: Cleanup, dry_run: bool = False
) -> None:
    """One download gone, with the folders it leaves empty; never outside the areas."""
    if not all(
        any(p.is_relative_to(area) for area in areas) for p in (target, link.parent.resolve())
    ):
        return  # not a download this runner made
    done.freed_bytes += target.stat().st_size
    if dry_run:
        done.deleted.append(rel)
        return
    hops = _hops(link)  # read before anything goes: the links on the way to the file
    target.unlink()
    for hop in [link, *hops]:  # every link that led there, now leading nowhere
        if hop.is_symlink() and any(hop.parent.resolve().is_relative_to(a) for a in areas):
            hop.unlink()
    if _in_shared_store(target):  # the store's own files for it
        for name in (f"{target.name}{SHARED_MANIFEST}", f"{target.name}{SHARED_LOCK}"):
            (target.parent / name).unlink(missing_ok=True)
    done.deleted.append(rel)
    _prune(target.parent, areas)  # `blobs/` first, so the repo folder can empty
    for folder in dict.fromkeys(hop.parent for hop in [*hops, link]):
        _prune(folder, areas)


def _in_shared_store(target: Path) -> bool:
    """`<hf>/blobs/<2 hex>/<64 hex>`: a file of huggingface_hub 2.x's shared blob store."""
    prefix = target.parent
    return (
        prefix.parent.name == "blobs"
        and len(prefix.name) == 2
        and target.name.startswith(prefix.name)
        and len(target.name) == 64
    )


def _hops(link: Path) -> list[Path]:
    """The links a snapshot link passes on the way to its file, the link itself left out:
    huggingface_hub 2.x links `<repo>/blobs/<etag>` on to its shared store, `<hf>/blobs/<2
    hex>/<xet hash>`, deduplicating a file across repos (its `.refs` beside it lists the repo
    links). A loop or a broken chain ends the walk."""
    found: list[Path] = []
    current = link
    while current.is_symlink() and len(found) < MAX_HOPS:
        following = current.parent / os.readlink(current)
        if not following.is_symlink():
            break
        found.append(following)
        current = following
    return found


def _delete_part(
    models_dir: Path, repo: str, path: str, revision: str, areas: list[Path], done: Cleanup
) -> None:
    """The `.part` a cut plain-file download left (Civitai, Ollama, ModelScope resume from
    it); the Hub's own unfinished blobs are huggingface_hub's to manage."""
    try:
        if is_civitai(repo):
            dest = civitai_path(models_dir, revision, path)
        elif (source := elsewhere(repo)) is not None:
            dest = source.place(models_dir, repo, path, revision)
        else:
            return
    except ValueError:  # a name that would leave its folder: nothing of ours
        return
    part = dest.with_name(dest.name + ".part")
    if part.is_file() and not part.is_symlink():
        _delete(str(part.relative_to(models_dir)), part, part.resolve(), areas, done)


def download_areas(models_dir: Path) -> list[Path]:
    """The folders mnm downloads into, resolved: the only places it ever deletes from."""
    return _areas(models_dir)


def file_behind(path: Path) -> Path | None:
    """The file a cache link (or a plain path) reaches; None when there is none."""
    return _target(path)


def _areas(models_dir: Path) -> list[Path]:
    subdirs = (HF_CACHE_SUBDIR, VOICES_SUBDIR, CIVITAI_SUBDIR, *(s.subdir for s in SOURCES))
    return [(models_dir / sub).resolve() for sub in subdirs]


def _target(path: Path) -> Path | None:
    """The file behind a cache link (or the file itself); None when there is none."""
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    return resolved if resolved.is_file() else None


def _prune(folder: Path, areas: list[Path]) -> None:
    """Remove folders the deletion left empty, up to (never including) the download area. A
    download's `.lock` (`locking.shared_lock`, held only while it writes) alone counts as
    empty: a download under way has its `.part` there too."""
    current = folder
    while current.resolve() not in areas and any(
        current.resolve().is_relative_to(area) for area in areas
    ):
        with contextlib.suppress(OSError):
            if [entry.name for entry in current.iterdir()] == [LOCK_FILE]:
                (current / LOCK_FILE).unlink()
            current.rmdir()  # only an empty folder goes
        if current.exists():
            return
        current = current.parent


def _file(repo: str, path: str, revision: str) -> tuple[str, str, str]:
    """One file another recipe may want: a Hub path at any revision (the cache keeps one
    copy), a Civitai file only within its own version (each has its own folder)."""
    pinned = is_civitai(repo) or elsewhere(repo) is not None  # a folder per revision there
    return (repo, path, revision if pinned else "")


def delete_files(rels: list[str], in_use: set[str], models_dir: Path) -> Cleanup:
    """Delete these downloads (relative to the models folder) unless something in `in_use`
    resolves to the same file; only inside the download areas, as `delete_own_files` does
    (a workflow's files when the workflow is removed with `--files`)."""
    areas = _areas(models_dir)
    in_use_targets = {_target(models_dir / rel) for rel in in_use} - {None}
    done = Cleanup()
    for rel in dict.fromkeys(rels):
        link = models_dir / rel
        target = _target(link)
        if target is None:
            continue
        if rel in in_use or target in in_use_targets:
            done.shared.append(rel)
            continue
        _delete(rel, link, target, areas, done)
    return done
