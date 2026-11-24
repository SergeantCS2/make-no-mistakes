"""Download repo files into the Hugging Face cache layout under the models directory.

Using the cache layout (not `--local-dir`) keeps resume working through `*.incomplete`
blobs, which is the lesson from Odysseus's issue tracker. Returns paths relative to the
models directory so the recipe can map them into the container.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from makenomistakes.download_civitai import civitai_fetcher, civitai_lookup
from makenomistakes.download_progress import FileLine
from makenomistakes.hub.civitai import is_civitai
from makenomistakes.hub.client import hub_token
from makenomistakes.hub.elsewhere import elsewhere
from makenomistakes.recipes.model import HF_CACHE_SUBDIR, Recipe
from makenomistakes.recipes.registry import spec_for

Fetcher = Callable[[str, str, str, Path, str | None], Path]
CacheLookup = Callable[[str, str, str, Path], Path | None]
# audio.cpp canonicalises every path it opens and reads the format from the extension, so a
# GGUF reached through the cache's `snapshots/<rev>/` link (a link into `blobs/<sha>`) is
# refused as "unsupported tensor source format" (CPU build, 2026-10-15). Its packages are
# fetched as plain files under `voices/<org>--<repo>/<path in the repo>` instead.
VOICES_SUBDIR = "voices"


def hub_cache_lookup(repo_id: str, filename: str, revision: str, cache_dir: Path) -> Path | None:
    """The `snapshots/<rev>/<file>` link when an earlier download left the file in the cache
    (another instance of the same files, a removed one): read from disk, never the Hub."""
    from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415  (as hub_fetcher)

    found = try_to_load_from_cache(repo_id, filename, cache_dir=cache_dir, revision=revision)
    # a link whose blob was deleted to free disk space is not a download
    return Path(found) if isinstance(found, str) and Path(found).exists() else None


def hub_fetcher(
    repo_id: str, filename: str, revision: str, cache_dir: Path, token: str | None
) -> Path:
    """Default fetcher backed by huggingface_hub (imported lazily; it is heavy).

    The returned path is the `snapshots/<rev>/<filename>` symlink, not the blob it points
    at: the filename must survive so llama.cpp can find shards and projectors by name, and
    the symlink is relative, so it also resolves inside the container mount.
    """
    from huggingface_hub import hf_hub_download  # noqa: PLC0415  (0.5 s import; CLI startup)

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            token=hub_token(token),  # none to a mirror
        )
    )


def hub_plain_fetcher(
    repo_id: str, filename: str, revision: str, dest_dir: Path, token: str | None
) -> Path:
    """A file as itself under `dest_dir/<repo folder>/<filename>` (huggingface_hub's
    `local_dir`, which keeps resume through its own `.cache/` beside the files)."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415  (as hub_fetcher)

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=dest_dir / plain_folder(repo_id),
            token=hub_token(token),  # none to a mirror
        )
    )


def plain_folder(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def plain_lookup(repo_id: str, filename: str, revision: str, dest_dir: Path) -> Path | None:
    path = dest_dir / plain_folder(repo_id) / filename
    return path if path.is_file() else None


@dataclass(frozen=True)
class _Store:
    """One of the two places a recipe's files live (the engine row says which)."""

    subdir: str
    plain: bool


def _store(recipe: Recipe) -> _Store:
    plain = spec_for(recipe).plain_files
    return _Store(VOICES_SUBDIR if plain else HF_CACHE_SUBDIR, plain)


def wanted_files(recipe: Recipe) -> list[tuple[str, str, str, str]]:
    """Every file the recipe runs: key, repo, path in the repo, revision."""
    own = recipe.own_files
    up = recipe.upscaler
    return [
        *((f, recipe.repo_id, f, recipe.revision) for f in own),
        *((c.key, c.repo_id, c.path, c.revision) for c in recipe.companions),
        *((lora.key, lora.repo_id, lora.path, lora.revision) for lora in recipe.loras),
        *([(up.key, up.repo_id, up.path, up.revision)] if up else []),
    ]


def download_recipe_files(
    recipe: Recipe,
    models_dir: Path,
    token: str | None,
    fetch: Fetcher | None = None,
    on_file: Callable[[str], None] | None = None,
    checkpoint: Callable[[Recipe], object] | None = None,
    civitai_token: str | None = None,
) -> Recipe:
    """Fetch every file the recipe needs; returns a copy with `local_paths` filled in.

    Every file goes through `fetch`, which for huggingface_hub fetches only what is missing
    or changed on the Hub (a recipe's revision is usually `main`, which moves). `on_file` says
    which files are already here (else a second instance of one model would print
    "Downloading" for files it has). `checkpoint` is called with the partially-filled
    recipe after each file, so a failure on a later file keeps the earlier ones recorded.
    `fetch` left unset is huggingface_hub's, into the layout the engine row asks for (the
    cache, or plain files); an injected one (tests) serves both layouts.
    """
    store = _store(recipe)
    cache_dir = (models_dir / store.subdir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    hub_pair = (
        plain_lookup if store.plain else hub_cache_lookup,
        fetch or (hub_plain_fetcher if store.plain else hub_fetcher),
    )
    # Civitai files (`civitai/<model id>` repos) come from Civitai into their own folder; an
    # injected fetcher (tests) serves them too.
    civitai_pair = (civitai_lookup(models_dir), fetch or civitai_fetcher(models_dir, civitai_token))
    # Ollama's and ModelScope's files come from there as plain files, checked by sha256

    def pair(repo_id: str) -> tuple[CacheLookup, Fetcher]:
        if is_civitai(repo_id):
            return civitai_pair
        if (where := elsewhere(repo_id)) is not None:
            return where.lookup(models_dir), fetch or where.fetcher(models_dir)
        return hub_pair

    current = recipe
    for key, repo_id, repo_file, revision in wanted_files(recipe):
        known = current.local_paths.get(key)
        if known is not None and (models_dir / known).exists():
            continue
        label = repo_file if repo_id == recipe.repo_id else f"{repo_id} {repo_file}"
        look, get = pair(repo_id)
        if on_file:
            here = look(repo_id, repo_file, revision, cache_dir) is not None
            on_file(
                FileLine(
                    f"Already downloaded: {label} (checking the Hub)"
                    if here
                    else f"Downloading {label}"
                )
            )
        path = get(repo_id, repo_file, revision, cache_dir, token)
        relative = str(path.absolute().relative_to(models_dir.resolve()))
        current = current.model_copy(update={"local_paths": {**current.local_paths, key: relative}})
        if checkpoint:
            checkpoint(current)
    return current


@dataclass(frozen=True)
class CacheStatus:
    files: int
    cached: int  # of them already in the cache
    cached_bytes: int


def cache_status(
    recipe: Recipe, models_dir: Path, lookup: CacheLookup = hub_cache_lookup
) -> CacheStatus:
    """What the store already holds of the recipe's files, so `add` claims only the rest."""
    found = present_files(recipe, models_dir, lookup)
    here = [path for path in found.values() if path is not None]
    return CacheStatus(len(found), len(here), sum(path.stat().st_size for path in here))


def present_files(
    recipe: Recipe, models_dir: Path, lookup: CacheLookup = hub_cache_lookup
) -> dict[str, Path | None]:
    """Each file the recipe wants (by its key), where the store holds it already, read from
    disk only; None when it does not (`mnm disk` counts a file a recipe awaits as listed)."""
    store = _store(recipe)
    if store.plain:
        lookup = plain_lookup
    cache_dir = (models_dir / store.subdir).resolve()

    def look(repo: str) -> CacheLookup:
        where = elsewhere(repo)
        if is_civitai(repo):
            return civitai_lookup(models_dir)
        return where.lookup(models_dir) if where else lookup

    return {
        key: look(repo)(repo, path, rev, cache_dir) for key, repo, path, rev in wanted_files(recipe)
    }
