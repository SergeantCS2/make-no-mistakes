"""Ollama's GGUFs as plain files under `<models>/ollama/<namespace>--<name>/<tag>/`:
the blob streamed to `.part`, resumed with Range where a download stopped, and kept only
when its sha256 is the manifest's digest (Ollama names blobs by it, so a cut or changed file
is never taken as the model)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from makenomistakes import __version__
from makenomistakes.download_verified import fetch_verified
from makenomistakes.hub.ollama import SUBDIR, OllamaClient, OllamaError, ref_of

Fetcher = Callable[[str, str, str, Path, str | None], Path]


def ollama_path(models_dir: Path, repo_id: str, filename: str, tag: str) -> Path:
    """The file's place (arguments in a lookup's order: repo, file, revision); ValueError when a
    name would leave its folder."""
    ref = ref_of(repo_id, tag)
    folder = models_dir / SUBDIR / f"{ref.namespace}--{ref.name}" / tag
    path = folder / filename
    if (
        path.parent != folder
        or "/" in tag
        or tag in ("", ".", "..")
        or not filename.endswith(".gguf")
    ):
        raise ValueError(f"not an Ollama file name: {repo_id} {tag}/{filename}")
    return path


def ollama_fetcher(models_dir: Path, api: OllamaClient | None = None) -> Fetcher:
    """A Fetcher for `ollama/` repos: `revision` is the tag. `api` can stand in for the
    registry client."""
    registry = api or OllamaClient()

    def fetch(repo_id: str, filename: str, tag: str, _cache: Path, _token: str | None) -> Path:
        try:
            dest = ollama_path(models_dir, repo_id, filename, tag)
            ref = ref_of(repo_id, tag)
            resolved = registry.resolve(ref)
            url = registry.blob_url(ref, resolved.model.digest)
        except (ValueError, OllamaError) as exc:
            raise OSError(str(exc)) from exc
        return fetch_verified(
            registry.stream,
            url,
            dest,
            resolved.model.digest.removeprefix("sha256:"),
            resolved.model.size,
            "Ollama",
            f"makenomistakes/{__version__}",
        )

    return fetch
