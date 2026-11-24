"""ModelScope's files as plain files under `<models>/modelscope/<org>--<repo>/<revision>/`,
each kept only when its sha256 is the one ModelScope lists."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path, PurePosixPath

from makenomistakes import __version__
from makenomistakes.download_verified import fetch_verified
from makenomistakes.hub.modelscope import SUBDIR, ModelScopeClient, ModelScopeError, ref_of

Fetcher = Callable[[str, str, str, Path, str | None], Path]


def modelscope_path(models_dir: Path, repo_id: str, path: str, revision: str) -> Path:
    """The file's place (arguments in a lookup's order: repo, file, revision); ValueError when a
    name would leave its folder."""
    ref = ref_of(repo_id, revision)
    folder = models_dir / SUBDIR / f"{ref.org}--{ref.repo}" / revision
    parts = PurePosixPath(path).parts
    if not parts or any(p in ("", ".", "..") for p in parts) or "/" in revision or not revision:
        raise ValueError(f"not a ModelScope file name: {repo_id} {revision}/{path}")
    return folder.joinpath(*parts)


def modelscope_fetcher(models_dir: Path, api: ModelScopeClient | None = None) -> Fetcher:
    """A Fetcher for `modelscope/` repos; `api` can stand in for the API client."""
    client = api or ModelScopeClient()

    def fetch(repo_id: str, path: str, revision: str, _cache: Path, _token: str | None) -> Path:
        try:
            dest = modelscope_path(models_dir, repo_id, path, revision)
            ref = ref_of(repo_id, revision)
            listed = client.resolve(ref).file(path)
        except (ValueError, ModelScopeError) as exc:
            raise OSError(str(exc)) from exc
        if not listed.sha256:
            raise OSError(f"ModelScope lists no sha256 for {path}: mnm will not take it unchecked.")
        return fetch_verified(
            client.stream,
            client.file_url(ref, path),
            dest,
            listed.sha256,
            listed.size,
            "ModelScope",
            f"makenomistakes/{__version__}",
        )

    return fetch
