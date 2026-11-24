"""Civitai downloads: a version's file as a plain file under `<models>/civitai/<version>/`.

Each file downloads from its own link in the version's API answer (the bare version link
serves the version's *primary* file, which may be another one: a pickle beside the
safetensors). Civitai redirects to its CDN, which honours Range, so an interrupted download
resumes from its `.part` file; a 206 whose range does not start where the `.part` ends is
never appended (the `.part` starts over). A token (Authorization: Bearer) unlocks files that
need a login; httpx drops it on the redirect to the CDN's own host.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable
from pathlib import Path

import httpx

from makenomistakes import __version__
from makenomistakes.download_progress import (
    STALL_SECONDS,
    Transfer,
    copy_chunks,
    progress_bar,
    worded_timeout,
)
from makenomistakes.hub.civitai import SUBDIR, CivitaiClient, CivitaiError, refused
from makenomistakes.locking import shared_lock

Fetcher = Callable[[str, str, str, Path, str | None], Path]
CHUNK = 1 << 20
TIMEOUT = httpx.Timeout(30.0, read=STALL_SECONDS)  # a minute without a byte: stalled
_RANGE_RE = re.compile(r"bytes (\d+)-\d+/(\d+|\*)")
_DONE_RE = re.compile(r"bytes \*/(\d+)")


def civitai_path(models_dir: Path, version: str, filename: str) -> Path:
    """The file's place; ValueError when the name would leave its version's folder."""
    folder = models_dir / SUBDIR / version
    path = folder / filename
    if path.parent != folder or filename in ("", ".", "..") or not version.isdigit():
        raise ValueError(f"not a Civitai file name: {version}/{filename}")
    return path


def civitai_lookup(models_dir: Path) -> Callable[[str, str, str, Path], Path | None]:
    """A CacheLookup for `civitai/<model id>` repos (the engine's cache folder is unused)."""

    def lookup(_repo_id: str, filename: str, version: str, _cache: Path) -> Path | None:
        path = civitai_path(models_dir, version, filename)
        return path if path.is_file() else None

    return lookup


def civitai_fetcher(
    models_dir: Path,
    token: str | None,
    client: httpx.Client | None = None,
    api: CivitaiClient | None = None,
) -> Fetcher:
    """A Fetcher for `civitai/<model id>` repos: `revision` is the version id. `client` and
    `api` can stand in for the clients; by default one API client per fetcher (its
    version answers cached across the recipe's files) and one connection per download."""
    civitai = api or CivitaiClient(token)

    def fetch(_repo_id: str, filename: str, version: str, _cache: Path, _hf: str | None) -> Path:
        try:
            dest = civitai_path(models_dir, version, filename)
        except ValueError as exc:
            raise OSError(str(exc)) from exc
        dest.parent.mkdir(parents=True, exist_ok=True)
        with shared_lock(dest.parent)():  # a second download of it waits, then finds it whole
            return one(dest, filename, version)

    def one(dest: Path, filename: str, version: str) -> Path:
        if dest.is_file():
            return dest
        part = dest.with_name(dest.name + ".part")
        try:
            listed = civitai.file(version, filename)
        except CivitaiError as exc:
            raise OSError(str(exc)) from exc
        url = listed.url(int(version))
        headers = {"User-Agent": f"makenomistakes/{__version__}"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        have = part.stat().st_size if part.exists() else 0
        if have:
            headers["Range"] = f"bytes={have}-"
        with contextlib.ExitStack() as stack:
            http = client or stack.enter_context(
                httpx.Client(follow_redirects=True, timeout=TIMEOUT)
            )
            try:
                done = _stream(http, url, headers, part, have, bool(token), listed.size_bytes)
            except httpx.HTTPError as exc:
                got = part.stat().st_size if part.exists() else have
                raise worded_timeout(exc, Transfer("Civitai", listed.size_bytes), got) from exc
        if not done:  # the server would not resume where the .part ends: start it over
            part.unlink(missing_ok=True)
            raise OSError("Civitai would not resume the download; start again to fetch it whole.")
        _check_size(part, listed.size_bytes)
        part.rename(dest)
        return dest

    return fetch


SIZE_SLACK = 2048  # the API lists sizes in KB with rounding


def _check_size(part: Path, expected: int) -> None:
    """A download that ended early (a 200 with no length, cut) is never taken as the file:
    a short `.part` stays to resume, a longer one goes."""
    have = part.stat().st_size
    if not expected or abs(have - expected) <= SIZE_SLACK:
        return
    if have < expected:
        raise OSError(
            f"Civitai's download stopped at {have:,} of {expected:,} bytes; start again to resume."
        )
    part.unlink()
    raise OSError("Civitai sent more than the file's listed size; start again to fetch it whole.")


def _stream(
    http: httpx.Client,
    url: str,
    headers: dict[str, str],
    part: Path,
    have: int,
    token: bool,
    size: int = 0,
) -> bool:
    """Write the answer to `.part` (a bar on a terminal; a minute without a chunk stops it);
    False when it cannot continue the `.part` as it is."""
    with http.stream("GET", url, headers=headers, follow_redirects=True) as response:
        status = response.status_code
        if status in (401, 403):
            raise OSError(refused(status, token, url))
        if status == httpx.codes.REQUESTED_RANGE_NOT_SATISFIABLE:
            total = _DONE_RE.fullmatch(response.headers.get("Content-Range", ""))
            return total is not None and int(total.group(1)) == have  # all there already
        if response.is_error:
            raise OSError(f"Civitai answered HTTP {status} for {url}")
        mode, start = "wb", 0
        if status == httpx.codes.PARTIAL_CONTENT:
            span = _RANGE_RE.fullmatch(response.headers.get("Content-Range", ""))
            if span is None or int(span.group(1)) != have:
                return False
            mode, start = "ab", have
        name = part.name.removesuffix(".part")
        with part.open(mode) as out, progress_bar(name, size, start) as bar:
            copy_chunks(response.iter_bytes(CHUNK), out, bar)
    return True
