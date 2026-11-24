"""One file streamed to `<dest>.part`, resumed with Range where an earlier download stopped,
and kept only when its sha256 is the one its source lists: Ollama's blobs are
named by their digest, ModelScope lists each file's. A cut file is kept as `.part` to resume;
one that is whole but wrong goes. One download writes a folder at a time (the folder's
`shared_lock`): two appending to one `.part` could keep a corrupt file as verified."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from makenomistakes.download_progress import (
    Transfer,
    copy_chunks,
    progress_bar,
    worded_timeout,
)
from makenomistakes.locking import shared_lock

if TYPE_CHECKING:
    from hashlib import _Hash as Hash  # typeshed's name for a running digest

CHUNK = 1 << 20


Stream = Callable[[str, dict[str, str]], AbstractContextManager[httpx.Response]]


def fetch_verified(
    stream: Stream, url: str, dest: Path, sha256: str, size: int, source: str, agent: str
) -> Path:
    """`stream(url, headers)` opens the GET (the source client's connection, which can be swapped);
    `sha256` is the hex digest the file must have, `source` names the site in messages."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with shared_lock(dest.parent)():  # one writer per `.part`: its digest is the file's
        return _fetch(stream, url, dest, sha256, size, source, agent)


def _fetch(
    stream: Stream, url: str, dest: Path, sha256: str, size: int, source: str, agent: str
) -> Path:
    if dest.is_file():  # another download finished it while this one waited
        return dest
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    have = 0
    if part.exists():
        with part.open("rb") as kept:
            for chunk in iter(lambda: kept.read(CHUNK), b""):
                digest.update(chunk)
                have += len(chunk)
    if not (have and size and have >= size):  # a whole part (cut before its rename) is checked
        how = Transfer(source, size, agent)
        found = _download(stream, url, part, digest, have, how)
        if found is None and digest.hexdigest() != sha256.lower():  # 416, and not the file
            part.unlink()  # longer than the file, or another file: anew (`have` 0 never 416s)
            found = _download(stream, url, part, hashlib.sha256(), 0, how)
        digest = found or digest  # a 416 on the whole file (its size unknown) keeps it
    if digest.hexdigest() != sha256.lower():
        got = part.stat().st_size
        if size and got < size:
            raise OSError(
                f"{source}'s download stopped at {got:,} of {size:,} bytes; start again to resume."
            )
        part.unlink()
        raise OSError(f"{source}'s file did not match its sha256; start again to fetch it whole.")
    part.rename(dest)
    return dest


def _download(
    stream: Stream, url: str, part: Path, digest: Hash, have: int, how: Transfer
) -> Hash | None:
    """Append the rest to `part` (all of it when the server will not resume) and return the
    digest of the whole file; None when the server refuses the range (416). A bar shows it
    on a terminal; a minute without a chunk stops it (`download_progress`)."""
    source, agent, size = how.source, how.agent, how.size
    headers = {"User-Agent": agent}
    if have:
        headers["Range"] = f"bytes={have}-"
    try:
        with stream(url, headers) as response:
            if have and response.status_code == httpx.codes.REQUESTED_RANGE_NOT_SATISFIABLE:
                return None
            if have and response.status_code == httpx.codes.OK:
                digest = hashlib.sha256()  # no resume here: from the start
                part.unlink()
                have = 0
            elif response.is_error:
                raise OSError(f"{source} answered HTTP {response.status_code} for {url}")
            name = part.name.removesuffix(".part")
            with part.open("ab") as out, progress_bar(name, size, have) as bar:
                copy_chunks(response.iter_bytes(CHUNK), out, bar, digest.update)
    except httpx.HTTPError as exc:
        got = part.stat().st_size if part.exists() else have
        raise worded_timeout(exc, how, got) from exc
    return digest
