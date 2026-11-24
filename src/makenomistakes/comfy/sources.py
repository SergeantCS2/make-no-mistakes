"""Where a ComfyUI workflow comes from: a link to comfy.org, Civitai (a
`Workflows` version: a small zip), the Civitai Archive (Civitai's ids), Hugging Face, GitHub,
or a file (`.json`, a `.png` ComfyUI saved, a `.zip`).

Everything is read in memory with limits: a zip gives up only its `.json` files and PNGs (at
most `MAX_ENTRIES`, `MAX_BYTES` in all), never a path, never anything to run. Only these
sites are asked, and the Civitai token goes only to Civitai's API host.
"""

from __future__ import annotations

import io
import re
import struct
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import httpx

from makenomistakes.hub.civitai import CivitaiClient, CivitaiError, CivitaiRef, parse_civitai
from makenomistakes.hub.client import HUB_URL
from makenomistakes.hub.refs import GITHUB_HOSTS, HUB_HOSTS

MAX_BYTES = 20 * 2**20  # a workflow is kilobytes; a zip of a few, with preview pictures
MAX_ENTRIES = 50
COMFY_HOSTS = frozenset({"comfy.org", "www.comfy.org"})
HF_HOSTS = HUB_HOSTS
RAW_GITHUB = "raw.githubusercontent.com"
_DOWNLOAD = re.compile(r"/workflows/download/([A-Za-z0-9][A-Za-z0-9_.-]{0,120})\.json")
_PNG = b"\x89PNG\r\n\x1a\n"
_ZIP = b"PK\x03\x04"
WHERE = (
    "comfy.org/workflows, Civitai (a Workflows model), the Civitai Archive, Hugging Face, "
    "GitHub, or a .json, .png or .zip file"
)


class SourceError(ValueError):
    """Nothing importable at that reference; the message says why and what works."""


@dataclass(frozen=True)
class Found:
    """One workflow file, as fetched: its name (without the suffix) and its JSON bytes."""

    name: str
    data: bytes
    source: str


def fetch(reference: str, http: httpx.Client, civitai: CivitaiClient) -> list[Found]:
    """Every workflow at a link (a Civitai zip may hold several)."""
    text = reference.strip()
    url = urlparse(text if "://" in text else f"https://{text}")
    host = (url.hostname or "").lower()
    if host in COMFY_HOSTS:
        return _comfy_org(url.path, http, text)
    if host in HF_HOSTS:
        return _direct(_hf_resolve(url.path), http, text)
    if host in GITHUB_HOSTS:
        return _direct(_github_raw(url.path), http, text)
    if host == RAW_GITHUB:
        return _direct(f"https://{RAW_GITHUB}{url.path}", http, text)
    try:
        ref = parse_civitai(text)
    except CivitaiError as exc:
        raise SourceError(str(exc)) from exc
    if ref is not None:
        return _civitai(ref, civitai, text)
    raise SourceError(f"{text} is not a workflow link mnm reads: use {WHERE}.")


def from_file(path: Path) -> list[Found]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SourceError(f"Cannot read {path}: {exc.strerror or exc}.") from exc
    if len(data) > MAX_BYTES:
        raise SourceError(f"{path} is larger than {MAX_BYTES // 2**20} MiB: not a workflow.")
    return unpack(path.name, data, str(path))


def unpack(name: str, data: bytes, source: str) -> list[Found]:
    """A file's workflows: a zip's JSON files (and PNGs), a PNG's embedded one, or itself."""
    stem = PurePosixPath(name).stem or "workflow"
    if data.startswith(_ZIP):
        return _zip(data, source)
    if data.startswith(_PNG):
        found = png_workflow(data)
        if found is None:
            raise SourceError(f"{name} is a picture without a ComfyUI workflow in it.")
        return [Found(stem, found, source)]
    return [Found(stem, data, source)]


def _inflate(value: bytes) -> bytes:
    """A compressed iTXt chunk, at most MAX_BYTES once inflated (a bomb is refused)."""
    inflater = zlib.decompressobj()
    out = inflater.decompress(value, MAX_BYTES + 1)
    if len(out) > MAX_BYTES or inflater.unconsumed_tail:
        raise SourceError("The picture's workflow is larger than a workflow is.")
    return out


def png_workflow(data: bytes) -> bytes | None:
    """The `workflow` text ComfyUI's SaveImage writes into a PNG (`prompt` when only the API
    format is there); None when neither is. A malformed chunk is a SourceError."""
    try:
        return _png_texts(data)
    except (zlib.error, IndexError, struct.error, UnicodeDecodeError) as exc:
        raise SourceError(f"The picture's text chunks are damaged ({exc}).") from exc


def _png_texts(data: bytes) -> bytes | None:
    texts: dict[str, bytes] = {}
    offset = len(_PNG)
    while offset + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[offset : offset + 8])
        body = data[offset + 8 : offset + 8 + length]
        if kind == b"tEXt" and b"\0" in body:
            key, value = body.split(b"\0", 1)
            texts[key.decode("latin-1")] = value
        elif kind == b"iTXt" and b"\0" in body:
            key, rest = body.split(b"\0", 1)
            compressed, rest = rest[0], rest[2:]
            _lang, _, rest = rest.partition(b"\0")
            _translated, _, value = rest.partition(b"\0")
            texts[key.decode("latin-1")] = _inflate(value) if compressed else value
        elif kind == b"IEND":
            break
        offset += 12 + length
    return texts.get("workflow") or texts.get("prompt")


def _zip(data: bytes, source: str) -> list[Found]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise SourceError(f"{source} is not a readable zip ({exc}).") from exc
    entries = [
        info
        for info in archive.infolist()
        if not info.is_dir()
        and not info.filename.startswith("__MACOSX/")
        and not PurePosixPath(info.filename).name.startswith("._")
        and PurePosixPath(info.filename).suffix.lower() in (".json", ".png")
    ]
    if len(entries) > MAX_ENTRIES or sum(i.file_size for i in entries) > MAX_BYTES:
        raise SourceError(f"{source} holds more than a workflow pack does: not imported.")
    found: list[Found] = []
    for info in entries:
        name = PurePosixPath(info.filename.replace("\\", "/")).name  # a name, never a path
        body = archive.read(info)
        if name.lower().endswith(".png"):
            embedded = png_workflow(body)
            if embedded is not None:
                found.append(Found(PurePosixPath(name).stem, embedded, source))
            continue
        found.append(Found(PurePosixPath(name).stem, body, source))
    if not found:
        raise SourceError(f"{source} holds no workflow (.json) file.")
    return found


def _get(url: str, http: httpx.Client) -> bytes:
    try:
        with http.stream("GET", url) as response:
            if response.status_code in (401, 403):
                raise SourceError(f"{url} needs an account (HTTP {response.status_code}).")
            if response.is_error:
                raise SourceError(f"{url} answered HTTP {response.status_code}.")
            data = bytearray()
            for chunk in response.iter_bytes():
                data += chunk
                if len(data) > MAX_BYTES:
                    raise SourceError(f"{url} is larger than a workflow is: not imported.")
            return bytes(data)
    except httpx.HTTPError as exc:
        raise SourceError(f"Could not fetch {url}: {exc}") from exc


def default_http() -> httpx.Client:
    """The client an import from a link asks with (the CLI's and the web's; read at each
    import, so a caller can swap it)."""
    return httpx.Client(timeout=30.0, follow_redirects=True)


def _direct(url: str, http: httpx.Client, source: str) -> list[Found]:
    return unpack(PurePosixPath(urlparse(url).path).name, _get(url, http), source)


def _comfy_org(path: str, http: httpx.Client, source: str) -> list[Found]:
    """`/workflows/download/<id>.json`, or a workflow's page, whose download link it reads."""
    found = _DOWNLOAD.search(path)
    if found is None:
        page = _get(f"https://comfy.org{path}", http).decode("utf-8", "replace")
        found = _DOWNLOAD.search(page)
        if found is None:
            raise SourceError(f"{source} links no workflow file (is it a workflow's page?).")
    workflow_id = found.group(1)
    url = f"https://comfy.org/workflows/download/{workflow_id}.json?filename={workflow_id}"
    return [Found(workflow_id, _get(url, http), source)]


def _hf_resolve(path: str) -> str:
    parts = [p for p in path.split("/") if p]
    for section in ("blob", "resolve"):
        if section in parts:
            at = parts.index(section)
            if at >= 2 and len(parts) > at + 2:
                repo = "/".join(parts[:at])
                rest = "/".join(parts[at + 1 :])
                return f"{HUB_URL}/{repo}/resolve/{rest}"  # the mirror where HF_ENDPOINT names one
    raise SourceError("A Hugging Face link to a workflow names its file: …/blob/main/<file>.json")


def _github_raw(path: str) -> str:
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 5 and parts[2] in ("blob", "raw"):
        return f"https://{RAW_GITHUB}/{parts[0]}/{parts[1]}/{'/'.join(parts[3:])}"
    raise SourceError("A GitHub link to a workflow names its file: …/blob/main/<file>.json")


def _civitai(ref: CivitaiRef, civitai: CivitaiClient, source: str) -> list[Found]:
    try:
        resolved = civitai.resolve(ref)
    except CivitaiError as exc:
        raise SourceError(str(exc)) from exc
    if resolved.type.lower() != "workflows":
        raise SourceError(
            f"{source} is a Civitai {resolved.type or 'model'}, not a workflow: `mnm add` it."
        )
    version = resolved.version
    files = [
        f
        for f in version.files
        if PurePosixPath(f.name).suffix.lower() in (".zip", ".json", ".png")
    ]
    if not files:
        raise SourceError(f"Civitai version {version.id} has no workflow file.")
    found: list[Found] = []
    for file in files:
        account = SourceError(
            f"{file.name}'s creator requires a Civitai account: add a key with `mnm setup` "
            "(CIVITAI_TOKEN), then import again."
        )
        try:
            data = _get(file.url(version.id), civitai.http)  # bounded, as every fetch is
        except SourceError as exc:
            raise (account if "needs an account" in str(exc) else exc) from None
        if b'"Unauthorized"' in data[:200]:
            raise account
        found += unpack(file.name, data, source)
    return found
