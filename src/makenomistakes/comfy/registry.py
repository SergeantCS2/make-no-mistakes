"""The public Comfy Registry (`api.comfy.org`): a pack's node, its versions, which pack provides
a node class, and a version's zip from the Registry's own CDN only. Read only, no account; every
answer typed at the edge."""

from __future__ import annotations

import re
from collections.abc import Callable
from urllib.parse import quote, urlparse

import httpx
from pydantic import BaseModel, ValidationError, field_validator

REGISTRY = "https://api.comfy.org"
DOWNLOAD_HOSTS = frozenset({"cdn.comfy.org"})
MAX_ZIP = 100 * 2**20  # the largest vetted zip is 19 MB
MAX_HOPS = 5  # redirects a download follows, each to the CDN
_PACK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")


class PackError(ValueError):
    """A pack mnm will not approve or install, worded for the user."""


def folder_of(pack: str) -> str:
    """The pack's folder under `custom_nodes/`: its Registry id (or a GitHub repository's
    name), checked."""
    if not _PACK_ID.match(pack) or ".." in pack:
        raise PackError(f"'{pack}' is not a pack's name (letters, digits, '.', '_' and '-').")
    return pack


class RegistryPublisher(BaseModel):
    id: str = ""
    name: str = ""
    status: str = ""


class RegistryVersion(BaseModel):
    version: str
    status: str = ""
    status_reason: str = ""
    deprecated: bool = False
    downloadUrl: str = ""  # noqa: N815 - the Registry's name
    dependencies: list[str] | None = None


class RegistryNode(BaseModel):
    id: str
    name: str = ""
    repository: str = ""  # shown as a link: only an http(s) address is kept
    license: str = ""
    status: str = ""
    downloads: int = 0
    github_stars: int = 0
    publisher: RegistryPublisher | None = None
    latest_version: RegistryVersion | None = None

    @field_validator("repository")
    @classmethod
    def _web_only(cls, value: str) -> str:
        """A publisher writes this field: a `javascript:` or other scheme never becomes a
        link on mnm's own pages."""
        scheme = urlparse(value.strip()).scheme.lower()
        return value.strip() if scheme in ("http", "https") else ""


def default_http() -> httpx.Client:
    """The client a `Registry` built without one asks with."""
    return httpx.Client(timeout=30.0, follow_redirects=True)


class Registry:
    """The public Comfy Registry (`api.comfy.org`): read only, no account."""

    def __init__(self, http: Callable[[], httpx.Client] | None = None) -> None:
        self._http = http or default_http  # read at construction, so a caller can swap it

    def _get(self, path: str) -> bytes | None:
        try:
            with self._http() as http:
                response = http.get(f"{REGISTRY}{path}")
        except httpx.HTTPError as exc:
            raise PackError(f"The Comfy Registry did not answer: {exc}") from exc
        if response.status_code == 404:
            return None
        if not response.is_success:
            raise PackError(f"The Comfy Registry answered {response.status_code}.")
        return response.content

    def node(self, pack: str) -> RegistryNode | None:
        body = self._get(f"/nodes/{folder_of(pack)}")
        return None if body is None else _parse(RegistryNode, body)

    def version(self, pack: str, version: str) -> RegistryVersion | None:
        if not re.fullmatch(r"[A-Za-z0-9.+_-]{1,64}", version):
            raise PackError(f"'{version}' is not a version.")
        body = self._get(f"/nodes/{folder_of(pack)}/versions/{version}")
        return None if body is None else _parse(RegistryVersion, body)

    def pack_of(self, node_class: str) -> str | None:
        """The Registry pack that provides a node class, when it knows one."""
        if not re.fullmatch(r"[^/?#]{1,200}", node_class):
            return None
        body = self._get(f"/comfy-nodes/{quote(node_class, safe='')}/node")
        return None if body is None else _parse(RegistryNode, body).id

    def download(self, url: str) -> bytes:
        """A version's zip, from the Registry's own CDN only (each redirect's hop checked
        too), at most `MAX_ZIP`."""
        try:
            with self._http() as http:
                for _hop in range(MAX_HOPS):
                    _on_the_cdn(url)
                    with http.stream("GET", url, follow_redirects=False) as response:
                        if not response.is_redirect:
                            return _zip_of(response)
                        url = str(response.url.join(response.headers.get("location", "")))
        except httpx.HTTPError as exc:
            raise PackError(f"The pack's download failed: {exc}") from exc
        raise PackError(f"The pack's download was redirected more than {MAX_HOPS} times.")


def _on_the_cdn(url: str) -> None:
    host = (urlparse(url).hostname or "").lower()
    if urlparse(url).scheme != "https" or host not in DOWNLOAD_HOSTS:
        raise PackError(f"The pack's download is not on the Registry's CDN: {url}")


def _zip_of(response: httpx.Response) -> bytes:
    if not response.is_success:
        raise PackError(f"The pack's download answered {response.status_code}.")
    data = bytearray()
    for chunk in response.iter_bytes():
        data += chunk
        if len(data) > MAX_ZIP:
            raise PackError(f"The pack's zip is larger than {MAX_ZIP // 2**20} MiB.")
    return bytes(data)


def _parse[T: BaseModel](model: type[T], body: bytes) -> T:
    try:
        return model.model_validate_json(body)
    except ValidationError as exc:
        raise PackError("The Comfy Registry answered something mnm cannot read.") from exc
