"""Minimal, typed client for the Hub endpoints we need.

    GET /api/models/{repo}?expand[]=...        -> HubModel
    GET /api/models/{repo}/tree/{rev}?recursive -> list[TreeEntry]
    GET /api/models?search=&filter=&sort=      -> list[SearchHit]
    GET /{repo}/resolve/{rev}/{path}            -> ranged bytes (GGUF header)

Shapes were verified against the live API in October 2026. Everything unknown is ignored
rather than rejected.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from typing import Annotated, Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, field_validator

from makenomistakes import __version__


def _or(kind: type, default: Callable[[], object]) -> Callable[[object], object]:
    """The Hub writes `null`, a string or a number where a shape is expected:
    anything that is not the shape reads as its empty value, never an error."""
    return lambda value: value if isinstance(value, kind) else default()


DictOrEmpty = Annotated[dict[str, Any], BeforeValidator(_or(dict, dict))]
ListOrEmpty = Annotated[list[str], BeforeValidator(_or(list, list))]
TextOrNone = Annotated[str | None, BeforeValidator(_or(str, lambda: None))]


def _raw_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "").strip().rstrip("/")


def hub_endpoint() -> str:
    """The Hub mnm asks and downloads from: `HF_ENDPOINT` when set (`https://hf-mirror.com`
    where huggingface.co is blocked, or a mirror under a path such as Artifactory's
    `https://host/api/hf`; huggingface_hub's downloads honour it too), else huggingface.co.
    Only an http(s) address is taken."""
    found = _raw_endpoint()
    return found if re.fullmatch(r"https?://[^/\s?#]+(/[^\s?#]*)?", found) else OFFICIAL_HUB


OFFICIAL_HUB = "https://huggingface.co"
HUB_URL = hub_endpoint()


def hub_token(token: str | None) -> str | bool | None:
    """What a Hub request may carry: the user's token for huggingface.co only. A mirror
    (`HF_ENDPOINT`, whatever its shape: huggingface_hub takes it as given) is another site,
    so it gets none, and False keeps huggingface_hub from sending a token of its own (the
    environment's, its login file's): Civitai's rule, a link alias never becomes a token
    recipient."""
    official = HUB_URL == OFFICIAL_HUB and _raw_endpoint() in ("", OFFICIAL_HUB)
    return token if official else False


MAX_TREE_PAGES = 100  # of 1,000 entries each
USER_AGENT = f"makenomistakes/{__version__}"
_EXPAND = (
    "gguf",
    "safetensors",
    "config",
    "cardData",
    "pipeline_tag",
    "library_name",
    "tags",
    "gated",
)


class HubError(RuntimeError):
    """Base class for Hub failures with a message meant for the user."""


class NotFoundError(HubError):
    pass


class GatedOrPrivateError(HubError):
    pass


class _Loose(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class GgufMeta(_Loose):
    total: int | None = None
    architecture: str | None = None
    context_length: int | None = None
    chat_template: str | None = None


class SafetensorsMeta(_Loose):
    parameters: Annotated[dict[str, int], BeforeValidator(_or(dict, dict))] = Field(
        default_factory=dict
    )
    total: int | None = None


class ConfigMeta(_Loose):
    architectures: ListOrEmpty = Field(default_factory=list)
    model_type: TextOrNone = None
    tokenizer_config: DictOrEmpty = Field(default_factory=dict)
    processor_config: DictOrEmpty = Field(default_factory=dict)

    @property
    def chat_template(self) -> str | None:
        for source in (self.tokenizer_config, self.processor_config):
            template = source.get("chat_template")
            if isinstance(template, str) and template:
                return template
        return None


class WidgetParameters(_Loose):
    negative_prompt: TextOrNone = None  # every card is parsed: anything but text is dropped


class CardWidget(_Loose):
    """One example on a model card (image models show their prompts and settings here)."""

    parameters: Annotated[WidgetParameters, BeforeValidator(_or(dict, dict))] = Field(
        default_factory=WidgetParameters
    )


class CardData(_Loose):
    license: str | None = None
    base_model: str | None = None
    # The Hub's own "a newer version of this model is available" (`new_version` in a
    # card's metadata: `mistralai/Mistral-7B-Instruct-v0.2` names v0.3)
    new_version: str | None = None
    widget: list[CardWidget] = Field(default_factory=list)

    @field_validator("widget", mode="before")
    @classmethod
    def _examples_only(cls, value: object) -> object:
        """Cards write one example as a mapping, several as a list; anything else is noise."""
        items = [value] if isinstance(value, dict) else value if isinstance(value, list) else []
        return [item for item in items if isinstance(item, dict)]

    @field_validator("license", "base_model", "new_version", mode="before")
    @classmethod
    def _first_string(cls, value: object) -> str | None:
        """Model cards write these as a string or a list of strings; keep the first."""
        if isinstance(value, list):
            value = value[0] if value else None
        return value if isinstance(value, str) and value else None


class HubModel(_Loose):
    id: str
    pipeline_tag: str | None = None
    library_name: str | None = None
    tags: ListOrEmpty = Field(default_factory=list)
    gated: bool | str = False
    config: ConfigMeta = Field(default_factory=ConfigMeta)
    gguf: GgufMeta = Field(default_factory=GgufMeta)
    safetensors: SafetensorsMeta = Field(default_factory=SafetensorsMeta)
    card_data: CardData = Field(default_factory=CardData, alias="cardData")

    @field_validator("config", "gguf", "safetensors", "card_data", mode="before")
    @classmethod
    def _none_is_absent(cls, value: object) -> object:
        return {} if value is None else value

    @field_validator("gated", mode="before")
    @classmethod
    def _gated_or_open(cls, value: object) -> object:
        return False if value is None else value

    @property
    def is_gated(self) -> bool:
        return bool(self.gated)

    @property
    def library_name_lower(self) -> str:
        return (self.library_name or "").lower()

    @property
    def chat_template(self) -> str | None:
        return self.gguf.chat_template or self.config.chat_template


class SearchHit(_Loose):
    id: str
    downloads: int = 0
    pipeline_tag: str | None = None


class LfsInfo(_Loose):
    size: int
    oid: str = ""  # the file's sha256 (the Civitai Archive's mirrors are checked against it)


class TreeEntry(_Loose):
    type: str
    path: str
    size: int = 0  # a directory entry may carry none
    lfs: LfsInfo | None = None

    @field_validator("lfs", mode="before")
    @classmethod
    def _lfs_needs_a_size(cls, value: object) -> object:
        return value if isinstance(value, dict) and "size" in value else None

    @property
    def is_file(self) -> bool:
        return self.type == "file"


class _Commit(_Loose):
    sha: str


class HubStatusError(HubError):
    """A file read answered with an HTTP error; `status` lets a source word it."""

    def __init__(self, status: int, url: str) -> None:
        super().__init__(f"HTTP {status} reading {url}")
        self.status = status


class HubFileSource:
    """Range reads of one repo file, for header parsing without downloading the model.

    `resolve` links redirect to a CDN that honours Range (206). If a server answers 200
    instead, the body is streamed and cut at `start + length`, never read whole.
    """

    def __init__(self, url: str, client: httpx.Client) -> None:
        self._url = url
        self._client = client

    def read_range(self, start: int, length: int) -> bytes:
        headers = {"Range": f"bytes={start}-{start + length - 1}"}
        try:
            with self._client.stream("GET", self._url, headers=headers) as r:  # client follows
                if r.status_code == httpx.codes.PARTIAL_CONTENT:
                    return r.read()
                if r.status_code != httpx.codes.OK:
                    raise HubStatusError(r.status_code, self._url)
                return _slice_stream(r.iter_bytes(), start, length)
        except httpx.HTTPError as exc:  # timeouts, resets: a reason, never a traceback
            raise HubError(f"Could not read {self._url}: {exc}") from exc


def _slice_stream(chunks: Iterable[bytes], start: int, length: int) -> bytes:
    out = bytearray()
    seen = 0
    for chunk in chunks:
        chunk_start = seen
        seen += len(chunk)
        if seen <= start:
            continue
        begin = max(start - chunk_start, 0)
        out.extend(chunk[begin : begin + (length - len(out))])
        if len(out) >= length:
            break
    return bytes(out)


class HubClient:
    """Thin wrapper over httpx. Pass `transport` in tests to avoid the network."""

    def __init__(
        self,
        token: str | None = None,
        base_url: str = HUB_URL,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 20.0,
    ) -> None:
        token = token if base_url == OFFICIAL_HUB else None  # never to a mirror
        self._has_token = bool(token)
        headers = {"User-Agent": USER_AGENT}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        # A renamed repo answers 307 to its new name with a plain-text body; following it
        # keeps old references (and kit entries) working instead of failing to parse.
        self._http = httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )

    def search(
        self, *, search: str | None = None, filters: tuple[str, ...] = (), limit: int = 5
    ) -> list[SearchHit]:
        """Most-downloaded repos matching a text search and/or Hub filters (e.g. `gguf`)."""
        params: list[tuple[str, str]] = [
            ("sort", "downloads"),
            ("direction", "-1"),
            ("limit", str(limit)),
            *[("filter", f) for f in filters],
        ]
        if search:
            params.append(("search", search))
        data = self._get("/api/models", "search", params=tuple(params))
        if not isinstance(data, list):
            raise HubError("Unexpected search response from the Hub.")
        return _parse_list(SearchHit, data, "search")

    def file_source(self, repo_id: str, revision: str, path: str) -> HubFileSource:
        url = f"/{quote(repo_id, safe='/')}/resolve/{quote(revision, safe='')}/{quote(path)}"
        return HubFileSource(str(self._http.base_url.join(url)), self._http)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> HubClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def model_info(self, repo_id: str) -> HubModel:
        params = tuple(("expand[]", name) for name in _EXPAND)
        data = self._get(f"/api/models/{quote(repo_id, safe='/')}", repo_id, params=params)
        return _parse(HubModel, data, repo_id)

    def json_file(self, repo_id: str, path: str, revision: str = "main") -> object:
        """A small JSON file the tree lists (a diffusers `model_index.json`), parsed. The
        caller types it; HubError when it cannot be read or is not JSON."""
        url = f"/{quote(repo_id, safe='/')}/resolve/{quote(revision, safe='')}/{quote(path)}"
        data: object = self._get(url, f"{repo_id}/{path}", params=())
        return data

    def tree(self, repo_id: str, revision: str = "main") -> list[TreeEntry]:
        """Every file: the Hub pages this listing (`Link: <…>; rel="next"`), and a big repo
        (a hundred voice packages, every quant of a model) runs past the first page."""
        url = f"/api/models/{quote(repo_id, safe='/')}/tree/{quote(revision, safe='')}"
        params: tuple[tuple[str, str], ...] | None = (("recursive", "true"),)
        entries: list[TreeEntry] = []
        for _ in range(MAX_TREE_PAGES):
            response = self._response(url, repo_id, params)
            data = self._json(response, repo_id)
            if not isinstance(data, list):
                raise HubError(f"Unexpected tree response for {repo_id}.")
            entries.extend(_parse_list(TreeEntry, data, repo_id))
            following = response.links.get("next", {}).get("url")
            if not following:
                return entries
            url, params = following, None  # the cursor and the query are in the link itself
        raise HubError(f"{repo_id} lists more than {MAX_TREE_PAGES} pages of files.")

    def commit(self, repo_id: str, revision: str = "main") -> str:
        """The commit a revision (a branch, a tag) points at now: a link pinned to it keeps
        naming the same files."""
        url = f"/api/models/{quote(repo_id, safe='/')}/revision/{quote(revision, safe='')}"
        found = _parse(_Commit, self._get(url, repo_id, params=(("expand[]", "sha"),)), repo_id)
        return found.sha

    def path_info(self, repo_id: str, revision: str, path: str) -> TreeEntry | None:
        """The Hub's record of one file (its size and sha256), without listing the repo: a
        mirror repo can hold a hundred thousand files. None when it is not there."""
        url = f"/api/models/{quote(repo_id, safe='/')}/paths-info/{quote(revision, safe='')}"
        response = self._response(url, repo_id, None, data={"paths": path})
        data = self._json(response, repo_id)
        if not isinstance(data, list):
            raise HubError(f"Unexpected paths-info response for {repo_id}.")
        found = _parse_list(TreeEntry, data, repo_id)
        return next((e for e in found if e.is_file and e.path == path), None)

    def _get(self, url: str, repo_id: str, params: tuple[tuple[str, str], ...]) -> Any:
        """Raises only HubError subclasses; the body is returned as parsed JSON."""
        return self._json(self._response(url, repo_id, params), repo_id)

    def _response(
        self,
        url: str,
        repo_id: str,
        params: tuple[tuple[str, str], ...] | None,
        data: dict[str, str] | None = None,  # a form: the request is a POST
    ) -> httpx.Response:
        try:
            if data is None:
                response = self._http.get(url, params=params)
            else:
                response = self._http.post(url, data=data)
        except httpx.HTTPError as exc:
            host = httpx.URL(url).host or self._http.base_url.host  # a mirror's
            raise HubError(f"Could not reach {host}: {exc}") from exc
        if response.status_code == httpx.codes.NOT_FOUND:
            raise NotFoundError(f"{repo_id} does not exist on the Hub.")
        if response.status_code == httpx.codes.UNAUTHORIZED and not self._has_token:
            # Without a token the Hub answers 401 for repos that do not exist at all.
            raise NotFoundError(
                f"{repo_id} does not exist on the Hub, or is private. Check the spelling; "
                "for a private repo set HF_TOKEN in ~/.config/makenomistakes/secrets.env."
            )
        if response.status_code in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
            raise GatedOrPrivateError(
                f"Your token cannot access {repo_id}. Accept the model's terms on "
                "huggingface.co or check the token's permissions."
            )
        if response.is_error:
            raise HubError(f"Hub returned HTTP {response.status_code} for {repo_id}.")
        return response

    def _json(self, response: httpx.Response, repo_id: str) -> Any:
        try:
            return response.json()
        except ValueError as exc:  # JSONDecodeError, and a body that is not UTF-8
            raise HubError(
                f"Hub answered HTTP {response.status_code} for {repo_id} with a body that is "
                "not JSON; try again, or check the repo on huggingface.co."
            ) from exc


def _parse[M: BaseModel](model: type[M], data: object, what: str) -> M:
    """Typed boundary: a payload the Hub changed shape on becomes a HubError, not a traceback."""
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise HubError(
            f"Unexpected response from the Hub for {what}: {exc.error_count()} field errors."
        ) from exc


def _parse_list[M: BaseModel](model: type[M], data: list[object], what: str) -> list[M]:
    return [_parse(model, item, what) for item in data]
