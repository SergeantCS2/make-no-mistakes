"""ModelScope as a source: `modelscope.cn/models/<org>/<repo>` (a file link
narrows it to that file) is read through ModelScope's own API, which answers like the Hub: the
repo's files with their sizes and sha256, the model's task and tags. That is Hub-shaped facts
(`modelscope/<org>--<repo>` at revision `master`), so the planner reads a GGUF repo's quants
and its header (by range) as it does a Hugging Face one. GGUF only: a safetensors repo is
pointed at its Hugging Face copy. Files download as plain files under `<models>/modelscope/`,
each kept only when its sha256 is the listed one."""

from __future__ import annotations

import re
from contextlib import AbstractContextManager
from dataclasses import dataclass
from urllib.parse import quote, urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from makenomistakes import __version__
from makenomistakes.hub.client import HubError, HubFileSource, HubModel, TreeEntry
from makenomistakes.hub.refs import ModelRef

MODELSCOPE_DOMAIN = "modelscope.cn"
API = f"https://{MODELSCOPE_DOMAIN}/api/v1/models"
PREFIX = "modelscope/"
SUBDIR = "modelscope"  # <models>/modelscope/<org>--<repo>/<revision>/<path>
_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_SECTIONS = ("summary", "files", "feedback", "resolve", "file")
DOWNLOAD_TIMEOUT = httpx.Timeout(60.0, connect=30.0)


class ModelScopeError(HubError):
    """A ModelScope model mnm cannot plan, worded for the user."""


@dataclass(frozen=True)
class ModelScopeRef:
    org: str
    repo: str
    revision: str = "master"
    path: str | None = None  # a file the link named: that quant

    @property
    def repo_id(self) -> str:
        return f"{PREFIX}{self.org}--{self.repo}"

    @property
    def page(self) -> str:
        return f"https://{MODELSCOPE_DOMAIN}/models/{self.org}/{self.repo}"


def is_modelscope(repo_id: str) -> bool:
    return repo_id.startswith(PREFIX)


def ref_of(repo_id: str, revision: str = "master") -> ModelScopeRef:
    org, _, repo = repo_id.removeprefix(PREFIX).partition("--")
    return ModelScopeRef(org, repo, revision)


def parse_modelscope(text: str) -> ModelScopeRef | None:
    """A modelscope.cn model link, or None when the text is not one."""
    raw = text.strip().strip("\"'")
    if "://" not in raw:
        raw = f"https://{raw}"
    parsed = urlparse(raw)
    if (parsed.hostname or "").lower() not in (MODELSCOPE_DOMAIN, f"www.{MODELSCOPE_DOMAIN}"):
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 3 or parts[0] != "models":
        raise ModelScopeError(
            f"{text.strip()} is not a model page: use modelscope.cn/models/<org>/<model>."
        )
    org, repo, rest = parts[1], parts[2], parts[3:]
    malformed = "--" in org or not (_PART.match(org) and _PART.match(repo))  # `--` joins them
    if malformed or (rest and rest[0] not in _SECTIONS):
        raise ModelScopeError(f"{text.strip()} does not name a ModelScope model.")
    path = None
    if rest[:1] in (["resolve"], ["file"]) and len(rest) > 2:  # resolve/<rev>/<file>
        path = "/".join(rest[2:])
    return ModelScopeRef(org, repo, path=path)


class _Loose(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class MsFile(_Loose):
    path: str = Field(alias="Path")
    type: str = Field(default="blob", alias="Type")
    size: int = Field(default=0, alias="Size")
    sha256: str = Field(default="", alias="Sha256")


class _Files(_Loose):
    files: list[MsFile] = Field(default_factory=list, alias="Files")


class _Task(_Loose):
    name: str = Field(default="", alias="Name")


class _Model(_Loose):
    tasks: list[_Task] | None = Field(default=None, alias="Tasks")
    tags: list[str] | None = Field(default=None, alias="Tags")
    libraries: list[str] | None = Field(default=None, alias="Libraries")


class _Answer(_Loose):
    code: int = Field(default=0, alias="Code")
    message: str = Field(default="", alias="Message")
    data: dict[str, object] | None = Field(default=None, alias="Data")


@dataclass(frozen=True)
class Resolved:
    ref: ModelScopeRef
    files: list[MsFile]
    model: _Model

    def as_hub(self) -> tuple[HubModel, list[TreeEntry], ModelRef]:
        tags = [*(self.model.tags or []), *(self.model.libraries or [])]
        task = next((t.name for t in self.model.tasks or [] if t.name), None)
        info = HubModel(id=self.ref.repo_id, pipeline_tag=task, tags=["modelscope", *tags])
        tree = [TreeEntry(type="file", path=f.path, size=f.size) for f in self.files]
        return (
            info,
            tree,
            ModelRef(self.ref.repo_id, revision=self.ref.revision, path=self.ref.path),
        )

    def file(self, path: str) -> MsFile:
        found = next((f for f in self.files if f.path == path), None)
        if found is None:
            raise ModelScopeError(f"{self.ref.page} has no file {path}.")
        return found


class ModelScopeClient:
    """ModelScope's public API: read only, no account."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": f"makenomistakes/{__version__}"},
        )
        self._resolved: dict[tuple[str, str], Resolved] = {}

    def close(self) -> None:
        self._client.close()

    def _data(self, url: str, ref: ModelScopeRef) -> dict[str, object]:
        try:
            response = self._client.get(url)
        except httpx.HTTPError as exc:
            raise ModelScopeError(f"ModelScope did not answer: {exc}") from exc
        try:
            answer = _Answer.model_validate_json(response.content)
        except ValidationError as exc:
            raise ModelScopeError(f"ModelScope answered HTTP {response.status_code}.") from exc
        if response.status_code == httpx.codes.NOT_FOUND or answer.data is None:
            raise ModelScopeError(
                f"ModelScope has no {ref.page} ({answer.message or 'not found'})."
            )
        return answer.data

    def resolve(self, ref: ModelScopeRef) -> Resolved:
        key = (ref.repo_id, ref.revision)
        if key in self._resolved:
            found = self._resolved[key]
            return Resolved(ref, found.files, found.model)
        base = f"{API}/{ref.org}/{ref.repo}"
        try:
            model = _Model.model_validate(self._data(base, ref))
            listed = _Files.model_validate(
                self._data(f"{base}/repo/files?Recursive=true&Revision={quote(ref.revision)}", ref)
            )
        except ValidationError as exc:
            raise ModelScopeError(f"{ref.page}: ModelScope's answer could not be read.") from exc
        files = [f for f in listed.files if f.type == "blob"]
        if not any(f.path.lower().endswith(".gguf") for f in files):
            raise ModelScopeError(
                f"{ref.page} has no GGUF file: mnm takes GGUF models from ModelScope (a "
                "safetensors model: paste its Hugging Face repo)."
            )
        found = Resolved(ref, files, model)
        self._resolved[key] = found
        return found

    def file_url(self, ref: ModelScopeRef, path: str) -> str:
        """The API's file route: it answers ranged reads itself (the page's redirects to a CDN
        address that changes)."""
        return (
            f"{API}/{ref.org}/{ref.repo}/repo?Revision={quote(ref.revision)}&FilePath={quote(path)}"
        )

    def file_source(self, repo_id: str, revision: str, path: str) -> HubFileSource:
        return HubFileSource(self.file_url(ref_of(repo_id, revision), path), self._client)

    def stream(self, url: str, headers: dict[str, str]) -> AbstractContextManager[httpx.Response]:
        return self._client.stream("GET", url, headers=headers, timeout=DOWNLOAD_TIMEOUT)
