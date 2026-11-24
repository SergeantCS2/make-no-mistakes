"""Civitai as a second source: links, the typed API, and a version as Hub-shaped facts.

A Civitai *model* has versions; a version has files (usually one primary `.safetensors`).
The plan reads that file's header over HTTP like a Hub file, so the rest of `mnm` (family
from the header, checkpoint or LoRA, kits, recipes) works unchanged. Inside mnm a version is
the repo `civitai/<model id>` at revision `<version id>`; its files download as plain files
under `<models>/civitai/<version id>/`.

API (public, https://developer.civitai.com/docs/api/public-rest): `GET /api/v1/models/<id>`,
`GET /api/v1/model-versions/<id>`, `GET /api/download/models/<version id>` (a redirect to
the CDN, which honours Range). A `CIVITAI_TOKEN` (Authorization: Bearer) unlocks downloads
that need a login; httpx drops the header on the redirect to the CDN's own host.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from makenomistakes import __version__
from makenomistakes.hub.civitai_archive import HOSTS as ARCHIVE_HOSTS
from makenomistakes.hub.civitai_archive import ArchiveClient, Archived, from_the_archive
from makenomistakes.hub.client import (
    HubClient,
    HubError,
    HubFileSource,
    HubModel,
    HubStatusError,
    TreeEntry,
)
from makenomistakes.hub.elsewhere import elsewhere
from makenomistakes.hub.lmstudio import LmStudioClient
from makenomistakes.hub.modelscope import ModelScopeClient, is_modelscope
from makenomistakes.hub.ollama import OllamaClient, is_ollama
from makenomistakes.hub.refs import CIVITAI_DOMAINS, ModelRef, hosts

API_URL = "https://civitai.com"
# The links recognised as Civitai's; API calls always go to API_URL.
HOSTS = hosts(*CIVITAI_DOMAINS)
# The hosts a download may use from a file's own link, which the token goes with: the API's
# own (its answers name it in every `downloadUrl`), never every alias a link may name.
_TOKEN_HOSTS = frozenset({"civitai.com", "www.civitai.com"})
REPO_PREFIX = "civitai/"
SUBDIR = "civitai"  # under the models folder: <models>/civitai/<version id>/<file>
LORA_TYPES = frozenset({"LORA", "LoCon", "DoRA", "LORA (LyCORIS)"})
RUNNABLE_TYPES = frozenset({"Checkpoint"}) | LORA_TYPES
_SHORT_RE = re.compile(r"^civitai:(\d+)(?:@(\d+))?$", re.IGNORECASE)


class CivitaiError(HubError):
    """A Civitai answer this tool cannot use, worded for the user."""


class CivitaiRefusedError(CivitaiError):
    """Civitai's model, read, is one mnm does not run (not a checkpoint or LoRA, no
    safetensors file, an odd file name): a refusal, never a read that failed."""


class CivitaiGoneError(CivitaiError):
    """Civitai has no such model or version (HTTP 404): deleted, or a wrong link. The Civitai
    Archive may hold a copy (`SourcesHub.archived`)."""


@dataclass(frozen=True)
class CivitaiRef:
    model_id: int | None = None
    version_id: int | None = None


def parse_civitai(text: str) -> CivitaiRef | None:
    """`https://civitai.com/models/<id>[/slug][?modelVersionId=<v>]`, a download link
    `…/api/download/models/<v>`, or `civitai:<id>[@<v>]`; None for anything else."""
    raw = text.strip().strip("\"'")
    short = _SHORT_RE.match(raw)
    if short:
        return CivitaiRef(int(short.group(1)), int(short.group(2)) if short.group(2) else None)
    url = urlparse(raw if "://" in raw else f"https://{raw}")
    host = (url.hostname or "").lower()
    if host not in HOSTS and host not in ARCHIVE_HOSTS:  # the archive keeps Civitai's ids
        return None
    parts = [p for p in url.path.split("/") if p]
    version = parse_qs(url.query).get("modelVersionId", [None])[0]
    if parts[:3] == ["api", "download", "models"] and len(parts) > 3 and parts[3].isdigit():
        return CivitaiRef(None, int(parts[3]))
    if len(parts) >= 2 and parts[0] == "models" and parts[1].isdigit():
        return CivitaiRef(int(parts[1]), int(version) if version and version.isdigit() else None)
    if host in ARCHIVE_HOSTS:  # `/<platform>/models/…`: another site's model, mirrored
        raise CivitaiError(
            f"{raw} is not one of the Civitai Archive's Civitai models: paste a link of the "
            "form civitaiarchive.com/models/<id>?modelVersionId=<version>."
        )
    raise CivitaiError(
        f"{raw} is a Civitai page but not a model: paste a model's link "
        "(civitai.com/models/<id>, optionally with ?modelVersionId=<version>)."
    )


def is_civitai(repo_id: str) -> bool:
    return repo_id.startswith(REPO_PREFIX)


# The one place the `civitai/<model id>` @ `<version id>` naming is spelled out.
def civitai_repo(model_id: int) -> str:
    return f"{REPO_PREFIX}{model_id}"


def civitai_reference(repo_id: str, version: str) -> str:
    """What to paste to plan that version again (`mnm replan`)."""
    return f"civitai:{repo_id.removeprefix(REPO_PREFIX)}@{version}"


def civitai_page(repo_id: str, version: str) -> str:
    return f"{API_URL}/models/{repo_id.removeprefix(REPO_PREFIX)}?modelVersionId={version}"


class _Loose(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class FileMetadata(_Loose):
    format: str | None = None  # SafeTensor, PickleTensor, GGUF, Diffusers …
    fp: str | None = None
    size: str | None = None  # full, pruned


class CivitaiFile(_Loose):
    id: int
    name: str
    size_kb: float = Field(default=0, alias="sizeKB")
    type: str = "Model"
    primary: bool = False
    metadata: FileMetadata = Field(default_factory=FileMetadata)
    download_url: str = Field(default="", alias="downloadUrl")
    hashes: dict[str, str] = Field(default_factory=dict)  # SHA256, AutoV2, … (uppercase hex)

    @property
    def size_bytes(self) -> int:
        return int(self.size_kb * 1024)

    @property
    def safetensors(self) -> bool:
        return self.name.endswith(".safetensors") or self.metadata.format == "SafeTensor"

    @property
    def plain_name(self) -> bool:
        """A name that is one file: it becomes `<models>/civitai/<version>/<name>`."""
        name = self.name
        return bool(name) and name not in (".", "..") and not any(c in name for c in "/\\\0")

    def url(self, version_id: int) -> str:
        """Where this file downloads from: its own link (the query names the file among the
        version's), never a host outside Civitai's (the token goes with the request)."""
        link = urlparse(self.download_url)
        if link.scheme == "https" and (link.hostname or "").lower() in _TOKEN_HOSTS:
            return self.download_url
        return f"{API_URL}/api/download/models/{version_id}"


class ModelSummary(_Loose):
    name: str = ""
    type: str = ""
    nsfw: bool = False


class CivitaiVersion(_Loose):
    id: int
    model_id: int | None = Field(default=None, alias="modelId")
    name: str = ""
    base_model: str = Field(default="", alias="baseModel")
    files: list[CivitaiFile] = Field(default_factory=list)
    trained_words: list[str] = Field(default_factory=list, alias="trainedWords")
    model: ModelSummary | None = None

    def weights_file(self) -> CivitaiFile:
        """The file a plan runs: the primary safetensors model file, else the largest."""
        models = [f for f in self.files if f.type in {"Model", "Pruned Model"} and f.safetensors]
        odd = [f.name for f in models if not f.plain_name]
        if odd:
            raise CivitaiRefusedError(
                f"Civitai version {self.id} names a file that is not a plain file name "
                f"({odd[0]!r}); it is refused rather than written outside the models folder."
            )
        if not models:
            names = ", ".join(f.name for f in self.files) or "none"
            raise CivitaiRefusedError(
                f"Civitai version {self.id} has no safetensors model file (files: {names}). "
                "Pickle checkpoints (.ckpt, .pt) can run code when loaded; this tool reads "
                "safetensors only."
            )
        return next((f for f in models if f.primary), max(models, key=lambda f: f.size_bytes))


class CivitaiStats(_Loose):
    download_count: int = Field(default=0, alias="downloadCount")


class CivitaiModel(_Loose):
    id: int
    name: str
    type: str
    nsfw: bool = False
    versions: list[CivitaiVersion] = Field(default_factory=list, alias="modelVersions")
    stats: CivitaiStats = Field(default_factory=CivitaiStats)  # `mnm what`'s columns


class SearchPage(_Loose):
    items: list[CivitaiModel] = Field(default_factory=list)


@dataclass(frozen=True)
class Resolved:
    """One version of one Civitai model, with what the plan needs to know about it."""

    model_id: int
    name: str
    type: str
    version: CivitaiVersion

    @property
    def repo_id(self) -> str:
        return civitai_repo(self.model_id)

    @property
    def ref(self) -> ModelRef:
        return ModelRef(self.repo_id, revision=str(self.version.id))

    def as_hub(self) -> tuple[HubModel, list[TreeEntry]]:
        """The version as a Hub model with a one-file tree: the image classification (a
        checkpoint, a LoRA by its tag) and the header read take it from there."""
        info = as_hub_model(self.name, self.type, self.repo_id)
        weights = self.version.weights_file()
        tree = [TreeEntry(type="file", path=weights.name, size=weights.size_bytes)]
        return info, tree


def as_hub_model(name: str, kind: str, repo_id: str) -> HubModel:
    """A Civitai file of type `kind` described as a Hub model, for the image classification
    (a checkpoint, a LoRA by its tag): Civitai's own versions and the Civitai Archive's
    copies of deleted ones, whose mirror repos carry no metadata of their own."""
    if kind not in RUNNABLE_TYPES:
        raise CivitaiRefusedError(
            f"{name} is a Civitai {kind}, not a checkpoint or LoRA: this tool runs "
            "checkpoints (and LoRAs on an image instance, `mnm lora add`)."
        )
    tags = ["safetensors", "civitai", kind.lower(), *(["lora"] if kind in LORA_TYPES else [])]
    return HubModel(id=repo_id, pipeline_tag="text-to-image", tags=tags)


class CivitaiClient:
    """Thin typed wrapper over Civitai's API. Pass `transport` in tests."""

    def __init__(
        self,
        token: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 20.0,
    ) -> None:
        headers = {"User-Agent": f"makenomistakes/{__version__}"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._has_token = bool(token)
        self._versions: dict[str, CivitaiVersion] = {}
        self.http = httpx.Client(
            base_url=API_URL,
            headers=headers,
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )

    def close(self) -> None:
        self.http.close()

    def resolve(self, ref: CivitaiRef) -> Resolved:
        """A version (the newest when the link names none) with its model's name and type."""
        if ref.version_id is not None:
            version = self._get(f"/api/v1/model-versions/{ref.version_id}", CivitaiVersion)
            summary = version.model or ModelSummary()
            model_id = version.model_id or ref.model_id
            if model_id is None:
                raise CivitaiError(f"Civitai version {version.id} names no model.")
            return Resolved(model_id, summary.name or f"civitai-{model_id}", summary.type, version)
        if ref.model_id is None:
            raise CivitaiError("A Civitai link names a model or a version; this one names none.")
        model = self._get(f"/api/v1/models/{ref.model_id}", CivitaiModel)
        if not model.versions:
            raise CivitaiError(f"Civitai model {model.id} has no versions.")
        return Resolved(model.id, model.name, model.type, model.versions[0])

    def file(self, version_id: str, filename: str) -> CivitaiFile:
        """One named file of a version, as the API lists it (its link and its size)."""
        version = self._versions.get(version_id)
        if version is None:
            version = self._get(f"/api/v1/model-versions/{version_id}", CivitaiVersion)
            self._versions[version_id] = version
        found = next((f for f in version.files if f.name == filename), None)
        if found is None or not found.plain_name:
            raise CivitaiError(f"Civitai version {version_id} has no file named {filename!r}.")
        return found

    def file_url(self, version_id: str, filename: str) -> str:
        """The download link of one named file of a version (the bare version link serves
        its primary file, which may be another one: a pickle beside the safetensors)."""
        return self.file(version_id, filename).url(int(version_id))

    def file_source(self, version_id: str, filename: str) -> HubFileSource:
        """Range reads of one of a version's files (its header), never the whole file."""
        return _CivitaiFile(self.file_url(version_id, filename), self.http, self._has_token)

    def by_hash(self, sha256: str) -> CivitaiVersion | None:
        """The version holding a file of this sha256; None when Civitai knows none."""
        try:
            return self._get(f"/api/v1/model-versions/by-hash/{sha256}", CivitaiVersion)
        except CivitaiGoneError:
            return None

    def search(
        self,
        query: str,
        limit: int = 20,
        types: tuple[str, ...] = (),
        sfw: bool = False,
        sort: str | None = None,
    ) -> list[CivitaiModel]:
        """Civitai's model search (by model name, not file name): one page. `types`
        (`Checkpoint`, `LORA`), `sfw` (Civitai's `nsfw=false`) and `sort` (its own words:
        `Most Downloaded`, `Highest Rated`)."""
        params: list[tuple[str, str | int]] = [("query", query), ("limit", limit)]
        params += [("types", t) for t in types]
        if sfw:
            params.append(("nsfw", "false"))
        if sort is not None:
            params.append(("sort", sort))
        return self._get(f"/api/v1/models?{urlencode(params)}", SearchPage).items

    def _get[T: BaseModel](self, path: str, model: type[T]) -> T:
        try:
            response = self.http.get(path)
        except httpx.HTTPError as exc:
            raise CivitaiError(f"Could not reach Civitai: {exc}") from exc
        if response.status_code in (401, 403):
            raise CivitaiError(refused(response.status_code, self._has_token, path))
        if response.status_code == 404:
            raise CivitaiGoneError(
                f"Civitai has nothing at {path} (HTTP 404): deleted, or a wrong link."
            )
        if response.is_error:
            raise CivitaiError(f"Civitai answered HTTP {response.status_code} for {path}.")
        try:
            return model.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise CivitaiError(f"Civitai's answer for {path} was not understood: {exc}") from exc


class SourcesHub(HubClient):
    """The Hugging Face client that also reads Civitai versions: a `civitai/<id>` repo's
    file is read from Civitai, everything else (kit companions included) from the Hub."""

    def __init__(  # noqa: PLR0913 - each client can be swapped
        self,
        token: str | None = None,
        civitai_token: str | None = None,
        *,
        civitai: CivitaiClient | None = None,
        archive: ArchiveClient | None = None,
        lmstudio: LmStudioClient | None = None,
        transport: httpx.BaseTransport | None = None,  # the Hub's, in tests
        ollama: OllamaClient | None = None,
        modelscope: ModelScopeClient | None = None,
    ) -> None:
        super().__init__(token=token, transport=transport)
        self.civitai = civitai or CivitaiClient(token=civitai_token)
        self.archive = archive or ArchiveClient()
        self.lmstudio = lmstudio or LmStudioClient()
        self.ollama = ollama or OllamaClient()
        self.modelscope = modelscope or ModelScopeClient()

    def archived(self, ref: CivitaiRef, gone: CivitaiGoneError) -> Archived:
        """The Hub copy of a version Civitai no longer has, from the Civitai Archive;
        `ArchiveError` (a `HubError`) says why there is none."""
        return from_the_archive(self, self.archive, ref.model_id, ref.version_id, str(gone))

    def file_source(self, repo_id: str, revision: str, path: str) -> HubFileSource:
        if is_civitai(repo_id):
            return self.civitai.file_source(revision, path)
        if is_ollama(repo_id):
            return self.ollama.file_source(repo_id, revision)
        if is_modelscope(repo_id):
            return self.modelscope.file_source(repo_id, revision, path)
        return super().file_source(repo_id, revision, path)

    def model_info(self, repo_id: str) -> HubModel:
        _not_on_the_hub(repo_id)
        return super().model_info(repo_id)

    def tree(self, repo_id: str, revision: str = "main") -> list[TreeEntry]:
        _not_on_the_hub(repo_id)
        return super().tree(repo_id, revision)

    def json_file(self, repo_id: str, path: str, revision: str = "main") -> object:
        _not_on_the_hub(repo_id)
        return super().json_file(repo_id, path, revision)

    def close(self) -> None:
        self.civitai.close()
        self.archive.close()
        self.lmstudio.close()
        self.ollama.close()
        self.modelscope.close()
        super().close()

    def __enter__(self) -> SourcesHub:
        return self


def _not_on_the_hub(repo_id: str) -> None:
    """A Civitai version (or an Ollama or ModelScope repo) is read through its own client,
    never asked of Hugging Face."""
    if is_civitai(repo_id):
        raise CivitaiError(f"{repo_id} is a Civitai model: it has no Hugging Face page.")
    if (where := elsewhere(repo_id)) is not None:
        raise where.error(f"{repo_id} is a {where.name} model: it has no Hugging Face page.")


SECRETS_HINT = (
    "Create an API key on civitai.com (Account settings, API keys) and add the line "
    "CIVITAI_TOKEN=<key> to ~/.config/makenomistakes/secrets.env; it is read again at the "
    "next request (each request builds its Civitai client afresh), no restart needed."
)


def refused(status: int, token_sent: bool, what: str) -> str:
    """Civitai's 401/403 for `what` (the API path or file read), worded by whether a key went
    with the request."""
    if not token_sent:
        return f"Civitai wants a login for {what} (HTTP {status}). {SECRETS_HINT}"
    return (
        f"Civitai refused {what} with your CIVITAI_TOKEN (HTTP {status}): it may be early "
        "access or need its terms accepted on civitai.com while logged in, or the key may be "
        "wrong or revoked (Account settings, API keys)."
    )


class _CivitaiFile(HubFileSource):
    """A Civitai file's range reads, its refusals worded for the user."""

    def __init__(self, url: str, client: httpx.Client, token_sent: bool) -> None:
        super().__init__(url, client)
        self._url = url
        self._token_sent = token_sent

    def read_range(self, start: int, length: int) -> bytes:
        try:
            return super().read_range(start, length)
        except HubStatusError as exc:
            if exc.status in (401, 403):
                raise CivitaiError(refused(exc.status, self._token_sent, self._url)) from exc
            raise
