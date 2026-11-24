"""Ollama's library as a source: `ollama.com/library/qwen3:8b` (or a user's
`ollama.com/<user>/<name>:<tag>`) is one GGUF in Ollama's registry. The manifest names the
model layer (a GGUF blob, read by range for the header like a Hub file) and a config blob
(format, family, quant); the blob downloads as a plain file under `<models>/ollama/`, checked
against its sha256 digest. A tag is `ollama/<namespace>--<name>` at revision `<tag>` to the
rest of mnm, as a Civitai version is `civitai/<id>`.

Refused, with the reason: a model that is not GGUF, and the ones with a projector or adapter
layer (Ollama's own runner reads those; llama.cpp would load the text half and nothing else).
A plain `qwen3:8b` keeps its own path (a GGUF search on the Hub, `guide.guide_ollama_tag`)."""

from __future__ import annotations

import re
from contextlib import AbstractContextManager
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from makenomistakes import __version__
from makenomistakes.hub.client import HubError, HubFileSource, HubModel, TreeEntry
from makenomistakes.hub.refs import ModelRef

OLLAMA_DOMAIN = "ollama.com"
REGISTRY = "https://registry.ollama.ai"
PREFIX = "ollama/"
SUBDIR = "ollama"  # <models>/ollama/<namespace>--<name>/<tag>/<file>.gguf
MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"
MODEL_LAYER = "application/vnd.ollama.image.model"
REFUSED_LAYERS = {
    "application/vnd.ollama.image.projector": "a vision projector",
    "application/vnd.ollama.image.adapter": "an adapter (LoRA)",
}
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_LIBRARY_LINK = re.compile(r'href="/library/([^"/?#:]+)"')  # Its search page's models
DOWNLOAD_TIMEOUT = httpx.Timeout(60.0, connect=30.0)


class OllamaError(HubError):
    """An Ollama model mnm cannot plan, worded for the user."""


class OllamaRefusedError(OllamaError):
    """Ollama's model, read, is one mnm does not run (a projector or adapter layer, not one
    GGUF): a refusal of the model, never a read that failed (`mnm what`'s *refused*)."""


@dataclass(frozen=True)
class OllamaRef:
    namespace: str  # `library` for Ollama's own models
    name: str
    tag: str = "latest"

    @property
    def repo_id(self) -> str:
        return f"{PREFIX}{self.namespace}--{self.name}"

    @property
    def page(self) -> str:
        return f"https://{OLLAMA_DOMAIN}/{self.namespace}/{self.name}:{self.tag}"


def is_ollama(repo_id: str) -> bool:
    return repo_id.startswith(PREFIX)


def ref_of(repo_id: str, tag: str) -> OllamaRef:
    """The reference a recipe's repo id and revision stand for."""
    namespace, _, name = repo_id.removeprefix(PREFIX).partition("--")
    return OllamaRef(namespace, name, tag)


def parse_ollama(text: str) -> OllamaRef | None:
    """An ollama.com model link, or None when the text is not one (a Hub link, a name)."""
    raw = text.strip().strip("\"'")
    if "://" not in raw:
        raw = f"https://{raw}"
    parsed = urlparse(raw)
    if (parsed.hostname or "").lower() not in (OLLAMA_DOMAIN, f"www.{OLLAMA_DOMAIN}"):
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 3 and parts[2] in ("tags", "blobs"):
        parts = parts[:2]  # the tags page of a model
    if len(parts) != 2:
        raise OllamaError(
            f"{text.strip()} is not a model page: use ollama.com/library/<name>:<tag> "
            "(or ollama.com/<user>/<name>:<tag>)."
        )
    namespace, named = parts
    name, _, tag = named.partition(":")
    tag = tag or "latest"
    # `--` joins the namespace to the name in a repo id: one inside a namespace would split
    # it in the wrong place (`ref_of`)
    if "--" in namespace or not (_NAME.match(namespace) and _NAME.match(name) and _TAG.match(tag)):
        raise OllamaError(f"{text.strip()} does not name an Ollama model and tag.")
    return OllamaRef(namespace, name, tag)


class Layer(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    media_type: str = Field(alias="mediaType")
    digest: str
    size: int = 0


class Manifest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    config: Layer
    layers: list[Layer]


class OllamaConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_format: str = ""
    model_family: str = ""
    file_type: str = ""


@dataclass(frozen=True)
class Resolved:
    ref: OllamaRef
    model: Layer
    config: OllamaConfig

    @property
    def filename(self) -> str:
        """The GGUF's name in mnm: the tag, with the quant when the tag does not say it."""
        quant = self.config.file_type
        tagged = quant and quant.lower() in self.ref.tag.lower()
        stem = f"{self.ref.name}-{self.ref.tag}" + ("" if tagged or not quant else f"-{quant}")
        return f"{stem}.gguf"

    def as_hub(self) -> tuple[HubModel, list[TreeEntry], ModelRef]:
        family = self.config.model_family.lower()
        embedding = "bert" in family or "embed" in self.ref.name
        info = HubModel(
            id=self.ref.repo_id,
            pipeline_tag="feature-extraction" if embedding else "text-generation",
            tags=["gguf", "ollama", *([family] if family else [])],
        )
        tree = [TreeEntry(type="file", path=self.filename, size=self.model.size)]
        return info, tree, ModelRef(self.ref.repo_id, revision=self.ref.tag)


class OllamaClient:
    """Ollama's public registry: read only, no account."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": f"makenomistakes/{__version__}"},
        )
        self._resolved: dict[tuple[str, str], Resolved] = {}

    def close(self) -> None:
        self._client.close()

    def stream(self, url: str, headers: dict[str, str]) -> AbstractContextManager[httpx.Response]:
        """A blob's download on this client's connection (swappable, as for reads)."""
        return self._client.stream("GET", url, headers=headers, timeout=DOWNLOAD_TIMEOUT)

    def _path(self, ref: OllamaRef) -> str:
        return f"{REGISTRY}/v2/{ref.namespace}/{ref.name}"

    def blob_url(self, ref: OllamaRef, digest: str) -> str:
        if not _DIGEST.match(digest):
            raise OllamaError(f"{ref.page}: the registry named a blob mnm cannot read ({digest}).")
        return f"{self._path(ref)}/blobs/{digest}"

    def _get(self, url: str, ref: OllamaRef, accept: str = "") -> bytes:
        try:
            response = self._client.get(url, headers={"Accept": accept} if accept else None)
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama's registry did not answer: {exc}") from exc
        if response.status_code == httpx.codes.NOT_FOUND:
            raise OllamaError(f"Ollama has no {ref.page} (check the name and the tag).")
        if response.is_error:
            raise OllamaError(f"Ollama's registry answered HTTP {response.status_code}.")
        return response.content

    def search(self, words: str, limit: int = 5, category: str | None = None) -> list[str]:
        """The library's models matching `words`, in the order Ollama's search page
        lists them, of its `category` when given (`tools`, `vision`, `embedding`: the page's
        own `c=`). The page is HTML (ollama.com has no search API: `/api/tags` lists its own
        cloud models): the names come from its `/library/<name>` links, each checked as a
        model's name is (`_NAME`), so a page that changed shape gives none rather than wrong
        ones."""
        params = {"q": words} | ({"c": category} if category else {})
        try:
            response = self._client.get(f"https://{OLLAMA_DOMAIN}/search", params=params)
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama's library search did not answer: {exc}") from exc
        if response.is_error:
            raise OllamaError(f"Ollama's library search answered HTTP {response.status_code}.")
        names: list[str] = []
        for name in _LIBRARY_LINK.findall(response.text):
            if _NAME.fullmatch(name) and name not in names:
                names.append(name)
        return names[:limit]

    def resolve(self, ref: OllamaRef) -> Resolved:
        """The tag's GGUF and what its config says, refused in words when it is not one."""
        key = (ref.repo_id, ref.tag)
        if key in self._resolved:
            return self._resolved[key]
        try:
            manifest = Manifest.model_validate_json(
                self._get(f"{self._path(ref)}/manifests/{ref.tag}", ref, MANIFEST)
            )
        except ValidationError as exc:
            raise OllamaError(f"{ref.page}: Ollama's answer was not a model manifest.") from exc
        refused = [
            REFUSED_LAYERS[x.media_type] for x in manifest.layers if x.media_type in REFUSED_LAYERS
        ]
        if refused:  # said from the manifest alone: nothing else is fetched
            raise OllamaRefusedError(
                f"{ref.page} comes with {' and '.join(refused)}, which Ollama's own runner "
                "reads and llama.cpp does not: take a GGUF of it from Hugging Face instead."
            )
        try:
            config = OllamaConfig.model_validate_json(
                self._get(self.blob_url(ref, manifest.config.digest), ref)
            )
        except ValidationError as exc:
            raise OllamaError(f"{ref.page}: Ollama's model config could not be read.") from exc
        models = [x for x in manifest.layers if x.media_type == MODEL_LAYER]
        if len(models) != 1 or (config.model_format and config.model_format != "gguf"):
            raise OllamaRefusedError(f"{ref.page} is not one GGUF model file mnm can run.")
        found = Resolved(ref, models[0], config)
        self._resolved[key] = found
        return found

    def file_source(self, repo_id: str, tag: str) -> HubFileSource:
        """Range reads of the tag's GGUF (the registry redirects to a CDN that honours them)."""
        ref = ref_of(repo_id, tag)
        return HubFileSource(self.blob_url(ref, self.resolve(ref).model.digest), self._client)
