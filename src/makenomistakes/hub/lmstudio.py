"""LM Studio's model catalog as a source: `lmstudio.ai/models/<owner>/<name>` is a
`model.yaml` (an open format, modelyaml.org) whose concrete models are Hugging Face repos,
GGUF and MLX builds side by side. A link resolves to the first GGUF repo it names, and from
there the plan is an ordinary Hugging Face one; MLX builds run only on Apple silicon.

The catalog has no public JSON API: the model page carries the `model.yaml` text in its
server-rendered payload (Next.js `self.__next_f.push([1, "…"])` strings, the file as one
`<id>:T<hex length>,<text>` row), which is read here and nowhere else. A family page
(`lmstudio.ai/models/<family>`) lists models, which are offered to pick from.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
import yaml
from pydantic import BaseModel, ValidationError

from makenomistakes.hub.client import HubError
from makenomistakes.hub.refs import LMSTUDIO_DOMAIN, hosts

SITE = f"https://{LMSTUDIO_DOMAIN}"
HOSTS = hosts(LMSTUDIO_DOMAIN)
TIMEOUT_SECONDS = 20.0
MAX_BASE_DEPTH = 3  # a model.yaml whose `base` names another model.yaml, followed this deep
MAX_PAGE_BYTES = 8 * 1024 * 1024  # a model page is ~0.3 MB
# The site's own sections: `lmstudio.ai/<owner>/<name>` is a model (as `lms get` reads it)
# only when its first part is none of these.
_SECTIONS = frozenset({"models", "docs", "blog", "download", "app", "careers", "api"})
_PUSH_RE = re.compile(r'self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)')
_TEXT_ROW_RE = re.compile(rb"(?:^|\n)[0-9a-f]+:T([0-9a-f]+),")
_MODEL_PATH = r"[A-Za-z0-9_-][A-Za-z0-9_.-]*/[A-Za-z0-9_-][A-Za-z0-9_.-]*"
_MODEL_LINK_RE = re.compile(rf'href="/models/({_MODEL_PATH})"')
_MODEL_PATH_RE = re.compile(_MODEL_PATH)


class LmStudioError(HubError):
    """An LM Studio page this tool cannot use, worded for the user."""


@dataclass(frozen=True)
class LmStudioRef:
    """`owner/name` for a model, or a family's name alone."""

    path: str

    @property
    def is_family(self) -> bool:
        return "/" not in self.path

    @property
    def link(self) -> str:
        return f"lmstudio.ai/models/{self.path}"


def parse_lmstudio(text: str) -> LmStudioRef | None:
    """`https://lmstudio.ai/models/<owner>/<name>`, `lmstudio.ai/<owner>/<name>` (as `lms
    get` takes it) or a family page `lmstudio.ai/models/<family>`; None for anything else."""
    raw = text.strip().strip("\"'")
    url = urlparse(raw if "://" in raw else f"https://{raw}")
    if (url.hostname or "").lower() not in HOSTS:
        return None
    parts = [p for p in url.path.split("/") if p]
    if parts[:1] == ["models"] and len(parts) in (2, 3):
        return LmStudioRef("/".join(parts[1:]))
    if len(parts) == 2 and parts[0].lower() not in _SECTIONS:
        return LmStudioRef("/".join(parts))
    raise LmStudioError(
        f"{raw} is an LM Studio page but not a model: paste a model's link "
        "(lmstudio.ai/models/<owner>/<name>)."
    )


class Source(BaseModel):
    type: str
    user: str = ""
    repo: str = ""

    @property
    def repo_id(self) -> str | None:
        """The Hugging Face repo, for a Hugging Face source with both parts."""
        return f"{self.user}/{self.repo}" if self.type == "huggingface" and self.user else None


class ConcreteModel(BaseModel):
    key: str
    sources: list[Source] = []

    @property
    def gguf(self) -> bool:
        repos = [s.repo for s in self.sources]
        return self.key.lower().endswith("gguf") or any("gguf" in r.lower() for r in repos)


class ModelYaml(BaseModel):
    """The parts of a `model.yaml` this tool reads: its name and its builds (`base`), or the
    other model it builds on (a string)."""

    model: str
    base: list[ConcreteModel] | str


@dataclass(frozen=True)
class LmStudioModel:
    name: str  # `owner/name`, as LM Studio calls it
    gguf_repos: tuple[str, ...]  # Hugging Face repos, LM Studio's order
    other_builds: tuple[str, ...]  # the rest (MLX): named when nothing here runs


@dataclass(frozen=True)
class LmStudioFamily:
    name: str
    models: tuple[str, ...]  # `owner/name` of each model the page lists


def model_yaml(page: str) -> ModelYaml | None:
    """The `model.yaml` a model page carries in its payload, or None (a family page, or a
    page whose shape changed)."""
    chunks = []
    for literal in _PUSH_RE.findall(page):
        try:
            chunks.append(json.loads(literal))
        except ValueError:
            continue
    # The row's length counts UTF-8 bytes; a chunk boundary may split a surrogate pair.
    payload = "".join(c for c in chunks if isinstance(c, str)).encode("utf-8", "surrogatepass")
    for row in _TEXT_ROW_RE.finditer(payload):
        size = int(row.group(1), 16)
        if row.end() + size > len(payload):  # cut short: a part of the file reads as valid
            continue
        text = payload[row.end() : row.end() + size].decode(errors="replace")
        if "model:" not in text or "base:" not in text:
            continue
        try:
            return ModelYaml.model_validate(yaml.safe_load(text))
        except (yaml.YAMLError, ValidationError):
            continue
    return None


def family_models(page: str) -> tuple[str, ...]:
    """The models a family page links to, in its order, once each."""
    return tuple(dict.fromkeys(_MODEL_LINK_RE.findall(page)))


class LmStudioClient:
    """Reads LM Studio's model pages (no token: the catalog is public)."""

    def __init__(self, transport: httpx.BaseTransport | None = None) -> None:
        self.http = httpx.Client(
            base_url=SITE, transport=transport, timeout=TIMEOUT_SECONDS, follow_redirects=True
        )

    def resolve(self, ref: LmStudioRef) -> LmStudioModel | LmStudioFamily:
        page = self._page(ref.path)
        if ref.is_family:
            return LmStudioFamily(ref.path, family_models(page))
        return self._model(ref.path, page, MAX_BASE_DEPTH)

    def _model(self, path: str, page: str, depth: int) -> LmStudioModel:
        found = model_yaml(page)
        if found is None:
            raise LmStudioError(
                f"lmstudio.ai/models/{path} carries no model.yaml this tool can read: paste "
                "the Hugging Face repo its page names under *Based on*."
            )
        if isinstance(found.base, str):  # built on another model: its builds are the ones
            if not _MODEL_PATH_RE.fullmatch(found.base):
                raise LmStudioError(f"{found.model} builds on {found.base!r}, not a model's name.")
            if depth <= 1:
                raise LmStudioError(f"{found.model} builds on models too deep to follow.")
            return self._model(found.base, self._page(found.base), depth - 1)
        gguf = [s.repo_id for c in found.base if c.gguf for s in c.sources]
        other = [s.repo_id for c in found.base if not c.gguf for s in c.sources]
        return LmStudioModel(
            found.model,
            tuple(r for r in dict.fromkeys(gguf) if r),
            tuple(r for r in dict.fromkeys(other) if r),
        )

    def _page(self, path: str) -> str:
        try:
            response = self.http.get(f"/models/{path}")
        except httpx.HTTPError as exc:
            raise LmStudioError(f"Could not reach lmstudio.ai: {exc}") from exc
        if (response.url.host or "").lower() not in HOSTS:
            raise LmStudioError(f"lmstudio.ai redirected off the site ({response.url.host}).")
        if response.status_code == 404:
            raise LmStudioError(f"LM Studio has no model at lmstudio.ai/models/{path}.")
        if response.is_error:
            raise LmStudioError(f"lmstudio.ai answered HTTP {response.status_code} for {path}.")
        if len(response.content) > MAX_PAGE_BYTES:
            raise LmStudioError(f"lmstudio.ai/models/{path} is too large to be a model page.")
        return response.text

    def close(self) -> None:
        self.http.close()
