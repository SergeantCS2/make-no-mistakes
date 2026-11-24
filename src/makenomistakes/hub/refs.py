"""Turn whatever the user pasted into a precise reference.

Accepted forms (all case-preserving for the repo id):

    https://huggingface.co/org/name          (also http://, www., hf.co, and no scheme at all)
    https://huggingface.co/org/name/tree/main/subdir
    https://huggingface.co/org/name/blob/main/file.gguf
    https://huggingface.co/org/name/resolve/v1.0/file.gguf?download=true
    hf.co/org/name:Q4_K_M                    (the form shown by "Use this model" on the Hub)
    hf://org/name                            (and huggingface://)
    hf download org/name --include "*Q4*"    (a pasted CLI line; only the repo is kept)
    org/name
    org/name:Q4_K_M
    org/name/file.safetensors                (a file in the repo: one version of many)
    org/name/sub/folder                      (a folder in the repo: a voice package)
    /data/models/foo.gguf  ~/x  ./x          (local paths)

Rejected with a clear message: Ollama tags such as `qwen2.5:14b`,
dataset/space URLs, and anything that is not a two-segment repo id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import unquote, urlsplit

# hf-mirror.com: a mirror of the Hub with the same repo ids (set HF_ENDPOINT to download
# through it too, `hub.client.hub_endpoint`)
_HUB_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co", "hf-mirror.com", "www.hf-mirror.com"}
HUB_HOSTS = frozenset(_HUB_HOSTS)  # the one list other modules read (comfy's links)
GITHUB_HOSTS = frozenset({"github.com", "www.github.com"})  # comfy's workflow and pack links
_REPO_SEGMENT = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_REPO_ID_RE = re.compile(rf"^{_REPO_SEGMENT}/{_REPO_SEGMENT}$")
_QUANT_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CLI_PREFIX_RE = re.compile(r"^(?:huggingface-cli|hf-cli|hf)\s+(?:download|d)\s+", re.IGNORECASE)
_SCHEME_PREFIX_RE = re.compile(r"^(?:hf|huggingface)://", re.IGNORECASE)
_OLLAMA_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*:[A-Za-z0-9][A-Za-z0-9._-]*$")
_REVISIONED_SECTIONS = {"tree", "blob", "resolve"}
_IGNORED_SECTIONS = {"discussions", "commits", "settings"}
_NOT_MODEL_SECTIONS = {
    "datasets",
    "spaces",
    "collections",
    "papers",
    "blog",
    "docs",
    "tasks",
    "api",
}


class UnsupportedReferenceError(ValueError):
    """The input is syntactically recognisable but not something we can resolve."""


def ollama_link(tag: str) -> str:
    """The ollama.com link of a library tag as the Ollama CLI takes it (`Qwen2.5:14b`): the
    site's names are lower case, a tag keeps its own case (`q4_K_M`)."""
    name, colon, size = tag.partition(":")
    return f"ollama.com/library/{name.lower()}{colon}{size}"


class OllamaTagError(UnsupportedReferenceError):
    """`qwen2.5:14b`: a bare name:tag with no slash, as the Ollama CLI uses."""

    def __init__(self, tag: str) -> None:
        super().__init__(
            f"'{tag}' looks like an Ollama tag. Paste its link ({ollama_link(tag)}) "
            "or a Hugging Face repo instead (for example unsloth/Qwen2.5-14B-Instruct-GGUF)."
        )
        self.tag = tag


@dataclass(frozen=True)
class ModelRef:
    """A repository on the Hub, optionally narrowed to a revision, path, or quant tag."""

    repo_id: str
    revision: str = "main"
    path: str | None = None
    quant: str | None = None
    # Where the link came from when it was another source's, for the plan to say.
    via: str | None = field(default=None, compare=False)

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}"


@dataclass(frozen=True)
class LocalPath:
    """A file or directory on this machine."""

    path: str


# The other sources' domains, listed once: each module builds its hosts (`www.` included)
# from them, and the refusal below names them. Their links are read before this module sees
# one (`plan.parse`), so a link refused here is from none of them. Civitai's domains are one
# site with the same model and version ids; the Civitai Archive keeps those ids too.
CIVITAI_DOMAINS = ("civitai.com", "civitai.green", "civitai.red")
ARCHIVE_DOMAIN = "civitaiarchive.com"
LMSTUDIO_DOMAIN = "lmstudio.ai"
_ONLY = (
    f"Only Hugging Face (huggingface.co, hf-mirror.com), Civitai ({', '.join(CIVITAI_DOMAINS)}, "
    f"{ARCHIVE_DOMAIN}), Ollama (ollama.com), ModelScope (modelscope.cn) and LM Studio "
    f"({LMSTUDIO_DOMAIN}) links are supported, not "
    "{host}."
)


def hosts(*domains: str) -> frozenset[str]:
    """Each domain and its `www.` name."""
    return frozenset(h for d in domains for h in (d, f"www.{d}"))


def parse_reference(text: str) -> ModelRef | LocalPath:
    """Parse user input. Raises UnsupportedReferenceError with a message meant for the user."""
    raw = text.strip().strip("\"'")
    if not raw:
        raise UnsupportedReferenceError(
            "Paste a Hugging Face link or a repo id like org/model-name."
        )
    if raw.startswith(("/", "~", "./", "../")):
        return LocalPath(raw)

    spec = _repo_token_of_cli_line(raw) if _CLI_PREFIX_RE.match(raw) else _single_token(raw)
    spec = _SCHEME_PREFIX_RE.sub("https://hf.co/", spec)
    first = spec.split("/", 1)[0].lower()
    if "://" not in spec and first in _HUB_HOSTS:
        spec = "https://" + spec
    elif "://" not in spec and "/" in spec and _OTHER_HOST_RE.match(first):
        # `ollama.com/library/qwen`, `modelscope.cn/models/…`: a site, not an org
        raise UnsupportedReferenceError(_ONLY.format(host=first))
    return _parse_url(spec) if "://" in spec else _parse_repo_spec(spec)


def _single_token(raw: str) -> str:
    if any(ch.isspace() for ch in raw):
        raise UnsupportedReferenceError(f"'{raw}' contains spaces; paste one link or repo id.")
    return raw


def _repo_token_of_cli_line(raw: str) -> str:
    """`hf download [--opts] org/name [--opts]` keeps the token shaped like a repo id."""
    tokens = _CLI_PREFIX_RE.sub("", raw).split()
    repo_like = (
        t
        for i, t in enumerate(tokens)
        if "/" in t
        and not t.startswith(("-", ".", "/", "~"))  # `--local-dir ./x`: a folder, not the repo
        and not (i and tokens[i - 1] in _VALUE_OPTIONS)
        and not _is_glob(t)
    )
    token = next(repo_like, None)
    if token is None:
        raise UnsupportedReferenceError(f"'{raw}' has no repo id after the download command.")
    return token


# Options of `hf download` / `huggingface-cli download` that take the next token as a value.
_VALUE_OPTIONS = frozenset(
    {
        "--local-dir",
        "--cache-dir",
        "--include",
        "--exclude",
        "--revision",
        "--repo-type",
        "--token",
        "--max-workers",
    }
)
# Hub users and orgs are letters, digits and dashes: a first segment ending in `.<letters>`
# is a site (`modelscope.cn`, `ollama.com`), whatever its top-level domain.
_OTHER_HOST_RE = re.compile(r"^[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,24}$", re.I)


def _is_glob(token: str) -> bool:
    return any(ch in token for ch in "*?[")


def _parse_url(url: str) -> ModelRef:
    parts = urlsplit(url)
    if parts.hostname not in _HUB_HOSTS:
        raise UnsupportedReferenceError(_ONLY.format(host=parts.hostname or url))
    segments = [unquote(s) for s in parts.path.strip("/").split("/") if s]
    if segments and segments[0] in _NOT_MODEL_SECTIONS:
        raise UnsupportedReferenceError(f"{segments[0]} links are not models.")
    if len(segments) < 2:
        raise UnsupportedReferenceError(
            "A model link needs the form huggingface.co/org/model-name."
        )

    repo_id, quant = _split_quant("/".join(segments[:2]))
    _require_repo_id(repo_id)
    rest = segments[2:]
    if not rest or rest[0] in _IGNORED_SECTIONS:
        return ModelRef(repo_id, quant=quant)
    if rest[0] in _REVISIONED_SECTIONS:
        revision = rest[1] if len(rest) > 1 else "main"
        return ModelRef(repo_id, revision=revision, path="/".join(rest[2:]) or None, quant=quant)
    raise UnsupportedReferenceError(f"Unrecognised link section '/{rest[0]}' in {url}.")


def _parse_repo_spec(spec: str) -> ModelRef:
    if "/" not in spec:
        if _OLLAMA_TAG_RE.match(spec):
            raise OllamaTagError(spec)
        raise UnsupportedReferenceError(
            f"'{spec}' is not a repo id. Use the form org/model-name or a huggingface.co link."
        )
    spec = spec.rstrip("/")  # `org/name/` is `org/name`
    segments = spec.split("/")
    if len(segments) > 2:  # a file (one version of many) or a folder (a package) in the repo
        repo_id = "/".join(segments[:2])
        _require_repo_id(repo_id)
        path = "/".join(segments[2:])
        if not all(_PATH_SEGMENT_RE.match(seg) for seg in segments[2:]):
            raise UnsupportedReferenceError(f"'{path}' is not a file or folder path in {repo_id}.")
        return ModelRef(repo_id, path=path)
    repo_id, quant = _split_quant(spec)
    _require_repo_id(repo_id)
    return ModelRef(repo_id, quant=quant)


_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]*$")  # a Hub file or folder name


def _split_quant(spec: str) -> tuple[str, str | None]:
    """`org/name:Q4_K_M` -> (`org/name`, `Q4_K_M`). Only the second segment may carry a tag."""
    head, sep, tag = spec.partition(":")
    if not sep:
        return spec, None
    if not tag or not _QUANT_TAG_RE.match(tag):
        raise UnsupportedReferenceError(f"'{spec}' has an empty or invalid quant tag after ':'.")
    return head, tag


def _require_repo_id(repo_id: str) -> None:
    if not _REPO_ID_RE.match(repo_id):
        raise UnsupportedReferenceError(
            f"'{repo_id}' is not a valid repo id (expected org/model-name with letters, digits, "
            "'.', '_' or '-')."
        )
