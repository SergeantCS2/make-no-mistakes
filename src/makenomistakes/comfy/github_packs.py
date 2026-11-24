"""Node packs pinned on GitHub (a repository at a commit hash, never a branch): the best audio
packs are not on the Comfy Registry, so the user may approve one at a release tag or a commit.

How a pack is pinned:
- The user names `github:owner/repo@<tag or commit>`, or pastes a GitHub link to a release
  tag, a commit or a tree at one. A ref that is a branch (`main`, `latest` …) is refused: a
  branch moves.
- The ref is resolved to its commit with `git ls-remote`, and the code is fetched with `git`
  at exactly that commit (`git fetch --depth 1 <commit>`; `rev-parse HEAD` must answer it).
- mnm packs the checkout into a zip of its own making: files sorted, stored, fixed times, no
  `.git`, no links. The same commit gives the same bytes on any machine, so its sha256 is a
  pin a vetted row can carry and the build re-checks.

Not the sha256 of GitHub's own zip: GitHub does not promise its archives' bytes stay the same,
and its zip downloads can be refused where `git` still reaches GitHub. The commit is a content
hash in itself.

What the approval shows, as a Registry pack's does: the licence (its LICENSE file, read from
the code), the pip packages its requirements file asks for, and the repository's stars, open
issues and whether it is archived (GitHub's API, a note when it does not answer).
"""

from __future__ import annotations

import hashlib
import io
import re
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ValidationError

from makenomistakes.comfy.pack_rules import MAX_FILES, MAX_UNPACKED, requirements
from makenomistakes.comfy.registry import PackError
from makenomistakes.hardware.probe import CommandRunner
from makenomistakes.hub.refs import GITHUB_HOSTS

GITHUB = "https://github.com"
API = "https://api.github.com"
GIT_SECONDS = 300.0
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+/-]{0,99}$")
_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHORT_SHA = re.compile(r"^[0-9a-f]{7,39}$")
_PARTS = re.compile(r"\d+|[^\d]+")
ZIP_TIME = (1980, 1, 1, 0, 0, 0)
# git as the pack's own bytes: no prompt for a password (a private or mistyped repository
# fails), none of the user's config (no credential helper sends a token, no line-ending
# conversion or Git LFS filter changes a file), so the same commit packs the same everywhere
GIT_ENV = [
    "env",
    "GIT_TERMINAL_PROMPT=0",
    "GIT_ASKPASS=",
    "GIT_CONFIG_NOSYSTEM=1",
    "GIT_CONFIG_GLOBAL=/dev/null",
    "GIT_LFS_SKIP_SMUDGE=1",
]
HISTORY_REFS = ("+refs/heads/*:refs/remotes/origin/*", "+refs/tags/*:refs/tags/*")
HELD_FORMAT = ("--format=%(refname:short)", "refs/remotes/origin", "refs/tags")
# Licence files' first words, to the SPDX id they name (the common ones; the rest are shown
# as their first line). A versioned family is named with the version its text says right
# after its name ("GNU GENERAL PUBLIC LICENSE Version 2"), never one assumed.
_LICENCES = (
    ("mit license", "MIT"),
    ("bsd 3-clause", "BSD-3-Clause"),
    ("bsd 2-clause", "BSD-2-Clause"),
    ("the unlicense", "Unlicense"),
)
_VERSIONED = (
    ("gnu affero general public license", "AGPL"),
    ("gnu lesser general public license", "LGPL"),
    ("gnu library general public license", "LGPL"),
    ("gnu general public license", "GPL"),
    ("apache license", "Apache"),
    ("mozilla public license", "MPL"),
)
_VERSION = re.compile(r"[\s,]*version\s+(\d+)(\.\d+)?\b")
_CC_NC = re.compile(
    r"attribution-noncommercial(-sharealike|-noderivatives|-noderivs)?\s+(\d\.\d)\b"
)
_CC_PARTS = {"-sharealike": "-SA", "-noderivatives": "-ND", "-noderivs": "-ND"}


def newest_tags(tags: list[str], count: int = 3) -> list[str]:
    """The newest tags by their numbers (`v5.10.0` after `v5.9.2`), newest first."""

    def key(tag: str) -> list[tuple[int, int | str]]:
        return [(0, int(part)) if part.isdigit() else (1, part) for part in _PARTS.findall(tag)]

    return sorted(tags, key=key, reverse=True)[:count]


def _offer(ref: GitHubRef, listed: RemoteRefs) -> str:
    """What the user may name instead: the newest tags, else the main branch's commit now
    (pinned, it never moves)."""
    newest = newest_tags(list(listed.tags))
    if newest:
        return (
            f" Its newest tags: {', '.join(newest)} (`github:{ref.owner}/{ref.repo}@{newest[0]}`)."
        )
    branch = next((b for b in ("main", "master") if b in listed.heads), None)
    if branch is None:
        return " It has no tags: name a commit (all 40 characters)."
    return (
        f" It has no tags; its {branch} branch is at {listed.heads[branch]} now, and "
        f"`github:{ref.owner}/{ref.repo}@{listed.heads[branch]}` pins that commit."
    )


class GitHubError(PackError):
    """A GitHub pack mnm will not approve, worded for the user: a PackError, so every pack
    surface (the CLI, the web's look-up box and pages, `packs rm`) words it, never a 500."""


@dataclass(frozen=True)
class GitHubRef:
    owner: str
    repo: str
    ref: str  # a release tag or a commit, as the user gave it ("" when none was)

    @property
    def url(self) -> str:
        return f"{GITHUB}/{self.owner}/{self.repo}"

    @property
    def words(self) -> str:
        return f"github:{self.owner}/{self.repo}" + (f"@{self.ref}" if self.ref else "")


def parse_github(text: str) -> GitHubRef | None:
    """`github:owner/repo@ref`, `owner/repo@ref`, or a GitHub link to a release tag, a commit
    or a tree; None for anything else (a Registry id)."""
    text = text.strip()
    if text.lower().startswith(tuple(f"{host}/" for host in GITHUB_HOSTS)):  # no scheme
        text = f"https://{text}"
    if text.startswith(("https://", "http://")):
        found = urlparse(text)
        if (found.hostname or "").lower() not in GITHUB_HOSTS:
            return None
        parts = [p for p in found.path.split("/") if p]
        if len(parts) < 2:
            return None
        repo, _, ref = parts[1].partition("@")  # `github.com/o/r@v1`: the short form's ref
        owner, repo, rest = parts[0], repo.removesuffix(".git"), parts[2:]
        if rest[:2] == ["releases", "tag"] and len(rest) >= 3:
            ref = "/".join(rest[2:])
        elif rest[:1] in (["commit"], ["tree"]) and len(rest) >= 2:
            ref = "/".join(rest[1:])
        return _checked(owner, repo, ref)
    if text.startswith("github:"):
        text = text.removeprefix("github:")
    elif "/" not in text:
        return None
    name, _, ref = text.partition("@")
    owner, sep, repo = name.partition("/")
    if not sep:
        return None
    return _checked(owner, repo.removesuffix(".git"), ref)


def _checked(owner: str, repo: str, ref: str) -> GitHubRef:
    if not _NAME.match(owner) or not _NAME.match(repo) or ".." in repo:
        raise GitHubError(f"'{owner}/{repo}' is not a GitHub repository name.")
    if ref and (not _REF.match(ref) or ".." in ref):
        raise GitHubError(f"'{ref}' is not a release tag or a commit.")
    return GitHubRef(owner, repo, ref)


class RemoteRefs(BaseModel):
    """What `git ls-remote --tags --heads` lists: each tag's commit (an annotated tag's
    peeled one) and each branch's head."""

    tags: dict[str, str]
    heads: dict[str, str]


def parse_ls_remote(text: str) -> RemoteRefs:
    tags: dict[str, str] = {}
    peeled: dict[str, str] = {}
    heads: dict[str, str] = {}
    for line in text.splitlines():
        sha, _, name = line.strip().partition("\t")
        if not _SHA.match(sha):
            continue
        if name.startswith("refs/tags/") and name.endswith("^{}"):
            peeled[name.removeprefix("refs/tags/").removesuffix("^{}")] = sha
        elif name.startswith("refs/tags/"):
            tags[name.removeprefix("refs/tags/")] = sha
        elif name.startswith("refs/heads/"):
            heads[name.removeprefix("refs/heads/")] = sha
    return RemoteRefs(tags=tags | peeled, heads=heads)


class RepoFacts(BaseModel):
    """GitHub's API on the repository, as the approval page shows it."""

    stargazers_count: int = 0
    open_issues_count: int = 0
    archived: bool = False


@dataclass
class Fetched:
    """A pack's code at one commit, packed by mnm, and what it says of itself."""

    commit: str
    data: bytes
    licence: str
    requirements: list[str]
    notes: list[str]

    @property
    def sha256(self) -> str:
        """What the approval pins, shown on the page and posted back with its Approve."""
        return hashlib.sha256(self.data).hexdigest()


def default_http() -> httpx.Client:
    """The client `GitHub.facts` asks GitHub's API with when given none (read at each call,
    so a caller can swap it)."""
    return httpx.Client(timeout=20.0, follow_redirects=True)


class GitHub:
    """`git` for the code, GitHub's API for the repository's facts (a note when it does not
    answer)."""

    def __init__(self, runner: CommandRunner, http: httpx.Client | None = None) -> None:
        self.runner = runner
        self.http = http

    def _git(self, args: list[str], why: str) -> str:
        code, out, err = self.runner.run([*GIT_ENV, "git", *args], timeout=GIT_SECONDS)
        if code != 0:
            said = (err or out).strip().splitlines()[-1:] or [f"exit {code}"]
            # the runner names `env` (GIT_ENV's first word) in a timeout: git's, in fact
            raise GitHubError(f"git could not {why}: {said[0].removeprefix('env: ')[:300]}")
        return out

    def resolve(self, ref: GitHubRef) -> tuple[str, str]:
        """The commit `ref.ref` names, and what it is (`tag` or `commit`). A branch, a missing
        ref or no ref at all is refused with what to name instead."""
        listed = parse_ls_remote(
            self._git(["ls-remote", "--tags", "--heads", ref.url], f"read {ref.url}'s tags")
        )
        if not ref.ref:
            raise GitHubError(
                f"{ref.words} names no release: a GitHub pack is pinned to a release tag or a "
                "commit." + _offer(ref, listed)
            )
        if ref.ref in listed.tags:
            return listed.tags[ref.ref], "tag"
        if ref.ref in listed.heads or ref.ref.lower() in ("latest", "head"):
            raise GitHubError(
                f"'{ref.ref}' is a branch of {ref.owner}/{ref.repo}, and a branch moves: mnm pins "
                f"a release tag or a commit.{_offer(ref, listed)}"
            )
        if _SHA.match(ref.ref):
            return ref.ref, "commit"
        if _SHORT_SHA.match(ref.ref):
            raise GitHubError(f"'{ref.ref}' is a short commit: give all 40 characters.")
        raise GitHubError(f"{ref.owner}/{ref.repo} has no tag '{ref.ref}'.{_offer(ref, listed)}")

    def fetch(self, ref: GitHubRef, commit: str, *, in_history: bool = True) -> Fetched:
        """The code at `commit`, checked, packed by mnm. A commit named by its sha
        (`in_history`) must be in one of the repository's own branches or tags: GitHub serves
        a fork's commits through its parent's address too."""
        work = Path(tempfile.mkdtemp(prefix="mnm-github-pack-"))
        try:
            self._git(["init", "--quiet", str(work)], "make a folder to fetch into")
            self._git(
                ["-C", str(work), "fetch", "--quiet", "--depth", "1", ref.url, commit],
                f"fetch {ref.owner}/{ref.repo} at {commit[:12]}",
            )
            self._git(
                ["-C", str(work), "checkout", "--quiet", "--detach", "FETCH_HEAD"], "check it out"
            )
            head = self._git(["-C", str(work), "rev-parse", "HEAD"], "read the commit").strip()
            if head != commit:
                raise GitHubError(
                    f"git fetched {head[:12]}, not {commit[:12]}: nothing is approved."
                )
            notes = self._held(work, ref, commit) if in_history else []
            if (work / ".gitmodules").exists():
                notes.append("It has git submodules, which mnm does not fetch: they are missing.")
            data = pack_zip(work)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return Fetched(commit, data, licence_of(data), requirement_lines(data, ref.repo), notes)

    def _held(self, work: Path, ref: GitHubRef, commit: str) -> list[str]:
        """The branches and tags holding `commit`, read from their history (commits only:
        `--filter=tree:0`, small on a large repository); none is a refusal."""
        self._git(
            ["-C", str(work), "fetch", "--quiet", "--filter=tree:0", ref.url, *HISTORY_REFS],
            f"read {ref.owner}/{ref.repo}'s history",
        )
        holding = self._git(
            ["-C", str(work), "for-each-ref", "--contains", commit, *HELD_FORMAT],
            "check the commit is in its history",
        ).split()
        if not holding:
            raise GitHubError(
                f"{commit[:12]} is in no branch or tag of {ref.owner}/{ref.repo}: GitHub serves a "
                "fork's commits through its parent's address too, so mnm approves only a commit "
                "the repository holds. Nothing is approved."
            )
        named = ", ".join(h.removeprefix("origin/") for h in holding[:3])
        return [f"The commit is in its history ({named}{', …' if len(holding) > 3 else ''})."]

    def facts(self, ref: GitHubRef) -> RepoFacts | None:
        """Stars, open issues, archived; None when GitHub's API does not answer."""
        url = f"{API}/repos/{ref.owner}/{ref.repo}"
        try:
            if self.http is not None:
                response = self.http.get(url)
            else:
                with default_http() as client:
                    response = client.get(url)
            if not response.is_success:
                return None
            return RepoFacts.model_validate_json(response.content)
        except (httpx.HTTPError, ValidationError):
            return None


def pack_zip(folder: Path) -> bytes:
    """`folder`'s files in a zip mnm makes the same way every time: sorted, stored (no
    compression to vary), fixed times, a file's mode kept; `.git` left out; a link refused."""
    files = []
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder)
        if relative.parts[0] == ".git":
            continue
        if path.is_symlink():
            raise GitHubError(f"The pack holds a link ({relative}): mnm does not unpack links.")
        try:  # a name that is not UTF-8 has no zip name (zipfile raised UnicodeEncodeError)
            relative.as_posix().encode("utf-8")
        except UnicodeEncodeError:
            raise GitHubError(
                f"The pack holds a file whose name is not UTF-8 ({relative.as_posix()!r}): mnm "
                "does not pack it."
            ) from None
        if path.is_file():
            files.append((relative.as_posix(), path))
    if len(files) > MAX_FILES:
        raise GitHubError(f"The pack has more than {MAX_FILES} files.")
    total = sum(path.stat().st_size for _name, path in files)  # before any file is read
    if total > MAX_UNPACKED:
        raise GitHubError(
            f"The pack's files come to {total / 2**20:.0f} MiB, more than the "
            f"{MAX_UNPACKED // 2**20} MiB a pack is unpacked to."
        )
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, path in files:
            info = zipfile.ZipInfo(name, date_time=ZIP_TIME)
            mode = 0o755 if path.stat().st_mode & stat.S_IXUSR else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            archive.writestr(info, path.read_bytes())
    return out.getvalue()


def _read(data: bytes, wanted: tuple[str, ...]) -> str | None:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = {n.lower(): n for n in archive.namelist() if "/" not in n}
        for name in wanted:
            if name in names:
                return archive.read(names[name]).decode("utf-8", errors="replace")
    return None


def licence_of(data: bytes) -> str:
    """The licence its LICENSE file names (an SPDX id where mnm knows the text's first
    words, else that first line), or `no licence file`."""
    text = _read(data, ("license", "license.md", "license.txt", "licence", "copying"))
    if text is None:
        return "no licence file (all rights reserved unless its README says otherwise)"
    head = " ".join(text.split()).lower()[:300]
    named = _named(head)
    if named is not None:
        return named
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first[:80] or "an empty licence file"


def _named(head: str) -> str | None:
    """The SPDX id a licence text's head names, its version read from the text; None when
    mnm is not sure (shown as its first line)."""
    for words, spdx in _LICENCES:
        if words in head:
            return spdx
    for words, family in _VERSIONED:
        at = head.find(words)
        if at >= 0:
            found = _VERSION.match(head, at + len(words))
            return f"{family}-{found[1]}{found[2] or '.0'}" if found else None
    cc = _CC_NC.search(head)
    return f"CC-BY-NC{_CC_PARTS.get(cc[1] or '', '')}-{cc[2]}" if cc else None


def requirement_lines(data: bytes, pack: str) -> list[str]:
    """Its `requirements.txt` as the build installs it: a line pip would take past the pin
    (a URL, a file, an option) is a GitHubError here, before anything is approved."""
    try:
        return requirements(_read(data, ("requirements.txt",)) or "", pack)
    except PackError as exc:  # said as the GitHub fetch's refusal, where assess catches it
        raise GitHubError(str(exc)) from exc
