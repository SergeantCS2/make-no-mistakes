"""Custom node packs: the vetted list, the Comfy Registry's answers about a pack,
and the user's approvals.

A pack is installed only once it is approved, at one version, with the sha256 of its zip:
a vetted row carries the hash mnm recorded, any other pack is hashed when the user approves
it. The Registry refuses for us what it has banned, deleted or flagged. A pack off the
Registry is approved at a GitHub release tag or commit (`github_packs.py`): its
code fetched with git at that commit and packed by mnm, the zip's sha256 the pin. Approval
does not make code safe: the container has no internet, no secrets and the models read-only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from makenomistakes.comfy import cloud
from makenomistakes.comfy.cloud import cloud_classes_in, cloud_pack
from makenomistakes.comfy.github_packs import (
    GITHUB,
    Fetched,
    GitHub,
    GitHubError,
    GitHubRef,
    RepoFacts,
    parse_github,
)
from makenomistakes.comfy.registry import (
    PackError,
    Registry,
    RegistryNode,
    RegistryVersion,
    folder_of,
)
from makenomistakes.locking import shared_lock

ACTIVE_NODE = "NodeStatusActive"
ACTIVE_VERSION = "NodeVersionStatusActive"
SANDBOX_NOTE = (
    "Approving a pack lets its code run inside ComfyUI's container. That container has no "
    "internet, no secrets and no home folder, and reads the models folder only; still, "
    "approve only what you would run."
)


class PackWeight(BaseModel):
    """A file the pack would download at run time, fetched by mnm instead (no internet)."""

    model_config = ConfigDict(frozen=True)

    name: str  # inside the pack's weights folder: `rife/rife47.pth`
    url: str  # a Hugging Face file link


class VettedPack(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pack: str  # the Registry id
    version: str
    sha256: str  # of the Registry's zip of that version
    licence: str  # SPDX
    repository: str
    what: str
    requirements: str = "requirements.txt"
    weights_dir: str = ""  # the pack's own folder for its downloads, linked to mnm's copy
    weights: tuple[PackWeight, ...] = ()
    model_folders: tuple[str, ...] = ()  # ComfyUI model folders the pack adds
    note: str = ""
    classes: tuple[str, ...] = ()  # the node classes it registers: a workflow saved without
    # its pack's id is matched by these


class _Vetted(BaseModel):
    packs: tuple[VettedPack, ...]


# The list grows only by a change to mnm itself: `vetted.json` beside this file, data that
# reads well in a diff (`vetted_tools.py` checks each row against the Registry).
VETTED_FILE = Path(__file__).parent / "vetted.json"
VETTED = _Vetted.model_validate_json(VETTED_FILE.read_bytes()).packs


def vetted(pack: str, version: str | None = None) -> VettedPack | None:
    """The vetted row for a pack (its id in any case), at that version when one is given."""
    for row in VETTED:
        if row.pack.lower() == pack.lower() and version in (None, row.version):
            return row
    return None


def vetted_for_class(class_type: str) -> VettedPack | None:
    """The vetted pack that registers this node class (a workflow saved without `cnr_id`)."""
    return next((row for row in VETTED if class_type in row.classes), None)


# -- what mnm thinks of a pack ---------------------------------------------------------------


@dataclass
class Assessment:
    pack: str
    version: str
    row: VettedPack | None = None
    node: RegistryNode | None = None
    release: RegistryVersion | None = None
    refusals: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # A pack off the Registry, at a release tag or commit, its code already fetched
    github: GitHubRef | None = None
    fetched: Fetched | None = None
    facts: RepoFacts | None = None

    @property
    def ok(self) -> bool:
        return not self.refusals

    @property
    def licence(self) -> str:
        if self.row is not None:
            return self.row.licence
        if self.fetched is not None:
            return self.fetched.licence
        return licence_words(self.node.license if self.node else "")

    # Where a pack's code comes from is answered here, once: the Registry's release (its
    # zip downloaded at approval) or a GitHub commit (packed by mnm while it was assessed).

    @property
    def ready(self) -> bool:
        """Approvable as assessed: no refusal, and its code within reach."""
        return self.ok and (self.release is not None or self.fetched is not None)

    def code(self, registry: Registry) -> bytes:
        """The pack's zip."""
        if self.fetched is not None:
            return self.fetched.data
        assert self.release is not None  # `ready`
        return registry.download(self.release.downloadUrl)

    def approval(self, digest: str) -> Approval:
        """The record of the user's yes, in its source's shape."""
        if self.github is not None and self.fetched is not None:
            return Approval(
                pack=self.pack, version=self.version, sha256=digest, licence=self.licence,
                repository=self.github.url, vetted=self.row is not None,
                approved_at=datetime.now(UTC), source="github", ref=self.github.ref,
                commit=self.fetched.commit,
            )  # fmt: skip
        return Approval(
            pack=self.node.id if self.node is not None else self.pack, version=self.version,
            sha256=digest, licence=self.licence,
            repository=self.node.repository if self.node is not None else "",
            vetted=self.row is not None, approved_at=datetime.now(UTC),
        )  # fmt: skip


def assess(
    registry: Registry, pack: str, version: str | None = None, github: GitHub | None = None
) -> Assessment:
    """Everything the approval page shows, and the refusals that stop it. A GitHub pack
    (`github:owner/repo@ref`, or a link) is asked of `github`, its code fetched to read."""
    try:
        ref = parse_github(pack)
    except GitHubError as exc:
        return Assessment(pack, version or "", refusals=[str(exc)])
    if ref is not None:
        if version and not ref.ref:
            ref = GitHubRef(ref.owner, ref.repo, version)
        return assess_github(github, ref)
    return _assess_registry(registry, pack, version)


def _assess_registry(registry: Registry, pack: str, version: str | None) -> Assessment:
    row = vetted(pack)
    node = registry.node(pack)
    if node is None:
        return Assessment(pack, version or "", refusals=[f"The Registry has no pack '{pack}'."])
    wanted = version or (row.version if row else None)
    if wanted is None and node.latest_version is not None:
        wanted = node.latest_version.version
    found = Assessment(node.id, wanted or "", node=node)
    found.row = vetted(node.id, wanted)
    found.refusals += [f"Its {cloud.words(*sent)}." for sent in cloud_pack(node.id)]
    if row is not None and found.row is None:
        found.notes.append(f"mnm vetted version {row.version}; this is a separate approval.")
    if node.status != ACTIVE_NODE:
        found.refusals.append(f"The Registry marks the pack {_word(node.status)}.")
    if node.publisher is not None and node.publisher.status.endswith("Banned"):
        found.refusals.append(f"The Registry has banned its publisher {node.publisher.id}.")
    if not wanted:
        found.refusals.append("The pack has no published version.")
        return found
    found.release = registry.version(node.id, wanted)
    if found.release is None:
        found.refusals.append(f"The Registry has no version {wanted} of {node.id}.")
        return found
    if found.release.status != ACTIVE_VERSION:
        reason = f" ({found.release.status_reason})" if found.release.status_reason else ""
        found.refusals.append(
            f"The Registry marks version {wanted} {_word(found.release.status)}{reason}."
        )
    if found.release.deprecated:
        found.notes.append(f"Version {wanted} is deprecated by its author.")
    if not found.release.downloadUrl:
        found.refusals.append(f"Version {wanted} has no download.")
    if found.row is not None and found.row.note:
        found.notes.append(found.row.note)
    return found


def assess_github(github: GitHub | None, ref: GitHubRef) -> Assessment:
    """A GitHub pack: its ref resolved to a commit (a branch refused), its code fetched with
    git and packed by mnm, its licence and requirements read, the repository's facts asked."""
    found = Assessment(ref.repo, ref.ref, github=ref)
    if github is None:
        found.refusals.append(
            f"{ref.words} is a GitHub pack, fetched with git and approved at a release tag or "
            f"commit (`github:{ref.owner}/{ref.repo}@<tag or commit>`), not from the Registry."
        )
        return found
    try:
        commit, kind = github.resolve(ref)
        found.fetched = github.fetch(ref, commit, in_history=kind == "commit")
    except GitHubError as exc:
        found.refusals.append(str(exc))
        return found
    found.refusals += [
        f"Its {cloud.words(*sent)}." for sent in cloud_classes_in(found.fetched.data)
    ]
    found.version = ref.ref if kind == "tag" else commit[:12]
    found.notes += found.fetched.notes
    found.facts = github.facts(ref)
    if found.facts is None:
        found.notes.append("GitHub's API did not answer: its stars and open issues are not known.")
    elif found.facts.archived:
        found.notes.append("Its repository is archived: nobody fixes it any more.")
    return found


def licence_words(raw: str) -> str:
    """The Registry's licence field, which is pyproject's: `{"file": "LICENSE"}` names a file
    in the repository, `{"text": …}` holds it; else it is the name itself."""
    try:
        found = json.loads(raw) if raw.startswith("{") else raw
    except ValueError:
        found = raw
    if isinstance(found, dict):
        text, file = found.get("text"), found.get("file")
        if isinstance(text, str):
            return text[:80]
        if isinstance(file, str):
            return f"see {file} in its repository"
    return str(found or "not stated")[:80]


def _word(status: str) -> str:
    """`NodeVersionStatusFlagged` → `flagged`."""
    for prefix in ("NodeVersionStatus", "NodeStatus", "PublisherStatus"):
        status = status.removeprefix(prefix)
    return status.lower() or "unknown"


# -- approvals ---------------------------------------------------------------------------------


class Approval(BaseModel):
    pack: str
    version: str
    sha256: str
    licence: str
    repository: str = ""
    vetted: bool = False
    approved_at: datetime
    # Where the code came from; a GitHub pack's ref and the commit it named
    source: Literal["registry", "github"] = "registry"
    ref: str = ""
    commit: str = ""

    @property
    def reference(self) -> str:
        """What `mnm comfyui packs add` takes to fetch this approval's code again: a GitHub
        pack at the commit it pinned, never its tag, which may name other code by then."""
        if self.source == "github":
            return f"github:{self.repository.removeprefix(f'{GITHUB}/')}@{self.commit or self.ref}"
        return f"{self.pack} --version {self.version}"

    @property
    def names(self) -> set[str]:
        """What a workflow may call it, lowercased: its id, and a GitHub pack's `owner/repo`
        (a node's `aux_id`)."""
        found = {self.pack.lower()}
        if self.source == "github":
            found.add(self.repository.removeprefix(f"{GITHUB}/").removesuffix(".git").lower())
        return found


# 2: `source`, `ref`, `commit`; a schema-1 file reads as is (every new field has a
# default: its approvals are the Registry's) and is written back as 2, which an older mnm
# refuses instead of dropping a GitHub pin.
APPROVALS_SCHEMA = 2


class _Approvals(BaseModel):
    schema_version: int = APPROVALS_SCHEMA
    approvals: list[Approval] = Field(default_factory=list)


class PackStore:
    """`<state>/comfy/packs.json`: one approved version per pack; the zips kept beside it."""

    def __init__(self, state_dir: Path) -> None:
        self.folder = state_dir / "comfy" / "packs"
        self.path = state_dir / "comfy" / "packs.json"
        self._lock = shared_lock(self.folder)  # the CLI's add and a web click take turns

    def approvals(self) -> list[Approval]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise PackError(f"Could not read {self.path}: {exc}") from exc
        try:
            body = _Approvals.model_validate_json(text)
        except ValidationError as exc:
            raise PackError(f"{self.path} is not a list of approvals mnm can read.") from exc
        if body.schema_version > APPROVALS_SCHEMA:
            raise PackError(f"{self.path} is from a newer mnm: run `mnm update`.")
        return body.approvals

    def get(self, pack: str) -> Approval | None:
        """By its id, or a GitHub pack by `github:owner/repo` (any ref) as it was added."""
        asked = parse_github(pack)
        wanted = f"{asked.owner}/{asked.repo}".lower() if asked else pack.lower()
        return next((a for a in self.approvals() if wanted in a.names), None)

    def _save(self, approvals: list[Approval]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        body = _Approvals(approvals=sorted(approvals, key=lambda a: a.pack.lower()))
        tmp.write_text(body.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def zip_path(self, approval: Approval) -> Path:
        return self.folder / "zips" / f"{folder_of(approval.pack)}-{approval.version}.zip"

    def approve(self, found: Assessment, data: bytes) -> Approval:
        """Record the user's yes, with the zip's hash (a vetted row's must match)."""
        if not found.ok or (found.node is None and found.fetched is None):
            raise PackError("; ".join(found.refusals) or f"{found.pack} cannot be approved.")
        digest = hashlib.sha256(data).hexdigest()
        if found.row is not None and digest != found.row.sha256:
            raise PackError(
                f"{found.pack} {found.version}'s zip is not the one mnm vetted (sha256 "
                f"{digest[:12]}…, expected {found.row.sha256[:12]}…). Nothing was installed."
            )
        approval = found.approval(digest)
        target = self.zip_path(approval)
        with self._lock():  # read, changed and saved together: no approval lost to another
            before = self.approvals()
            _refuse_clash(before, approval)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            others = [a for a in before if a.pack.lower() != approval.pack.lower()]
            self._save([*others, approval])
            for replaced in (a for a in before if a not in others):  # an older version's zip
                if self.zip_path(replaced) != target:
                    self.zip_path(replaced).unlink(missing_ok=True)
        return approval

    def remove(self, pack: str) -> Approval:
        with self._lock():
            found = self.get(pack)
            if found is None:
                raise PackError(f"{pack} is not approved.")
            self._save([a for a in self.approvals() if a.pack.lower() != found.pack.lower()])
            self.zip_path(found).unlink(missing_ok=True)
        return found

    def zip_bytes(self, approval: Approval) -> bytes:
        """The approved zip, checked against its hash: a changed file is refused."""
        try:
            data = self.zip_path(approval).read_bytes()
        except OSError as exc:
            raise PackError(
                f"{approval.pack}'s zip is gone (a backup leaves the zips out): `mnm comfyui "
                f"packs add {approval.reference}` fetches it again."
            ) from exc
        if hashlib.sha256(data).hexdigest() != approval.sha256:
            raise PackError(f"{approval.pack}'s zip changed since it was approved; add it again.")
        return data


def approve_assessed(
    registry: Registry, store: PackStore, found: Assessment, confirmed: bool
) -> Approval:
    """The one approval sequence the CLI and the web share: refused stays refused, a pack
    mnm did not vet needs the user's yes (`confirmed`), then the zip is fetched and pinned."""
    check_approvable(found, confirmed)
    return store.approve(found, found.code(registry))


def check_approvable(found: Assessment, confirmed: bool) -> None:
    """The refusals before any download (the web says them at once, then downloads in a job)."""
    if not found.ready:
        raise PackError("; ".join(found.refusals) or f"{found.pack} is refused.")
    if found.row is None and not confirmed:
        raise PackError(f"{found.pack} {found.version} is not vetted by mnm: it needs your yes.")


def set_hash(approvals: list[Approval]) -> str:
    """The approved set, as the layer's tag suffix."""
    lines = sorted(f"{a.pack}@{a.version}:{a.sha256}" for a in approvals)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:10]


def _repository(url: str) -> str:
    return url.strip().lower().removesuffix("/").removesuffix(".git")


def _refuse_clash(approvals: list[Approval], approval: Approval) -> None:
    """One approval per pack name, and a new one replaces the old only when it is the same
    repository's code (a newer version, or the same repo from GitHub instead of the
    Registry): a fork, or another repository that happens to share the name, is refused
    rather than silently put in its place (ComfyUI loads one `custom_nodes/<name>`)."""
    for held in approvals:
        same = held.pack.lower() == approval.pack.lower()
        if same and _repository(held.repository) != _repository(approval.repository):
            where = held.repository or "the Registry"
            raise PackError(
                f"{held.pack} {held.version} is already approved, from {where}; "
                f"{approval.repository or approval.pack} is another pack of that name. "
                f"`mnm comfyui packs rm {held.pack}` first to use this one instead."
            )


def standing(
    pack: str, version: str | None, approvals: list[Approval], installed: list[str]
) -> str:
    """One pack a workflow uses, in a word or three: what the user does next."""
    approval = next((a for a in approvals if pack.lower() in a.names), None)
    if approval is not None and f"{approval.pack}@{approval.version}".lower() in {
        p.lower() for p in installed
    }:
        return "installed"
    if approval is not None:
        return "approved: install it"
    if "/" in pack:  # a workflow's `aux_id`: the GitHub repository, its `ver` the commit
        return "on GitHub: approve it at its commit"
    return "vetted: add it" if vetted(pack) is not None else "needs your approval"


def approved_names(state_dir: Path) -> set[str]:
    """Every name an approved pack goes by, lowercased (`Approval.names`): what a workflow's
    pack is looked up in; none when the file cannot be read."""
    try:
        return {name for a in PackStore(state_dir).approvals() for name in a.names}
    except PackError:
        return set()


def approved_ids(state_dir: Path) -> set[str]:
    """The packs the user approved (built or not); none when the file cannot be read."""
    try:
        return {a.pack for a in PackStore(state_dir).approvals()}
    except PackError:
        return set()
