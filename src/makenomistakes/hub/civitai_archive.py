"""The Civitai Archive (civitaiarchive.com) as a fallback: it keeps Civitai's model
and version ids, so its links are read as Civitai's, and for a version Civitai no longer has
(its API answers 404) it names where the files went: mirrors per file, Hugging Face ones
among them (`https://huggingface.co/<org>/<repo>/resolve/<rev>/<path>`), with the sha256
it recorded. mnm downloads only from Hugging Face and Civitai, so a mirror is used only on
the Hub, and only when the Hub's own record of the file has that sha256. That proves the copy
is the file the archive recorded, no more: the record itself is third-party data.

`GET /api/models/<id>?modelVersionId=<v>` is the one call; it has no lookup by version alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from makenomistakes.hub.client import HubClient, HubError, TreeEntry
from makenomistakes.hub.refs import (
    ARCHIVE_DOMAIN,
    ModelRef,
    UnsupportedReferenceError,
    hosts,
    parse_reference,
)

ARCHIVE_URL = f"https://{ARCHIVE_DOMAIN}"
HOSTS = hosts(ARCHIVE_DOMAIN)
TIMEOUT_SECONDS = 20.0
HUB_MIRROR_PREFIX = "https://huggingface.co/"


class ArchiveError(HubError):
    """No usable copy in the archive, worded for the user."""


class _Loose(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class Mirror(_Loose):
    url: str
    source: str
    deleted_at: str | None = Field(default=None, alias="deletedAt")
    is_gated: bool = False
    is_paid: bool = False

    @property
    def on_the_hub(self) -> bool:
        """A Hugging Face copy that is still there and needs nothing to download."""
        usable = not (self.deleted_at or self.is_gated or self.is_paid)
        return usable and self.source == "huggingface" and self.url.startswith(HUB_MIRROR_PREFIX)


class ArchiveFile(_Loose):
    name: str
    sha256: str = ""
    mirrors: list[Mirror] = []


class ArchiveVersion(_Loose):
    id: int | str  # Civitai's version id (a number on the archive's Civitai pages)
    name: str = ""
    files: list[ArchiveFile] = []


class ArchiveModel(_Loose):
    id: int | str
    name: str
    type: str = ""
    version: ArchiveVersion


@dataclass(frozen=True)
class HubCopy:
    """A file of a version Civitai no longer has, as a Hub link to plan from."""

    model: str  # `<name> (<version name>)`, the archive's words
    type: str  # Civitai's type: Checkpoint, LORA, …
    link: str  # `https://huggingface.co/<org>/<repo>/resolve/<rev>/<path>`
    sha256: str


def hub_copies(found: ArchiveModel) -> list[HubCopy]:
    """Every Hugging Face mirror of the version's files that carries a checksum to match,
    the archive's order."""
    label = f"{found.name} ({found.version.name})" if found.version.name else found.name
    return [
        HubCopy(label, found.type, mirror.url, file.sha256.lower())
        for file in found.version.files
        if file.sha256
        for mirror in file.mirrors
        if mirror.on_the_hub
    ]


def matches(entry: TreeEntry, sha256: str) -> bool:
    """The Hub's own record of the file carries the checksum the archive recorded."""
    return entry.lfs is not None and entry.lfs.oid.lower() == sha256


@dataclass(frozen=True)
class Archived:
    """What a plan reads instead of a version Civitai no longer has."""

    ref: ModelRef  # the Hub file, pinned to the commit whose record was checked
    copy: HubCopy
    entry: TreeEntry  # the Hub's record of it, the checksum matched


def from_the_archive(
    hub: HubClient, archive: ArchiveClient, model_id: int | None, version_id: int | None, gone: str
) -> Archived:
    """The Hub copy of a version Civitai no longer has (`gone` says so): the first of the
    archive's Hugging Face mirrors whose file the Hub records with the archive's checksum.
    The reference is pinned to the commit that record was read at, so the download fetches
    the bytes that were checked even if the mirror's owner replaces the file later."""
    if model_id is None:
        raise ArchiveError(
            f"{gone} The Civitai Archive finds a version by its model's page: paste "
            "civitaiarchive.com/models/<id>?modelVersionId=<version>."
        )
    try:
        found = archive.model(model_id, version_id)
    except ArchiveError as exc:
        raise ArchiveError(f"{gone} {exc}") from exc
    copies = hub_copies(found)
    if not copies:
        kept = sorted({m.source for f in found.version.files for m in f.mirrors}) or ["none"]
        raise ArchiveError(
            f"{gone} The Civitai Archive knows {found.name} but has no Hugging Face copy of "
            f"its files (its mirrors: {', '.join(kept)}); mnm downloads from Hugging Face and "
            "Civitai only."
        )
    unreadable: list[str] = []
    for copy in copies:
        try:
            ref = parse_reference(copy.link)
        except UnsupportedReferenceError:
            continue
        if not isinstance(ref, ModelRef) or ref.path is None:
            continue
        try:
            commit = hub.commit(ref.repo_id, ref.revision)
            entry = hub.path_info(ref.repo_id, commit, ref.path)
        except HubError:  # a mirror repo since deleted or made private: try the next one
            unreadable.append(ref.repo_id)
            continue
        if entry is not None and matches(entry, copy.sha256):
            return Archived(ModelRef(ref.repo_id, commit, ref.path), copy, entry)
    skipped = f" ({', '.join(unreadable)} could not be read)" if unreadable else ""
    raise ArchiveError(
        f"{gone} The Civitai Archive's Hugging Face copies of {found.name} do not match the "
        f"checksum it recorded{skipped}, so none is used."
    )


class ArchiveClient:
    """Reads the archive's model API (no token: it is public)."""

    def __init__(self, transport: httpx.BaseTransport | None = None) -> None:
        self.http = httpx.Client(
            base_url=ARCHIVE_URL,
            transport=transport,
            timeout=TIMEOUT_SECONDS,
            follow_redirects=True,
        )

    def model(self, model_id: int, version_id: int | None) -> ArchiveModel:
        params = {"modelVersionId": str(version_id)} if version_id is not None else None
        try:
            response = self.http.get(f"/api/models/{model_id}", params=params)
        except httpx.HTTPError as exc:
            raise ArchiveError(f"Could not reach civitaiarchive.com: {exc}") from exc
        if (response.url.host or "").lower() not in HOSTS:
            raise ArchiveError(f"civitaiarchive.com redirected off the site ({response.url.host}).")
        if response.status_code == 404:
            raise ArchiveError(
                f"The Civitai Archive has no model {model_id} either: check the link."
            )
        if response.is_error:
            raise ArchiveError(f"civitaiarchive.com answered HTTP {response.status_code}.")
        try:
            return ArchiveModel.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise ArchiveError(f"The Civitai Archive's answer was not understood: {exc}") from exc

    def close(self) -> None:
        self.http.close()
