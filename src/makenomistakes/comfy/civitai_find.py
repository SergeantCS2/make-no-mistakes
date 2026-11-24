"""A workflow's model file looked for on Civitai: by its sha256 when the workflow
records one (the same bytes: linked at once), else by its exact file name among the models a
search for the name's words returns. Civitai's search reads model names, never file names,
and two creators can share a file name, so a name match is only offered: the user confirms
it (`--set`, or the web's Use button)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urlencode

from makenomistakes.comfy.report import ModelNeed
from makenomistakes.hub.civitai import CivitaiClient, CivitaiVersion

SEARCH_LIMIT = 20
_SPLIT = re.compile(r"[\s_.\-]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


@dataclass(frozen=True)
class Match:
    """One Civitai file that may be the one a workflow names."""

    model_id: int
    version_id: int
    model: str  # the model's name on Civitai
    version: str
    file: str  # the file's name on Civitai (a hash match may name it differently)
    size: int
    same_bytes: bool  # found by its sha256

    @property
    def url(self) -> str:
        """The version's page; `file` names the file among its files for mnm's download."""
        query = urlencode({"modelVersionId": self.version_id, "file": self.file})
        return f"https://civitai.com/models/{self.model_id}?{query}"

    @property
    def label(self) -> str:
        return f"Civitai: {self.model} ({self.version}), {self.file}"


def queries(name: str) -> list[str]:
    """Searches for a file name's words: up to its first `_` (Civitai names a download
    `<ModelName>_<version>`), then all of them; camelCase split."""
    stem = PurePosixPath(name.replace("\\", "/")).stem
    head = stem.split("_", 1)[0]
    found = []
    for part in (head, stem):
        words = " ".join(w for w in _SPLIT.split(_CAMEL.sub(" ", part)) if w)
        if words and words not in found:
            found.append(words)
    return found


def find(client: CivitaiClient, need: ModelNeed) -> list[Match]:
    """The same bytes when the workflow records a sha256 (nothing else is offered then);
    else files of exactly this name. Civitai refusing or not answering raises CivitaiError
    (worded): the caller says so rather than "not found"."""
    if need.sha256:
        version = client.by_hash(need.sha256)
        return [] if version is None else _matches(version, sha256=need.sha256)
    base = PurePosixPath(need.name.replace("\\", "/")).name
    found: dict[tuple[int, str], Match] = {}
    for query in queries(base):
        for model in client.search(query, SEARCH_LIMIT):
            for version in model.versions:
                named = version.model_copy(update={"model_id": model.id})
                for match in _matches(named, name=base, model=model.name):
                    found.setdefault((match.version_id, match.file), match)
        if found:
            break
    return list(found.values())


def _matches(
    version: CivitaiVersion, sha256: str = "", name: str = "", model: str = ""
) -> list[Match]:
    if version.model_id is None:
        return []
    title = model or (version.model.name if version.model else "") or f"model {version.model_id}"
    return [
        Match(
            model_id=version.model_id,
            version_id=version.id,
            model=title,
            version=version.name,
            file=f.name,
            size=f.size_bytes,
            same_bytes=bool(sha256),
        )
        for f in version.files
        if f.plain_name
        and (
            f.hashes.get("SHA256", "").lower() == sha256
            if sha256
            else f.name.lower() == name.lower()
        )
    ]
