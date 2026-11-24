"""Imported workflows: one folder each under `<state>/comfy/workflows/<name>/`, the
file as fetched (`workflow.json`) beside what mnm knows about it (`meta.json`). Importing again
under the same name keeps the version before it in `versions/` (`mnm clean workflows` empties
those)."""

from __future__ import annotations

import hashlib
import re
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator

from makenomistakes.comfy.ui_format import FormatError, Workflow, decode, read_workflow
from makenomistakes.links import without_secrets
from makenomistakes.local_time import local
from makenomistakes.locking import shared_lock

SCHEMA_VERSION = 1
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_SLUG = re.compile(r"[^a-z0-9]+")


class WorkflowStoreError(ValueError):
    """A workflow store problem worded for the user."""


class Suggestion(BaseModel):
    """A file found on Civitai by its name only: offered, used once the user confirms."""

    url: str
    label: str
    size: int = 0


class WorkflowMeta(BaseModel):
    schema_version: int = SCHEMA_VERSION
    name: str
    title: str = ""
    source: str  # the link or file it came from
    sha256: str
    imported_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    shown: list[str] = Field(default_factory=list)  # the settings the user picked to show
    links: dict[str, str] = Field(default_factory=dict)  # model file name -> the user's link
    found: dict[str, list[Suggestion]] = Field(default_factory=dict)  # name -> to confirm

    def link(self, file: str, url: str) -> None:
        """Where a model file comes from: it replaces any offer for that file."""
        self.links[file] = without_secrets(url)
        self.found.pop(file, None)

    @field_validator("source")
    @classmethod
    def _no_secret_source(cls, value: str) -> str:
        """The link it came from, never with a token in its query."""
        return without_secrets(value)

    @field_validator("links")
    @classmethod
    def _no_secret_links(cls, value: dict[str, str]) -> dict[str, str]:
        return {file: without_secrets(url) for file, url in value.items()}


def slug(text: str) -> str:
    """A workflow name from a file's or a link's name: lowercase, dashes, at most 63."""
    found = _SLUG.sub("-", text.lower()).strip("-")[:63].strip("-")
    return found or "workflow"


RESERVED = frozenset({"packs", "import"})  # the Workflows tab's own pages


class WorkflowStore:
    def __init__(self, state_dir: Path) -> None:
        self.root = state_dir / "comfy" / "workflows"
        self._lock = shared_lock(self.root)  # one per process: every store object takes turns

    def update(self, name: str, change: Callable[[WorkflowMeta], None]) -> WorkflowMeta:
        """Change a workflow's meta in place: read, changed and saved under one lock, so a
        long job (Find) and a click (Use, the picked settings) never undo each other."""
        with self._lock():
            meta = self.meta(name)
            change(meta)
            self.save(meta)
            return meta

    def folder(self, name: str) -> Path:
        if not _NAME.match(name) or name in RESERVED:
            raise WorkflowStoreError(
                f"'{name}' is not a workflow name: lowercase letters, digits and dashes."
            )
        return self.root / name

    def free_name(self, wanted: str) -> str:
        """`wanted`, or `wanted-2`, `wanted-3` … when it is taken."""
        base = slug(wanted)
        name, number = (base, 2) if base not in RESERVED else (f"{base}-2", 3)
        while self.folder(name).exists():
            name = f"{base[:58]}-{number}"
            number += 1
        return name

    def add(self, name: str, data: bytes, source: str, title: str = "") -> WorkflowMeta:
        """Keep a workflow (it must read as one); the version it replaces goes to versions/."""
        with self._lock():  # a Find job finishing meanwhile never writes the old meta back
            try:
                read_workflow(decode(data))
            except FormatError as exc:
                why = (
                    "it is an API-format export; export it with Workflow → Export (not API)"
                    if "API-format" in str(exc)
                    else str(exc)
                )
                raise WorkflowStoreError(f"{source} is not a ComfyUI workflow: {why}.") from exc
            folder = self.folder(name)
            if (folder / "workflow.json").exists():
                versions = folder / "versions"
                versions.mkdir(parents=True, exist_ok=True)
                stamp = f"{local():%Y%m%d-%H%M%S-%f}"  # two imports a second keep both
                shutil.copy2(folder / "workflow.json", versions / f"{stamp}.json")
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "workflow.json").write_bytes(data)
            meta = WorkflowMeta(
                name=name,
                title=title or name,
                source=source,
                sha256=hashlib.sha256(data).hexdigest(),
            )
            self.save(meta)
            return meta

    def save(self, meta: WorkflowMeta) -> None:
        path = self.folder(meta.name) / "meta.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)

    def meta(self, name: str) -> WorkflowMeta:
        path = self.folder(name) / "meta.json"
        try:
            meta = WorkflowMeta.model_validate_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise WorkflowStoreError(
                f"No workflow named '{name}' (`mnm comfyui workflow ls` lists them)."
            ) from None
        except OSError as exc:
            raise WorkflowStoreError(f"{path} cannot be read: {exc}") from exc
        except ValidationError as exc:
            raise WorkflowStoreError(f"{path} cannot be read: {exc.errors()[0]['msg']}") from exc
        if meta.schema_version > SCHEMA_VERSION:
            raise WorkflowStoreError(f"{path} is from a newer mnm: run `mnm update`.")
        return meta

    def workflow(self, name: str) -> Workflow:
        try:
            data = (self.folder(name) / "workflow.json").read_bytes()
        except FileNotFoundError:  # worded: never the state folder's path
            raise WorkflowStoreError(
                f"There is no workflow named {name} (`mnm comfyui workflow ls` lists them)."
            ) from None
        return read_workflow(decode(data))

    def names(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / "meta.json").is_file())

    def all(self) -> list[WorkflowMeta]:
        """The workflows this version can read; `problems` names the others."""
        found = []
        for name in self.names():
            try:
                found.append(self.meta(name))
            except WorkflowStoreError:
                continue
        return found

    def problems(self) -> list[str]:
        """Workflow folders whose record this version cannot read: listed, never left out
        silently (cleanup leaves them alone too)."""
        bad = []
        for name in self.names():
            try:
                self.meta(name)
            except WorkflowStoreError as exc:
                bad.append(f"{name}: {exc}")
        return bad

    def remove(self, name: str) -> None:
        folder = self.folder(name)
        with self._lock():
            if not folder.is_dir():
                raise WorkflowStoreError(f"No workflow named '{name}'.")
            shutil.rmtree(folder)

    def versions(self) -> list[Path]:
        """Every kept earlier version, of every workflow (`mnm clean workflows`)."""
        if not self.root.is_dir():
            return []
        return sorted(self.root.glob("*/versions/*.json"))
