"""Engine images: what is on this box, explicit updates, and a rollback to the last build.

Engines are official images on named tags (`server-cuda`, `master-cuda`, `main-cuda`). Docker
never moves a pulled tag by itself, so a box keeps the build it first pulled until the user
runs `mnm engines update`. That pull remembers the image it replaced, and `mnm engines
rollback <engine>` re-tags it, so a new build that breaks a model is one command away from
undone. Containers keep running on the image they started with; a restart picks up the new
one. Docker itself is reached only through `DockerSupervisor`; this module keeps the ledger
and the policy.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from makenomistakes.backend import Backend, label
from makenomistakes.backend_now import backend_not_taken, box_backend, box_now
from makenomistakes.engine_images import ENGINE_NAMES, IMAGES
from makenomistakes.hub.facts import Engine
from makenomistakes.locking import shared_lock
from makenomistakes.recipes import SPECS, Recipe
from makenomistakes.supervisor import DockerSupervisor, LocalImage, SupervisorError

# The default tags (CUDA 12 on amd64): what a box mnm has not read uses.
ENGINE_IMAGES: dict[Engine, str] = {engine: row.image for engine, row in SPECS.items()}


def box_images(backend: Backend | None = None, arch: str | None = None) -> dict[Engine, str]:
    """Each engine's tag for this box's backend and arch; an engine with no image
    here is left out. Before any box was read: `ENGINE_IMAGES`."""
    box = box_now()
    if backend is None and box is None:
        return dict(ENGINE_IMAGES)
    backend = backend or box_backend()
    arch = arch or (box.arch if box is not None else "x86_64")
    found = {engine: row.image_for(backend, arch) for engine, row in SPECS.items()}
    return {engine: tag for engine, tag in found.items() if isinstance(tag, str)}


class EngineError(RuntimeError):
    pass


class EngineRecord(BaseModel):
    """What `mnm engines update` (or a rollback) replaced, so rollback can bring it back."""

    previous_id: str
    updated_at: datetime


class EngineLedger(BaseModel):
    records: dict[str, EngineRecord] = Field(default_factory=dict)  # keyed by image reference


def ledger_path(state_dir: Path) -> Path:
    return state_dir / "engines.json"


def load_ledger(state_dir: Path) -> EngineLedger:
    path = ledger_path(state_dir)
    try:
        return EngineLedger.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return EngineLedger()
    except ValidationError as exc:
        raise EngineError(f"{path} cannot be read ({exc.errors()[0]['msg']}); delete it.") from exc


@contextmanager
def _locked(state_dir: Path) -> Iterator[None]:
    """One `mnm` process changes the ledger at a time (a long update and a rollback in
    another terminal must not undo each other); re-entrant within a process and across its
    threads (`locking.shared_lock`, the same `.engines.lock` file as before)."""
    with shared_lock(state_dir, ".engines.lock")():
        yield


ledger_locked = _locked  # `mnm restore` writes a missing ledger under it


def _write_ledger(state_dir: Path, ledger: EngineLedger) -> None:
    """Atomic replace; callers hold `_locked`."""
    path = ledger_path(state_dir)
    tmp = path.with_name(f".engines.{os.getpid()}.tmp")
    tmp.write_text(ledger.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)


def _record(
    state_dir: Path,
    image: str,
    previous_id: str,
    now: datetime | None,
    keep_existing: bool = False,
) -> None:
    """Read, change, write under the lock: never from a copy loaded before a long pull.
    `keep_existing`: the new build is unknown, so a record already there (the way back to an
    older build) stays as it is."""
    with _locked(state_dir):
        ledger = load_ledger(state_dir)
        if keep_existing and image in ledger.records:
            return
        ledger.records[image] = EngineRecord(
            previous_id=previous_id, updated_at=now or datetime.now(UTC)
        )
        _write_ledger(state_dir, ledger)


@dataclass(frozen=True)
class EngineStatus:
    engine: Engine
    image: str
    local: LocalImage | None
    instances: list[str] = field(default_factory=list)  # recipes that run this image
    can_roll_back: bool = False

    @property
    def backend(self) -> str:
        """The backend the tag is built for, as words (`CUDA 12`, `Vulkan`, `CPU`); `custom`
        for an image mnm does not list."""
        found = IMAGES[self.engine].backend_of(self.image)
        return label(found) if found is not None else "custom"


def engine_status(
    docker: DockerSupervisor, recipes: list[Recipe], ledger: EngineLedger
) -> list[EngineStatus]:
    """One row per engine image: this box's backend's tag per engine plus any a
    recipe names (an image for the box's old backend, kept until `mnm replan` moves it)."""
    images: dict[str, Engine] = {image: engine for engine, image in box_images().items()}
    for recipe in recipes:
        images.setdefault(recipe.image, recipe.engine)
    return [
        EngineStatus(
            engine=engine,
            image=image,
            local=docker.image(image),
            instances=sorted(r.name for r in recipes if r.image == image),
            can_roll_back=image in ledger.records,
        )
        for image, engine in images.items()
    ]


def box_words() -> str:
    """`This box runs the CPU images (no card).`: the line `mnm engines` and System → Engines
    print over the rows; the default CUDA images before any box was read."""
    box = box_now()
    if box is None:
        return "This box is not read yet, so the CUDA images are assumed (`mnm doctor` reads it)."
    try:
        backend = box_backend()
    except ValueError as exc:  # MNM_BACKEND refused
        return f"{exc}."
    said = f"This box runs the {label(backend)} images"
    if backend is Backend.CPU and not box.cards:
        said += " (no card)"
    elif backend is Backend.VULKAN and not box.cards:
        said += " on Mesa's software device (slow)"
    if (why := backend_not_taken(box, backend)) is not None:
        said += f" ({why})"
    none = [ENGINE_NAMES[e] for e in SPECS if e not in box_images(backend, box.arch)]
    return f"{said}; none here for {', '.join(none)}." if none else f"{said}."


def images_to_update(rows: list[EngineStatus], wanted: set[Engine]) -> list[str]:
    """The named engines' images; with none named, every image this box has or an instance
    uses (an engine nobody uses is never fetched: each is a few GB)."""
    if wanted:
        return [r.image for r in rows if r.engine in wanted]
    return [r.image for r in rows if r.local is not None or r.instances]


def image_for(target: str) -> str:
    """`llamacpp` -> its official image for this box's backend; anything else is taken as an
    image reference."""
    if target not in set(Engine):
        return target
    engine = Engine(target)
    return box_images().get(engine, ENGINE_IMAGES[engine])


class Change(StrEnum):
    UPDATED = "updated"  # a newer build replaced the local one
    CURRENT = "current"  # the registry has nothing newer
    PULLED = "pulled"  # was not on the box; now it is
    FAILED = "failed"


@dataclass(frozen=True)
class UpdateResult:
    image: str
    change: Change
    before: LocalImage | None = None
    after: LocalImage | None = None
    detail: str = ""


def update(
    docker: DockerSupervisor,
    images: list[str],
    state_dir: Path,
    progress: Callable[[str], None] | None = None,
    now: datetime | None = None,
) -> list[UpdateResult]:
    """`docker pull` each image; remember what it replaced. Never removes the old image. A
    Docker error on one image is that image's FAILED result, never the end of the run."""
    results: list[UpdateResult] = []
    for image in images:
        try:
            results.append(_update_one(docker, image, state_dir, progress, now))
        except SupervisorError as exc:  # could not even inspect it
            results.append(UpdateResult(image, Change.FAILED, detail=str(exc)))
    return results


def _update_one(
    docker: DockerSupervisor,
    image: str,
    state_dir: Path,
    progress: Callable[[str], None] | None,
    now: datetime | None,
) -> UpdateResult:
    before = docker.image(image)
    if progress:
        progress(f"docker pull {image}")
    try:
        docker.pull(image)
    except SupervisorError as exc:
        return UpdateResult(image, Change.FAILED, before, detail=str(exc))
    try:
        after = docker.image(image)
    except SupervisorError as exc:  # pulled, but Docker cannot say what: keep the way back
        if before is not None:  # re-tagging the same id is harmless; an older record wins
            _record(state_dir, image, before.id, now, keep_existing=True)
        return UpdateResult(image, Change.FAILED, before, detail=str(exc))
    if before is None:
        return UpdateResult(image, Change.PULLED, None, after)
    if after is not None and after.id != before.id:
        _record(state_dir, image, before.id, now)
        return UpdateResult(image, Change.UPDATED, before, after)
    return UpdateResult(image, Change.CURRENT, before, after)


@dataclass(frozen=True)
class Rollback:
    image: str
    restored: LocalImage  # the build the tag points at after the rollback
    replaced: LocalImage | None  # what it pointed at before; None when the tag was missing


def rollback(docker: DockerSupervisor, image: str, state_dir: Path) -> Rollback:
    """Point the tag back at the build `update` replaced; the two swap places in the ledger
    (a second rollback goes forward again)."""
    with _locked(state_dir):
        ledger = load_ledger(state_dir)
        record = ledger.records.get(image)
        if record is None:
            raise EngineError(f"No earlier build of {image} is recorded; nothing to roll back to.")
        previous = docker.image(record.previous_id)
        if previous is None:
            raise EngineError(
                f"The earlier build of {image} ({record.previous_id[:19]}) is no longer on this "
                "box (`docker image prune` removes it). Nothing to roll back to."
            )
        current = docker.image(image)
        docker.tag(previous.id, image)
        if current is None:  # the tag was missing: nothing to go forward to
            del ledger.records[image]
        else:
            ledger.records[image] = EngineRecord(
                previous_id=current.id, updated_at=datetime.now(UTC)
            )
        _write_ledger(state_dir, ledger)
    return Rollback(image=image, restored=previous, replaced=current)
