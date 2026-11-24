"""`mnm disk`: what the models folder holds by instance, by source folder and by nothing
(the unlisted files), with the outputs', the state's, ComfyUI's, the apps' and the backups'
sizes and the disk policy's state, as one typed `DiskReport` the CLI, `--json` and the web's
System → Disk all read. Reading only: nothing here deletes or downloads."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from makenomistakes.comfy.models import LinkedModel, ModelError, Models
from makenomistakes.converted_models import converted_files
from makenomistakes.disk_last_used import last_use
from makenomistakes.disk_policy import DiskPolicy
from makenomistakes.disk_settings import DiskSettingsError, Usage, measure
from makenomistakes.disk_tree import folder_size, listed_targets, models_folders, unlisted_files
from makenomistakes.disk_types import (
    DiskReport,
    FolderSize,
    InstanceDisk,
    Keep,
    Level,
    PolicyState,
    SharedFile,
)
from makenomistakes.local_time import local
from makenomistakes.model_cleanup import file_behind
from makenomistakes.recipes import Recipe
from makenomistakes.slot_folder import SlotFolder
from makenomistakes.studio import style_weights
from makenomistakes.supervisor import ContainerStatus, SupervisorError

if TYPE_CHECKING:
    from makenomistakes.instances import InstanceService

NO_OWNER = "ComfyUI files no workflow names"
"""The row for linked files no workflow claims: a row label of its own, never after "workflow "
(which would read "workflow a ComfyUI link no workflow names")."""


def workflow_label(owner: str | None) -> str:
    """A workflow's row (and a shared file's user): "workflow wan5b", or `NO_OWNER`."""
    return f"workflow {owner}" if owner else NO_OWNER


def disk_report(
    service: InstanceService,
    usage: Usage | None = None,
    backups_dir: Path | None = None,  # `MNM_BACKUPS_DIR` (the automatic backups)
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    policy: DiskPolicy | None = None,
) -> DiskReport:
    """The whole report; InstanceError when the saved instances cannot be read."""
    settings = service.settings
    models = service.models_dir
    notes: list[str] = []
    recipes = service.all()
    links = _workflow_links(models, settings.expanded_state_dir(), notes)
    listed = _listed(recipes, links, models)
    instances = [_instance(service, r, listed, now(), notes) for r in recipes]
    sources, other = models_folders(models)
    try:
        total, free = (usage or measure())(models)
    except OSError as exc:
        total = free = 0
        notes.append(f"The models folder's disk cannot be read: {exc}.")
    try:
        state = (policy or DiskPolicy(service, usage=usage)).state()
    except DiskSettingsError as exc:
        notes.append(f"The disk policy is paused: {exc}.")
        state = PolicyState(keep=Keep(), gauge=None, level=Level.OK)
    state_dir = settings.expanded_state_dir()
    folders = [
        folder_size("outputs (MNM_OUTPUTS_DIR)", settings.expanded_outputs_dir()),
        folder_size("state (MNM_STATE_DIR)", state_dir),
        folder_size("of which ComfyUI's folders", state_dir / "comfyui"),
        folder_size("apps (MNM_APPS_DIR)", settings.expanded_apps_dir()),
        folder_size("backups (MNM_BACKUPS_DIR)", backups_dir or settings.expanded_backups_dir()),
    ]
    return DiskReport(
        models_dir=str(models),
        disk_total_bytes=total,
        disk_free_bytes=free,
        held_bytes=sum(f.size_bytes for f in [*sources, *other]),
        instances=instances,
        sources=sources,
        workflows=_by_workflow(links, models),
        unlisted=unlisted_files(models, listed),
        other=other,
        folders=[f for f in folders if f.exists or f.label.startswith("backups")],
        policy=state,
        notes=notes,
    )


def instance_rows(
    service: InstanceService, now: Callable[[], datetime] = lambda: datetime.now(UTC)
) -> tuple[list[InstanceDisk], list[str]]:
    """`mnm ls --disk`: each instance's files and last use, without the folder walk."""
    notes: list[str] = []
    recipes = service.all()
    links = _workflow_links(service.models_dir, service.settings.expanded_state_dir(), notes)
    listed = _listed(recipes, links, service.models_dir)
    return [_instance(service, r, listed, now(), notes) for r in recipes], notes


def _listed(recipes: list[Recipe], links: list[LinkedModel], models: Path) -> dict[Path, list[str]]:
    linked = [(item.target, workflow_label(o)) for item in links for o in _owners(item)]
    return listed_targets(recipes, [*linked, *style_weights.listed(models)], models)


def _owners(item: LinkedModel) -> list[str | None]:
    """A link's workflows, or None for one no workflow names."""
    return [*item.owners] or [None]


def _workflow_links(models: Path, state_dir: Path, notes: list[str]) -> list[LinkedModel]:
    try:
        return Models(models, state_dir).ledger().links
    except ModelError as exc:
        notes.append(f"Which files ComfyUI workflows use is unknown: {exc}.")
        return []


def _instance(
    service: InstanceService,
    recipe: Recipe,
    listed: dict[Path, list[str]],
    now: datetime,
    notes: list[str],
) -> InstanceDisk:
    models = service.models_dir
    own = files = 0
    shared: list[SharedFile] = []
    seen: set[Path] = set()
    held = [(rel, file_behind(models / rel)) for rel in recipe.local_paths.values()]
    made = [(str(f.relative_to(models)), f.resolve()) for f in converted_files(recipe, models)]
    for rel, target in [*held, *made]:  # A converted folder is the instance's too
        if target is None or target in seen:
            continue
        seen.add(target)
        files += 1
        size = _size(target)
        users = [u for u in listed.get(target, []) if u != recipe.name]
        if users:
            shared.append(SharedFile(path=rel, size_bytes=size, users=users))
        else:
            own += size
    container: ContainerStatus | None
    try:
        container = service.supervisor.status(recipe.name)
    except SupervisorError as exc:
        container = None
        if not any(n.startswith("Docker") for n in notes):
            notes.append(f"Docker could not be asked ({exc}): last used leaves out what runs.")
    used = last_use(
        recipe,
        service.measurements.readings(recipe.name),
        service.store.path(recipe.name),
        container,
        now,
    )
    evicted = recipe.evicted.at if recipe.evicted is not None else None
    return InstanceDisk(
        name=recipe.name,
        own_bytes=own,
        shared=shared,
        files=files,
        state=_state(recipe, service.files_present(recipe)),
        last_used=used.at,
        last_used_from=used.source,
        evicted=evicted if not recipe.local_paths else None,
        prompt_bytes=SlotFolder.of(models, recipe.name).held_bytes(),
    )


def _state(recipe: Recipe, present: bool) -> str:
    if present:
        return "on disk"
    if recipe.evicted is not None and not recipe.local_paths:
        return f"evicted {local(recipe.evicted.at):%Y-%m-%d}"
    return "partly on disk" if recipe.local_paths else "not downloaded"


def _by_workflow(links: list[LinkedModel], models: Path) -> list[FolderSize]:
    """Each workflow's linked files (they sit in hf/ or civitai/, counted there too)."""
    found: dict[str, tuple[int, int, set[Path]]] = {}
    for item in links:
        target = file_behind(models / item.target)
        if target is None:
            continue
        for owner in map(workflow_label, _owners(item)):
            size, files, seen = found.get(owner, (0, 0, set()))
            if target not in seen:
                seen.add(target)
                size, files = size + _size(target), files + 1
            found[owner] = (size, files, seen)
    return [
        FolderSize(label=owner, path="comfyui/", exists=True, size_bytes=size, files=files)
        for owner, (size, files, _seen) in sorted(found.items())
    ]


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0
