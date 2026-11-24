"""An instance under another name (`mnm upgrade` keeps the old model as `<name>-old`):
its recipe, its readings, its tools verdict, its tuning trials and its kept log move with it,
and its engine's links (sd.cpp's `views/`, `loras/`, `upscalers/`; audio.cpp's config) are
laid out again under the new name. Its container must be gone first (the caller parks it):
a container is named after its instance.

Order: the new recipe is written first and the old one deleted last, so a failure in between
leaves both names (`mnm rm` one), never neither. The tuner's queue entry and state go (a
queued tune of the old name is not this model's any more); the recipe's `tuned` block stays."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.json_folder import BadNameError
from makenomistakes.last_logs import log_path
from makenomistakes.recipes import spec_for
from makenomistakes.recipes.llamacpp_cache import cache_folder
from makenomistakes.recipes.model import Recipe, validate_instance_name
from makenomistakes.store import StoreError
from makenomistakes.supervisor import ContainerState, SupervisorError
from makenomistakes.tune.store import QualityQueued, TuneStore, TuneStoreError


def rename(service: InstanceService, recipe: Recipe, new_name: str) -> Recipe:
    """`recipe` saved as `new_name` and nothing else changed, with what is kept under its
    name; the old name is free afterwards. InstanceError when the new name is taken or
    invalid, or the container still exists."""
    try:
        validate_instance_name(new_name)
    except ValueError as exc:
        raise InstanceError(str(exc)) from exc
    if service.store.exists(new_name):
        raise InstanceError(f"An instance named '{new_name}' already exists.")
    try:
        state = service.supervisor.status(recipe.name).state
    except SupervisorError as exc:
        raise InstanceError(f"{recipe.name} was kept as it is: {exc}") from exc
    if state is not ContainerState.ABSENT:
        raise InstanceError(f"{recipe.name}'s container is still there: stop it first.")
    renamed = recipe.model_copy(update={"name": new_name})
    try:
        service.store.save(renamed)
    except StoreError as exc:
        raise InstanceError(str(exc)) from exc
    _move_state(service, recipe.name, new_name)
    _relink(service, recipe, renamed)
    try:
        service.store.delete(recipe.name)
    except StoreError as exc:
        raise InstanceError(
            f"{recipe.name} is saved as {new_name} too; its old recipe could not be removed "
            f"({exc}): `mnm rm {recipe.name}` (its downloads stay)."
        ) from exc
    return renamed


def _move(source: Path, target: Path) -> None:
    with contextlib.suppress(OSError):
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)


def _move_state(service: InstanceService, old: str, new: str) -> None:
    """Readings, the tools verdict, the tuning trials and the kept log: best effort each (a
    file that cannot move is left under the old name; nothing reads it there)."""
    state = service.settings.expanded_state_dir()
    measurements = service.measurements
    _move(measurements.path(old), measurements.path(new))
    with contextlib.suppress(OSError, BadNameError):
        _move(service.tools.files.path(old), service.tools.files.path(new))
    tunes = TuneStore(state)
    with contextlib.suppress(OSError, BadNameError):
        _move(tunes.path(old), tunes.path(new))
    with contextlib.suppress(OSError, BadNameError, TuneStoreError):
        if not isinstance(tunes.queued(old), QualityQueued):  # Its downloads need
            tunes.unqueue(old)  # the router's end of it (`TuneRunner._end_orphans`)
            tunes.drop_state(old)
    _move(log_path(state, old), log_path(state, new))
    models = service.models_dir  # Its saved prompts (keyed by settings, not the name)
    _move(cache_folder(models, old), cache_folder(models, new))


def _relink(service: InstanceService, old: Recipe, renamed: Recipe) -> None:
    """The engine's per-instance layout under the new name (the next start does it too)."""
    row = spec_for(old)
    with contextlib.suppress(OSError, KeyError, ValueError):
        if row.cleanup is not None:
            row.cleanup(old, service.models_dir)
        if row.prepare is not None and renamed.downloaded:
            row.prepare(renamed, service.models_dir)
