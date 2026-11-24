"""Starting an instance (`InstanceService.start`): download, the start-time upgrades, the
container, `/health`, auto-fix's retries, and the reading each successful start leaves for
the planner (the VRAM it took and how long it loaded)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from makenomistakes import slot_files
from makenomistakes.autofix import MAX_ATTEMPTS, Fix, can_fix, next_fix, shrinker
from makenomistakes.backend_now import recipe_backend_or_none
from makenomistakes.diagnose import diagnose
from makenomistakes.disk_policy import check as disk_check
from makenomistakes.disk_settings import redownload_words, redownloaded_words
from makenomistakes.instance_types import DockerError, InstanceError, Progress, Started
from makenomistakes.measure_window import Window
from makenomistakes.measurements import Kind, first_card_only, reading_for
from makenomistakes.music_speed import model_file
from makenomistakes.reading_stamp import CardBefore, stamp_for, window_before
from makenomistakes.recipes import Recipe, spec_for
from makenomistakes.recipes.cards import placement_note
from makenomistakes.router_client import not_answering_note
from makenomistakes.start_fixes import (
    StartRefusedError,
    retry_resources,
    start_upgrades,
)
from makenomistakes.store import StoreError
from makenomistakes.supervisor import Health, SupervisorError

if TYPE_CHECKING:
    from makenomistakes.instances import InstanceService


def run_start(
    service: InstanceService,
    recipe: Recipe,
    progress: Progress | None = None,
    auto_fix: bool = True,
) -> Started:
    """Download if needed, run the container, wait for /health.

    When the engine dies out of memory, the next recipe is planned from the fit model
    (header on disk, free VRAM now) or, without a header, one safe step smaller; it is
    saved, announced, and the start retried, at most `MAX_ATTEMPTS` times in all.
    Anything else fails with its diagnosis. The saved recipe is re-read before each
    change so a settings edit made meanwhile is kept.
    """
    fetching = redownload_words(recipe)  # The disk policy evicted its files
    fetched_again = redownloaded_words(recipe)
    if fetching is not None and progress:
        progress(fetching[0].upper() + fetching[1:])
    needed = not service.files_present(recipe)  # a download runs the disk policy itself
    recipe = service.get(service.download(recipe, progress).name)  # settings saved meanwhile win
    if not needed:
        disk_check(service, frozenset({recipe.name}))
    fixes: list[str] = []
    try:
        upgrades = start_upgrades(recipe, service.models_dir, service.memory)
    except StartRefusedError as exc:
        raise InstanceError(str(exc)) from exc
    for change, what in upgrades:
        recipe = service.change(recipe.name, change)
        fixes.append(what)
        if progress:
            progress(what)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            notes = _start_once(service, recipe, progress)
        except InstanceError as exc:
            exc.fixes = list(fixes)
            if not auto_fix or attempt == MAX_ATTEMPTS or not can_fix(recipe, exc.diagnoses):
                raise  # the container (exited or still loading) and its log stay as they are
            applied = _retry_fix(service, recipe, exc, fixes)
            recipe = applied.recipe
            fixes.append(applied.what)
            if progress:
                progress(applied.what)
            continue
        # Up: no longer stopped by the user. Only now, so a start that fails leaves the
        # flag as it was and the router never retries it on every request.
        service.mark_off(recipe, False)
        if fetched_again is not None:
            notes.insert(0, fetched_again)
        return Started(recipe.model_copy(update={"off": False}), fixes, notes)
    raise AssertionError("unreachable")  # pragma: no cover


def _retry_fix(
    service: InstanceService, recipe: Recipe, exc: InstanceError, fixes: list[str]
) -> Fix:
    """The fix for the next attempt, saved; `exc` again when none is left. A container
    still running is halted first so its memory is free when the card is read, never
    removed: a start nothing can help keeps its container and log for `mnm logs`,
    and the next `docker run` removes it."""
    try:
        service.supervisor.halt(recipe.name)
    except SupervisorError as halt_exc:
        raise InstanceError(str(halt_exc), exc.diagnoses, fixes) from halt_exc
    correction = service.engine_correction(recipe)  # measured on this card
    resources = retry_resources(recipe, service.models_dir, service.memory(), correction)
    if next_fix(recipe, exc.diagnoses, resources) is None:
        raise exc
    applied: list[Fix] = []
    try:
        service.store.update(recipe.name, shrinker(exc.diagnoses, resources, applied))
    except StoreError as store_exc:
        raise InstanceError(str(store_exc), exc.diagnoses, fixes) from store_exc
    if not applied:  # the recipe changed meanwhile and has nothing left to give
        raise exc
    return applied[0]


def _start_once(service: InstanceService, recipe: Recipe, progress: Progress | None) -> list[str]:
    """One attempt; what else the user should know comes back as notes."""
    if progress:
        progress(f"Starting {recipe.served_name} on port {recipe.port}")
    try:
        if (prepare := spec_for(recipe).prepare) is not None:  # link folders, a config
            prepare(recipe, service.models_dir)
    except (OSError, ValueError) as exc:
        raise InstanceError(f"Could not prepare {recipe.name}'s files: {exc}") from exc
    notes: list[str] = []
    try:
        service.supervisor.ensure_image(recipe.image, progress)  # a first pull takes minutes
    except SupervisorError as exc:
        raise DockerError(
            str(exc), diagnose(str(exc), recipe.engine, recipe_backend_or_none(recipe))
        ) from exc
    try:
        # `mnm rm` during the pull (or between auto-fix attempts) must not leave a
        # container nobody lists, holding the port and VRAM.
        if not service.store.exists(recipe.name):
            raise InstanceError(f"{recipe.name} was removed while starting; nothing runs.")
        window = service.card.open()  # what the card held before this engine ran
        before = window_before(window, own_running=False)  # what else held it
        began = time.monotonic()
        service.supervisor.start(recipe, service.models_dir)
        if not service.store.exists(recipe.name):  # `mnm rm` landed during the run
            service.supervisor.stop(recipe.name)
            raise InstanceError(f"{recipe.name} was removed while starting; nothing runs.")
        if progress:
            progress("Loading model…")
        health = service.supervisor.wait_ready(recipe)
    except SupervisorError as exc:
        raise InstanceError(
            str(exc), diagnose(str(exc), recipe.engine, recipe_backend_or_none(recipe))
        ) from exc
    if health is not Health.READY:
        review = service.review_or_none(recipe)
        raise InstanceError(
            f"{recipe.name} did not become ready in time (last state: {health.value}).",
            review.diagnoses if review else [],
        )
    _record_load(service, recipe, window, before, time.monotonic() - began)
    if (put_back := slot_files.after_start(service, recipe)) is not None:  # `cache=on`
        notes.append(f"{put_back}.")
    if (moved := placement_note(recipe)) is not None:  # Its cards are not here
        notes.append(f"{moved}.")
    if not service.router_up(service.settings):
        notes.append(not_answering_note(service.settings))
    return notes


def _record_load(
    service: InstanceService, recipe: Recipe, window: Window, before: CardBefore, seconds: float
) -> None:
    """The reading a successful start leaves: how long it took to answer ready, what the card
    grew by meanwhile (only while nothing else touched it), and the build the container runs.
    That growth is the instance's figure for an engine that allocates while it loads; one
    that loads its weights at the first request (`resident_at`) is measured then, from it,
    and a music model's task buffers come after ready (three loads could lower ACE-Step's
    figure 10% below its working set), so its figure is a task's peak
    (`request_peak`). Never fails the start it measures."""
    try:
        grew = service.card.grew(recipe.name, window)
        if not first_card_only(service.memory, recipe):  # several cards: unmeasured
            grew = None
        status = service.supervisor.status(recipe.name)
        resident = spec_for(recipe).resident_at == "load" and model_file(recipe) is None
        reading = reading_for(
            recipe,
            Kind.LOAD,
            status.build,
            estimate_bytes=service.raw_estimate(recipe),
            vram_bytes=grew if resident else None,
            ready_bytes=grew,
            container_started=status.started_at,
            load_seconds=round(seconds, 2),
            stamp=stamp_for(service, recipe, before),
        )
        service.measurements.add(recipe.name, reading)
    except Exception:  # a reading is a hint; the start succeeded
        return
