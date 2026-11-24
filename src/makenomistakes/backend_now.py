"""The backend this box runs, as the rest of mnm reads it: decided from the box as
`machine.json` last recorded it (the router's start, the doctor, `mnm ls`, a plan's save) and
`MNM_BACKEND`, read at the call. Before any box was recorded the CUDA 12 images and `--gpus`
stand (a fresh state folder before its first command). An instance's own backend is
its image's (`recipe_backend`): a recipe keeps the image it was planned with until
`mnm replan`/`mnm machine adopt` moves it, and its device arguments follow that image, never
the box alone.

What the ROCm and Vulkan arguments read from the host (`host_devices`: render nodes, group
numbers, the Vulkan loader's order) is read once per process and backend."""

from __future__ import annotations

import threading
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from makenomistakes.backend import (
    Backend,
    BackendError,
    HostDevices,
    apu_only,
    decide,
    not_taken,
    parse_backend,
)
from makenomistakes.engine_images import IMAGES
from makenomistakes.hardware import devices
from makenomistakes.hardware.fingerprint import MachineFingerprint
from makenomistakes.hardware.probe import SubprocessRunner, read_text_or_empty
from makenomistakes.machine_facts import FILE, MachineFactsError, load_file

if TYPE_CHECKING:
    from makenomistakes.config import Settings
    from makenomistakes.recipes.model import Recipe

BEFORE = Backend.CUDA12  # what a box mnm has not read yet runs
ROOT = Path("/")
_cache: dict[Path, tuple[tuple[int, int], MachineFingerprint | None]] = {}
_lock = threading.Lock()


def recorded_box(state_dir: Path) -> MachineFingerprint | None:
    """The box as `machine.json` last recorded it; None before any (or unreadable). Read
    again only when the file changed (`card_ids.live_ids`' rule)."""
    path = state_dir / FILE
    try:
        info = path.stat()
    except OSError:
        return None
    stamp = (info.st_mtime_ns, info.st_size)
    with _lock:
        cached = _cache.get(path)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    try:
        found = load_file(state_dir)
    except (MachineFactsError, OSError):
        found = None
    box = found.current.fingerprint if found is not None and found.current else None
    with _lock:
        _cache[path] = (stamp, box)
    return box


def backend_for(
    box: MachineFingerprint | None,
    chosen: str | None,
    gpu: str | None = None,
    rocm_measured: bool = False,
) -> Backend:
    """`decide` on a box; before any box was read, the user's choice or the CUDA 12 images.
    A choice these cards cannot run is a BackendError naming why."""
    if box is None:
        return parse_backend(chosen or "") or BEFORE
    return decide(box, chosen, gpu, rocm_measured)


def decide_here(settings: Settings, box: MachineFingerprint) -> Backend:
    """`decide` with everything the settings say: `MNM_BACKEND`, `MNM_GPU` and whether a tune
    on this box measured ROCm faster (the APU rule)."""
    return decide(box, settings.backend, settings.gpu, rocm_measured_here(settings, box))


def rocm_measured_here(settings: Settings, box: MachineFingerprint) -> bool:
    """A tune on this box adopted ROCm for an instance (its backend stage): what lets
    an APU's `MNM_BACKEND=rocm` move it off Vulkan. Only an APU asks."""
    if not apu_only(box) or parse_backend(settings.backend or "") is not Backend.ROCM:
        return False
    from makenomistakes.machine import saved_recipes  # noqa: PLC0415 - it imports this module

    return any(
        r.tuned is not None
        and r.tuned.changes.get("backend") == Backend.ROCM.value
        and r.tuned.key.machine in (None, box.id)
        for r in saved_recipes(settings)
    )


def box_backend() -> Backend:
    """This box's backend now (the settings and `machine.json` read at the call)."""
    from makenomistakes.config import settings_now  # noqa: PLC0415 - read at the call

    settings = settings_now()
    box = recorded_box(settings.expanded_state_dir())
    if box is None:
        return backend_for(None, settings.backend)
    return decide_here(settings, box)


def backend_not_taken(box: MachineFingerprint, decided: Backend) -> str | None:
    """`backend.not_taken` with the settings read at the call."""
    from makenomistakes.config import settings_now  # noqa: PLC0415 - read at the call

    settings = settings_now()
    return not_taken(box, settings.backend, settings.gpu, decided)


def box_now() -> MachineFingerprint | None:
    from makenomistakes.config import settings_now  # noqa: PLC0415 - read at the call

    try:
        return recorded_box(settings_now().expanded_state_dir())
    except ValueError:  # settings broken: they say so wherever they are read
        return None


def no_card() -> bool:
    """A box without a card, as last recorded (the CPU backend, or Vulkan on Mesa's
    software device): nothing reads a card's memory there."""
    box = box_now()
    return box is not None and not box.cards


def recipe_backend(recipe: Recipe) -> Backend:
    """The backend the instance's image is built for; an image mnm does not list (named by
    hand) runs on this box's."""
    found = IMAGES[recipe.engine].backend_of(recipe.image)
    return found if found is not None else box_backend()


def recipe_backend_or_none(recipe: Recipe) -> Backend | None:
    """`recipe_backend`, None when `MNM_BACKEND` is refused: a diagnosis never fails on it
    (every backend's rules run then)."""
    try:
        return recipe_backend(recipe)
    except BackendError:
        return None


def host_devices(backend: Backend) -> HostDevices:
    """What `backend.device_argv` reads from the host for ROCm and Vulkan; empty otherwise."""
    if backend not in (Backend.ROCM, Backend.VULKAN):
        return HostDevices()
    box = box_now()
    return _host(backend, box.id if box is not None else None)


@cache
def _host(backend: Backend, _box: str | None) -> HostDevices:
    """Read once per process, backend and box (`_box` keys a re-read after a change)."""
    box = box_now()
    groups = devices.group_ids(read_text_or_empty(ROOT / devices.GROUP_FILE))
    if backend is Backend.ROCM:
        nodes = tuple(c.render_node or "" for c in box.cards) if box is not None else ()
        if not any(nodes):  # a box recorded by an mnm that did not read render nodes
            nodes = devices.render_nodes(ROOT)
        return HostDevices(render_nodes=nodes, group_ids=groups)
    found = devices.vulkan_devices(SubprocessRunner())
    software = box is not None and not box.cards
    if found is None:
        return HostDevices(group_ids=groups, software=software)
    return HostDevices(
        group_ids=groups,
        vulkan_indices=devices.loader_indices(box.cards, found) if box is not None else None,
        software=software,
        software_index=devices.software_index(found),
    )


def runs_in_ram(settings: Settings | None = None) -> bool:
    """A new plan on this box is made in RAM: the box's backend is the processor's, a
    box with no card or `MNM_BACKEND=cpu` (`plan_budget.planning_budget`; an instance saved
    already sits where its own image runs, `ram_room.runs_on_processor`). Never raises: a
    backend these cards refuse is none."""
    from makenomistakes.config import settings_now  # noqa: PLC0415 - read at the call

    settings = settings or settings_now()
    box = recorded_box(settings.expanded_state_dir())
    try:
        found = backend_for(None, settings.backend) if box is None else decide_here(settings, box)
    except BackendError:
        return False
    return found is Backend.CPU


__all__ = [
    "BEFORE",
    "BackendError",
    "backend_for",
    "box_backend",
    "no_card",
    "recipe_backend",
    "recipe_backend_or_none",
    "runs_in_ram",
]
