"""The card a new instance is planned for: the VRAM free now plus what mnm's own running
instances hold (else `coder`, added while qwen3-8b ran, is planned against the 5 GiB left
and saved with 18 of 36 layers on the GPU, for good). Each instance is started on its
own or in a workflow, whose fit-together check counts the others; the plan says whose VRAM it
counted, and `mnm start` refuses or warns when they still run."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from makenomistakes.backend import BackendError
from makenomistakes.backend_now import decide_here, runs_in_ram
from makenomistakes.capacity import COMFYUI, Verdict, VramEstimate
from makenomistakes.fit import Budget
from makenomistakes.instance_types import UnreadableStateError
from makenomistakes.machine import observe_snapshot
from makenomistakes.plan import budget_for, budget_on, live_budget, on_backend, shared_budget
from makenomistakes.ram_budget import in_ram_budget
from makenomistakes.ram_room import ram_margin, runs_on_processor, with_ram

if TYPE_CHECKING:
    from makenomistakes.config import Settings
    from makenomistakes.hardware.probe import MemorySnapshot
    from makenomistakes.instance_types import InstanceStatus
    from makenomistakes.instances import InstanceService


def comfy_held(settings: Settings) -> int:
    """What a running ComfyUI gives back when a start frees its card first: PyTorch's pool,
    and its dynamic-VRAM buffers once a run has gone (`SystemStats.freeable`; never its CUDA
    context); 0 when it is not installed or not running. The pool alone (0.1 GiB of the
    2.3 GiB it held after a film) would leave `mnm sets start` and `mnm ls` short of room
    every start makes; the same figure counts everywhere room is
    judged (`room_verdicts.comfy_room`)."""
    from makenomistakes.room_verdicts import comfy_room  # noqa: PLC0415 - no cycle

    return comfy_room(settings).freeable


def planning_budget(
    service: InstanceService,
    snapshot: MemorySnapshot | None = None,
    statuses: list[InstanceStatus] | None = None,
    comfy: Callable[[], int] | None = None,
) -> Budget | None:
    """None when the GPU or the RAM cannot be read (the plan then probes and says why). A
    page that already read the card and the instances passes them, so nothing is asked
    twice. ComfyUI's freeable VRAM counts as free too: any start frees it first. The
    budget names the box's backend and arch (the box read and recorded as `live_budget`
    does), so a plan made on a card writes the image for it; a `MNM_BACKEND` the cards cannot
    run leaves it to `live_budget`, which says why. A plan the box's backend runs on the
    processor is made in the RAM (`ram_room`), priced as `live_budget` prices it, with what
    mnm's instances hold there counted free; a card's plan counts only what sits on cards."""
    room = with_ram(snapshot or service.memory(), ram_margin(service.settings))
    observed = observe_snapshot(service.settings, room)
    box = observed.file.current.fingerprint if observed.file and observed.file.current else None
    if runs_in_ram(service.settings):
        if box is None or room.ram is None or room.memory is None:
            return None
        base = in_ram_budget(service.settings, box, room.memory)
        held = holders(service, statuses, in_ram=True)
        return _beside_holders(room, base, held, room.ram.vram_used_bytes)
    budget = _planning_budget(service, room, statuses, comfy)
    if budget is None or box is None:
        return budget
    try:
        backend = decide_here(service.settings, box)
    except BackendError:
        return None
    return on_backend(budget, backend).model_copy(update={"backend": backend, "arch": box.arch})


def _planning_budget(
    service: InstanceService,
    snapshot: MemorySnapshot,
    statuses: list[InstanceStatus] | None,
    comfy: Callable[[], int] | None,
) -> Budget | None:
    gpu, ram = snapshot.gpu, snapshot.memory.available_bytes if snapshot.memory else None
    if gpu is None or ram is None:
        return None
    total = snapshot.memory.total_bytes if snapshot.memory else None
    held = holders(service, statuses, in_ram=False)
    comfy_bytes = (comfy or (lambda: comfy_held(service.settings)))()
    if comfy_bytes:
        held[0][COMFYUI] = comfy_bytes
    if (gpu.unified or gpu.carve_out) and snapshot.memory is not None:  # One pool
        budget = shared_budget(gpu, snapshot.memory)
        return _beside_holders(snapshot, budget, held, budget.vram_used_bytes)
    budget = budget_for(gpu, ram).model_copy(update={"ram_total_bytes": total})  # the floor
    return _beside_holders(snapshot, budget, held, gpu.vram_used_bytes)


def _beside_holders(
    snapshot: MemorySnapshot,
    budget: Budget,
    found: tuple[dict[str, int], dict[str, VramEstimate]],
    in_use: int | None,
) -> Budget:
    """`budget` with what mnm's running instances hold in its place counted free (a start
    parks them first) and named, never more than is `in_use`: on one card, one pool or the
    RAM alike; several cards each card's own (`across_cards`)."""
    held, placed = found
    memory = snapshot.memory
    several = not budget.no_card and len(snapshot.cards) > 1
    if several and memory is not None and memory.available_bytes:
        return across_cards(snapshot, memory.available_bytes, placed, held).model_copy(
            update={"ram_total_bytes": memory.total_bytes}
        )
    freed = min(sum(held.values()), in_use or 0)  # an estimate never frees more than is used
    if not freed:
        return budget
    return budget.model_copy(
        update={
            "vram_free_bytes": budget.vram_free_bytes + freed,
            "held_by": tuple(sorted(held)),
            "held_bytes": freed,
        }
    )


def holders(
    service: InstanceService, statuses: list[InstanceStatus] | None, in_ram: bool
) -> tuple[dict[str, int], dict[str, VramEstimate]]:
    """mnm's running instances in one place (the RAM's, `in_ram`, or the cards') and what each
    holds (`held_estimate`): by name, and each one's estimate with where it sits."""
    held: dict[str, int] = {}
    placed: dict[str, VramEstimate] = {}
    for status in statuses if statuses is not None else service.statuses():
        if status.running and runs_on_processor(status.recipe) == in_ram:
            found = service.held_estimate(status)
            if found.bytes:
                held[status.recipe.name] = found.bytes
                placed[status.recipe.name] = found
    return held, placed


def across_cards(
    snapshot: MemorySnapshot, ram: int, placed: dict[str, VramEstimate], held: dict[str, int]
) -> Budget:
    """The chosen cards' budget with what mnm's running instances hold on each card
    counted free there (ComfyUI's on the first card, where it runs)."""
    cards = snapshot.cards
    on: dict[int, int] = {}
    names: dict[int, set[str]] = {}
    for name, estimate in placed.items():
        for index, size in estimate.on or ((cards[0].index, estimate.bytes or 0),):
            on[index] = on.get(index, 0) + size
            names.setdefault(index, set()).add(name)
    if COMFYUI in held:
        on[cards[0].index] = on.get(cards[0].index, 0) + held[COMFYUI]
        names.setdefault(cards[0].index, set()).add(COMFYUI)
    freed = [
        card.model_copy(
            update={
                "vram_used_bytes": card.vram_used_bytes
                - min(on.get(card.index, 0), card.vram_used_bytes)
            }
        )
        for card in cards
    ]
    budget = budget_on(freed, ram)
    if not held:
        return budget
    each = tuple(  # what a plan on one card names (`Budget.one_card`)
        c.model_copy(
            update={
                "held_by": tuple(sorted(names.get(c.index, ()))),
                "held_bytes": before.vram_used_bytes - after.vram_used_bytes,
            }
        )
        for c, before, after in zip(budget.cards, cards, freed, strict=True)
    )
    return budget.model_copy(
        update={
            "cards": each,
            "held_by": tuple(sorted(held)),
            "held_bytes": sum(c.held_bytes for c in each),
        }
    )


def held_by_mnm(
    service: InstanceService, name: str, kept: frozenset[str] | None = None
) -> Verdict | None:
    """The fit-now verdict for `name` when mnm's own running instances hold the room it
    needs (it names them), else None. VRAM held by anything else (Windows apps) is the
    start's own business. `kept` (the router's) are never named as room, only as holders
    (`Verdict.kept`) when their VRAM is needed; a start by hand passes none. Unreadable
    state (a recipe file that does not load) counts as nothing held: the start judges, and
    its own file's problem surfaces there. Any other error goes to the caller."""
    try:
        statuses, snapshot = service.statuses(), service.memory()
        found = (
            service.verdicts(statuses, snapshot)  # a start by hand: every holder is room
            if kept is None
            else service.verdicts(statuses, snapshot, kept=kept)
        )
        verdict = found.get(name)
    except UnreadableStateError:
        return None
    if verdict is None or not verdict.warns or not (verdict.stop_first or verdict.kept):
        return None
    return verdict


def budget_now(service: InstanceService) -> Budget:
    """The budget `mnm add` plans against, read once: this one (mnm's running instances
    counted free), else the box's as `make_plan` would probe it (`live_budget`: a box this
    one cannot read, such as one without a card). Every hit of `mnm what` is graded on
    it."""
    return planning_budget(service) or live_budget(service.settings)
