"""Room for a set's start, made as the router makes it for a request (else `mnm sets start
desk` with zimage loaded refuses "qwen3-8b, embed need about 11.9 GiB together and only
3.6 GiB is free" and never says zimage holds the card).

The candidates are mnm's running instances outside the set that the router would park
(`held_back`: never a pinned one, a member of a set in use or of a persistent set), largest
estimate first; only as many as the set's waiting members need, by the same fit rule
`mnm sets start` judges with (`capacity.fit_together`).

ComfyUI gives its card back first (after a film it holds about 2.3 GiB, and `mnm sets
start pset` would refuse "only 9.0 GiB is free" with every model parked): `release_comfy`
asks it, as the router does before a load, and only then is the room judged. `--no-room`
frees nothing and says what the free would give."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from makenomistakes.capacity import FitNow, VramEstimate, fit_together
from makenomistakes.card_room import lighter_on, short_cards
from makenomistakes.comfy.engine import release_for
from makenomistakes.comfy.image import ComfyBusyError
from makenomistakes.config import Settings
from makenomistakes.hardware.probe import MemorySnapshot
from makenomistakes.held_back import held_back
from makenomistakes.instance_types import InstanceError, Started
from makenomistakes.instances import InstanceService
from makenomistakes.progress_words import parking_for_room
from makenomistakes.recipes.model import Recipe
from makenomistakes.room_verdicts import NOTHING, ComfyRoom, comfy_room, lighter
from makenomistakes.set_members import Member, WorkflowError
from makenomistakes.swap import SwapService, comfy_on_processor, frees_comfy_for

if TYPE_CHECKING:
    from makenomistakes.workflows import Workflow, WorkflowService

Progress = Callable[[str], None]


@dataclass(frozen=True)
class ComfyHooks:
    """ComfyUI as a set's start sees it: what it holds now (`room`) and the free that gives
    it back (`release`, `ComfyEngine.release_vram`); swappable."""

    room: Callable[[], ComfyRoom]
    release: Callable[[Progress | None], bool]
    in_ram: Callable[[], bool] = lambda: False  # ComfyUI's CPU build (`comfy_on_processor`)

    @classmethod
    def of(cls, settings: Settings) -> ComfyHooks:
        return cls(
            lambda: comfy_room(settings),
            release_for(settings),
            lambda: comfy_on_processor(settings),
        )


def release_comfy(
    comfy: ComfyHooks, name: str, progress: Progress | None, members: list[Member]
) -> bool:
    """Free ComfyUI's room before the room is judged (True when it was asked), unless every
    member loads where ComfyUI is not (`frees_comfy_for`); a run still going after a minute
    refuses the start, as a load beside it would run out of memory."""
    recipes = [m.recipe for m in members if not m.missing]
    if not frees_comfy_for(recipes, comfy.in_ram):
        return False
    try:
        return comfy.release(progress)
    except ComfyBusyError as exc:
        raise WorkflowError(f"{name} did not start: {exc}") from exc


def make_room(
    service: WorkflowService,
    workflow: Workflow,
    members: list[Member],
    progress: Progress | None,
    anyway: bool,
    room: bool,
) -> list[str]:
    """Park for the set what the router would park when its members do not fit
    (`room_for`); refuse (unless `anyway`) when that cannot make the room. With `room`
    False (`--no-room`) nothing was freed, so ComfyUI's card counts as held, and the
    refusal says when freeing it would do."""
    snapshot = service.instances.memory()
    owner = service.owner(workflow)
    comfy = NOTHING if not room else None  # None: what it holds after the free, read now
    verdict = service.fit_together(members, snapshot, comfy, owner)
    if verdict.level is not FitNow.NO_FIT:
        return []
    waiting = [(m.name, m.estimate) for m in members if not m.running and m.estimate]
    found = room_for(
        service.instances, service.store.list(), {m.name for m in members}, waiting, snapshot
    )
    if not room or not found.park or not found.enough:  # parking nothing makes no room
        if anyway:
            return []
        offer = service.comfy.room() if not room else NOTHING
        helps = offer.freeable and _fits(waiting, lighter(snapshot, offer.freeable), [])
        note = f"{verdict.note} {offer.offer(workflow.name)}" if helps else verdict.note
        raise WorkflowError(refusal(workflow.name, note, found, room))
    return park(service.instances, workflow.name, found, progress)


@dataclass(frozen=True)
class SetStarted:
    """A set's start: its members' starts, and what was parked to make their room."""

    started: list[Started]
    parked: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Room:
    """What a set's start would park (`park`, largest first), or why it cannot be made:
    `held` names the running instances outside the set the router never parks."""

    park: list[Recipe] = field(default_factory=list)
    enough: bool = True  # parking `park` lets the waiting members fit
    held: list[str] = field(default_factory=list)


def room_for(
    instances: InstanceService,
    sets: list[Workflow],
    own: set[str],
    waiting: list[tuple[str, VramEstimate]],
    snapshot: MemorySnapshot,
) -> Room:
    """The fewest parkable instances, largest first, whose estimates make the waiting
    members fit together; `enough` False when even all of them do not. Only those holding
    a place (a card, the RAM) the members are short on are offered, as `capacity.with_room`
    offers them (else a set on the processor's image parks a card's 9 GiB chat model, which
    frees nothing in the RAM, before the RAM's own holder)."""
    try:
        statuses = [instances.status(r) for r in instances.all()]
    except InstanceError:
        return Room(enough=False)
    kept = held_back(sets, statuses)
    running = [s.recipe for s in statuses if s.running and s.recipe.name not in own]
    first = snapshot.gpu.index if snapshot.gpu is not None else 0
    # each holder's bytes per place (a placement's on its cards, else its first card)
    sizes = {r.name: _on_cards(instances.vram_estimate(r), first) for r in running}
    short = _short_places(waiting, snapshot, first)
    offered = sorted(
        (
            r
            for r in running
            if r.name not in kept
            and not r.pinned
            and (not short or any(sizes[r.name].get(i) for i in short))
        ),
        key=lambda r: (-sum(sizes[r.name].values()), r.name),
    )
    chosen: list[Recipe] = []
    freed: list[dict[int, int]] = []
    for recipe in offered:
        if _fits(waiting, snapshot, freed):
            break
        chosen.append(recipe)
        freed.append(sizes[recipe.name])
    held = sorted(r.name for r in running if r.name in kept or r.pinned)
    return Room(chosen, _fits(waiting, snapshot, freed), held)


def _on_cards(estimate: VramEstimate, first: int) -> dict[int, int]:
    return dict(estimate.on) if estimate.on else {first: estimate.bytes or 0}


def _short_places(
    waiting: list[tuple[str, VramEstimate]], snapshot: MemorySnapshot, first: int
) -> set[int]:
    """The places the waiting members together do not fit comfortably on now (`short_cards`
    over their shares summed per place); empty when none reads short, and then every holder
    is offered, as before."""
    need: dict[int, int] = {}
    for _name, estimate in waiting:
        for index, size in _on_cards(estimate, first).items():
            need[index] = need.get(index, 0) + size
    together = VramEstimate(bytes=sum(need.values()), basis="the set", on=tuple(need.items()))
    return short_cards(together, snapshot)


def park(
    instances: InstanceService, name: str, room: Room, progress: Callable[[str], None] | None
) -> list[str]:
    """Park what `room` names, one after another, each after its replies in flight."""
    swap = SwapService(instances)
    for recipe in room.park:
        if progress:
            progress(parking_for_room(recipe.name, name) + "…")
        try:
            swap.park(recipe)
        except InstanceError as exc:
            raise InstanceError(
                f"{name}: {recipe.name} could not be parked to make room: {exc}"
            ) from exc
    return [r.name for r in room.park]


def refusal(name: str, note: str, room: Room, asked: bool) -> str:
    """Why a set does not start: its verdict, what the router would park (`--no-room` was
    given) or what holds the rest of the card and is never parked."""
    words = f"{name} will not fit: {note}"
    if not asked and room.park:
        names = ", ".join(r.name for r in room.park)
        words += f" Parking {names} would make room: `mnm sets start {name}` without --no-room."
    elif room.held:
        held = ", ".join(room.held)
        words += f" {held} hold the rest and are never parked (pinned, or in a set)."
    return words + " `--anyway` starts it regardless."


def _fits(
    waiting: list[tuple[str, VramEstimate]], snapshot: MemorySnapshot, freed: list[dict[int, int]]
) -> bool:
    """The waiting members fit together once `freed` is given back, card by card, the
    RAM's too (a box with no card has the RAM alone)."""
    if snapshot.gpu is None and snapshot.ram is None:
        return False
    after = lighter_on(snapshot, freed)
    return fit_together(waiting, after).level is not FitNow.NO_FIT
