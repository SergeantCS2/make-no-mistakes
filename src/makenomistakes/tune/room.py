"""Room for a trial's copy: what the router would park for a load, by its rules.

ComfyUI gives its card back first (it frees between jobs, and a job still running makes the
trial wait). Then the copy's fit-now verdict names what to park among what the router may
unload (`parkable`: never a pinned model, a member of a set in use or a persistent set, nor
one with a reply in flight), each after its replies; the instance itself counts as parkable
when it runs and nothing holds it, since the copy takes its place."""

from __future__ import annotations

from collections.abc import Callable

from makenomistakes.comfy.engine import release_for
from makenomistakes.comfy.image import ComfyBusyError
from makenomistakes.held_back import held_back_in
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.parkable import parkable
from makenomistakes.recipes import Recipe
from makenomistakes.recipes.model import TUNE_PREFIX
from makenomistakes.set_store import WorkflowStore
from makenomistakes.swap import Progress, Scope, SwapService
from makenomistakes.tune.words import reason

Busy = Callable[[str], int]


class NoRoomError(RuntimeError):
    """The card cannot be made free for the copy now (ComfyUI mid-run, a park that failed,
    the sets unreadable): the trial waits, uncounted."""


def held(service: InstanceService, workflows: WorkflowStore) -> str | None:
    """Why a queued tune waits: something the router never parks holds the card (a pinned
    model running, a set in use: `held_back`, the rule room-making keeps), or what holds it
    cannot be read. One recipe mnm cannot read lists none (a newer mnm's, a damaged file):
    every card job waits then, said once, rather than the tick raising every 5 s."""
    try:
        statuses = service.statuses()
    except InstanceError as exc:
        return f"mnm's instances cannot be listed now ({reason(exc)})"
    kept = held_back_in(workflows, statuses)
    if kept is None:
        return "the sets cannot be read (`mnm sets ls`)"
    holding = sorted(s.recipe.name for s in statuses if s.running and s.recipe.name in kept)
    if holding:
        return f"{', '.join(holding)} runs pinned or in a set in use, and keeps the card"
    return None


def trial_room(
    service: InstanceService,
    workflows: WorkflowStore,
    recipe: Recipe,
    busy: Busy,
    release: Callable[[Progress | None], bool] | None = None,
) -> Callable[[Recipe], list[str]]:
    """The trial's `room`: free ComfyUI, then park what the copy's verdict names."""
    free = release or release_for(service.settings)

    def room(copy: Recipe) -> list[str]:
        try:
            free(None)
        except ComfyBusyError as exc:
            raise NoRoomError(str(exc)) from exc
        members = parkable(service, workflows, recipe.name, busy)
        if members is None:
            raise NoRoomError("the sets cannot be read, so nothing is known to be parkable")
        if service.status(recipe).running and busy(recipe.name) == 0:
            members = [*members, recipe]
        scope = Scope(lambda _r: members, makes_way_unpriced=False)
        swap = SwapService(
            service, busy=lambda r: busy(r.name), scope=scope, release=lambda _p: False
        )
        try:
            return swap.park_for(copy, swap.to_park(copy))
        except InstanceError as exc:
            raise NoRoomError(str(exc)) from exc

    return room


def copies_room(
    service: InstanceService,
    workflows: WorkflowStore,
    busy: Busy,
    release: Callable[[Progress | None], bool] | None = None,
) -> Callable[[Recipe], list[str]]:
    """`trial_room` for whichever instance a copy (`tune-<name>`) is of."""

    def room(copy: Recipe) -> list[str]:
        original = service.find(copy.name.removeprefix(TUNE_PREFIX))
        if original is None:
            raise NoRoomError(f"{copy.name}'s instance is gone")
        return trial_room(service, workflows, original, busy, release)(copy)

    return room
