"""Room for the engine self-test's copy (else `mnm engines test sdcpp` answers "not even
zimage fits beside what runs now" while coder8b, which the router would park, holds the
card). The copy is a start like any other: it parks what
a start may park (never a pinned model or a member of a set in use, each after its replies
in flight), largest first and only as many as the room needs, as the router and `mnm sets
start` do. Unlike `mnm start --stop-others`, the instance the copy is made of counts as a
holder too: the copy runs beside it on a port of its own."""

from __future__ import annotations

from typing import TYPE_CHECKING

from makenomistakes.instance_types import InstanceError, Progress
from makenomistakes.parkable import parkable
from makenomistakes.recipes.model import SELFTEST_PREFIX, Recipe
from makenomistakes.swap import Scope, SwapService
from makenomistakes.workflows import WorkflowStore

if TYPE_CHECKING:
    from makenomistakes.instances import InstanceService

NOTHING_FITS = "not even "  # how `engine_selftest.choose`'s room refusal begins


def smallest_present(service: InstanceService, image: str) -> Recipe | None:
    """The downloaded instance on `image` with the smallest estimate (the one `choose` would
    try first), for making room."""
    present = [r for r in service.all() if r.image == image and service.files_present(r)]
    priced = sorted(
        ((service.vram_estimate(r).bytes, r) for r in present),
        key=lambda pair: (pair[0] is None, pair[0] or 0, pair[1].name),
    )
    return priced[0][1] if priced else None


def room_for_copy(
    service: InstanceService, recipe: Recipe, progress: Progress | None = None
) -> list[str]:
    """Park what makes room for a copy of `recipe`; the names parked (none when nothing of
    mnm's holds the room). InstanceError naming them when only models mnm never parks do."""
    store = WorkflowStore(service.settings.expanded_state_dir())
    free = parkable(service, store, SELFTEST_PREFIX)  # every instance a start may park
    if free is None:
        raise InstanceError("mnm cannot read the sets (`mnm sets ls` says why): it parks nothing")
    swap = SwapService(service, scope=Scope(lambda _r: free, makes_way_unpriced=False))
    chosen = swap.to_park(recipe)
    if chosen:
        return swap.park_for(recipe, chosen, progress)
    every = SwapService(service, scope=Scope(lambda _r: service.all(), makes_way_unpriced=False))
    parkable_names = {r.name for r in free}
    held = [r for r in every.to_park(recipe, pinned_too=True) if r.name not in parkable_names]
    if held:
        raise InstanceError(_held_words(held))
    return []


def _held_words(held: list[Recipe]) -> str:
    """The refusal when only what mnm never parks holds the room: each by why it is kept, so a
    pinned holder is named and a parkable one beside a set's member is never called "in a set
    in use"."""
    in_sets = [r.name for r in held if not r.pinned]
    pinned = [r.name for r in held if r.pinned]
    kinds = [f"{', '.join(in_sets)}, in a set in use"] if in_sets else []
    kinds += [f"{', '.join(pinned)}, pinned"] if pinned else []
    those = "it" if len(held) == 1 else "them"
    remedies = []
    if in_sets:
        remedies.append(f"`mnm sets stop <set>` frees {'it' if len(in_sets) == 1 else 'them'}")
    remedies += [f"`mnm set {name} keep=auto` lets it go" for name in pinned]
    return (
        f"the room it needs is held by {'; '.join(kinds)}, and mnm never parks {those} for "
        f"room; {'; '.join(remedies)}"
    )
