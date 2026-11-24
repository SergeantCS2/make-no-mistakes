"""What a start may park to make room, worded once for the router's loads and the user's
`--stop-others`: every other instance except what the router never
parks (`workflows.held_back`: pinned, a member of a set in use) and, for the router, one with
a reply in flight."""

from __future__ import annotations

from collections.abc import Callable

from makenomistakes.held_back import held_back
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.recipes.model import Recipe
from makenomistakes.set_store import WorkflowStore


def parkable(
    instances: InstanceService,
    store: WorkflowStore,
    name: str,
    busy: Callable[[str], int] = lambda _name: 0,
) -> list[Recipe] | None:
    """The instances a start of `name` may park, or None when the sets cannot be read (then
    nothing is parked: which members run together is unknown). An unreadable recipe raises
    `UnreadableStateError` from `instances.all()`."""
    recipes = instances.all()
    try:
        workflows = store.list()
    except InstanceError:
        return None
    kept = held_back(workflows, [instances.status(r) for r in recipes])
    return [r for r in recipes if r.name != name and r.name not in kept and busy(r.name) == 0]
