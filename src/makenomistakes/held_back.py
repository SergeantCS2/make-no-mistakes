"""What nobody parks to make room (`Router.scope`, `mnm start --stop-others`, ComfyUI's room,
the on-request verdicts): one rule, read by all of them."""

from __future__ import annotations

from typing import TYPE_CHECKING

from makenomistakes.instance_types import InstanceError, InstanceStatus

if TYPE_CHECKING:
    from makenomistakes.set_store import Workflow, WorkflowStore


def held_back(workflows: list[Workflow], statuses: list[InstanceStatus]) -> frozenset[str]:
    """A pinned instance (`keep=always`), the members of a plain set in use (started as a set
    and a member running: they run together on purpose), and every member of a
    persistent set (llama-swap's persistent groups). A running member of a started plain
    set was held already; what persistent adds is the idle sweep's (`persistent_members`)."""
    running = {s.recipe.name for s in statuses if s.running}
    together = {
        member
        for workflow in workflows
        if workflow.persistent or _in_use(workflow, running)
        for member in workflow.members.values()
    }
    return frozenset(together | {s.recipe.name for s in statuses if s.recipe.pinned})


def sets_holding(workflows: list[Workflow], statuses: list[InstanceStatus], name: str) -> list[str]:
    """The plain workflows in use that `held_back` keeps `name` loaded for, by name."""
    running = {s.recipe.name for s in statuses if s.running}
    return [w.name for w in workflows if name in w.members.values() and _in_use(w, running)]


def _in_use(workflow: Workflow, running: set[str]) -> bool:
    """A plain set started as a set (`mnm sets start`, until `mnm sets stop` or `rm`) with a
    member running: its members run together on purpose. A member the router loaded on
    request, or one started by hand, holds nothing (else the desk's coder, loaded by a request,
    would hold the card against its own chat). A persistent set is in use whenever a member
    runs."""
    started = workflow.started or workflow.persistent
    return not workflow.swap and started and bool(running & set(workflow.members.values()))


def held_back_in(store: WorkflowStore, statuses: list[InstanceStatus]) -> frozenset[str] | None:
    """`held_back` over a store's sets; None when they cannot be read (then no verdict says
    the router makes the room)."""
    try:
        workflows = store.list()
    except InstanceError:
        return None
    return held_back(workflows, statuses)


def persistent_members_in(store: WorkflowStore) -> frozenset[str] | None:
    """`persistent_members` over a store's sets; None when they cannot be read: then which
    members must stay is unknown, and the idle sweep unloads nothing (as `Router.park` keeps
    a model it cannot judge)."""
    try:
        workflows = store.list()
    except InstanceError:
        return None
    return persistent_members(workflows)


def persistent_members(workflows: list[Workflow]) -> frozenset[str]:
    """Every member of a persistent set: held for room (`held_back`) and, by the router's idle
    sweep, never unloaded for idleness either: a set pinned as one."""
    return frozenset(m for w in workflows if w.persistent for m in w.members.values())
