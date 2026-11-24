"""The router's unloading: the idle sweep
and a client's `keep_alive: 0` park, each decided again under the box's lock."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from makenomistakes.held_back import persistent_members_in, sets_holding
from makenomistakes.idle import IdleClock, due
from makenomistakes.instance_types import InstanceError, InstanceStatus
from makenomistakes.recipes import Recipe

if TYPE_CHECKING:
    from makenomistakes.instances import InstanceService
    from makenomistakes.journal import Note
    from makenomistakes.router_metrics import RouterCounts
    from makenomistakes.swap import SwapService
    from makenomistakes.workflows import WorkflowStore

NOT_LOADED_NOW = "not loaded"


@dataclass(frozen=True)
class Kept:
    """Why `Router.park` left an instance as it was; `loaded` False: there was nothing to
    unload (no journal line needed)."""

    why: str
    loaded: bool = True


class Parking:
    """`Router`'s unloading half; the router sets every attribute named here."""

    instances: InstanceService
    note: Note
    idle: IdleClock
    counts: RouterCounts
    _workflows: WorkflowStore
    _swap: SwapService
    _busy: dict[str, int]
    _count: threading.Lock
    _unloading: set[str]
    _sets_unreadable: bool

    def _forget_listed(self) -> None: ...  # the router's model-list cache

    # --- idle unload ------------------------------------------------------------

    def unload_idle(self) -> list[str]:
        """Unload what has gone idle past its limit (`idle.due`), each under the box's lock
        and decided again once it is held: a load or a request meanwhile keeps it. A request
        that arrives during the unload waits it out, then loads the model again. A persistent
        set's members are left alone; with the sets unreadable, every model is (which
        must stay is unknown), said once in the journal."""
        steady = persistent_members_in(self._workflows)
        if steady is None:
            if not self._sets_unreadable:
                self.note(
                    "The idle sweep unloads nothing: the sets cannot be read (`mnm sets ls`)."
                )
            self._sets_unreadable = True
            return []
        self._sets_unreadable = False
        return [
            s.recipe.name
            for s in self.instances.statuses()
            if s.recipe.name not in steady and self._due(s) and self._unload(s.recipe, self._due)
        ]

    def park(self, recipe: Recipe) -> Kept | None:
        """Unload it now, as an idle unload would (a client's `keep_alive: 0`), unless the
        router keeps it: what keeps it (`Kept`), or None once it is parked. Decided again
        under the box's lock, as the idle unload decides."""
        kept: list[Kept] = []

        def keeps(why: Kept | None) -> bool:
            if why is not None:
                kept.append(why)
            return why is None

        def parkable(status: InstanceStatus) -> bool:  # under the request count's lock
            return keeps(self._in_flight(status.recipe.name))

        if self._unload(recipe, parkable, slow=lambda status: keeps(self._kept(status))):
            return None
        return kept[0] if kept else Kept(NOT_LOADED_NOW, loaded=False)

    def _in_flight(self, name: str) -> Kept | None:
        """Replies in flight keep it (read under `_count`)."""
        busy = self._busy.get(name, 0)
        if busy:
            return Kept(f"{busy} other request{'s' if busy > 1 else ''} in flight")
        return None

    def _kept(self, status: InstanceStatus) -> Kept | None:
        """Why the router keeps it loaded against a client's `keep_alive: 0` by the rule it
        never breaks to make room (pinned, a member of a plain set in use); None when that
        rule lets it go. It reads the sets and Docker, so never under `_count` (every
        request would wait on it)."""
        recipe = status.recipe
        if not status.running:
            return Kept(NOT_LOADED_NOW, loaded=False)
        if recipe.pinned:
            return Kept("pinned (`keep=always`)")
        try:
            workflows = self._workflows.list()
        except InstanceError as exc:
            return Kept(f"the sets could not be read ({exc}), so none is known to be free")
        statuses = self.instances.statuses()
        if sets := sets_holding(workflows, statuses, recipe.name):
            names = ", ".join(sets)
            return Kept(f"a member of the set {names} in use (`mnm sets stop {sets[0]}`)")
        return None

    @contextmanager
    def unloading(self, name: str) -> Iterator[None]:
        """A park the router's load makes for room (`SwapService.park`): a request for it
        meanwhile waits the park out, then loads it again, as for an idle unload. Its stop
        saves its prompt first, seconds in which a request let through would be cut
        off by the `docker stop` after it."""
        with self._count:
            self._unloading.add(name)
        try:
            yield
        finally:
            with self._count:
                self._unloading.discard(name)

    def _unload(
        self,
        recipe: Recipe,
        still: Callable[[InstanceStatus], bool],
        slow: Callable[[InstanceStatus], bool] | None = None,
    ) -> bool:
        """Park it under the box's lock if `slow` (outside the request count's lock) and
        `still` (inside it) hold once the box's lock is held."""
        name = recipe.name
        with self._swap.turn(recipe):
            again = self.instances.status(recipe)
            if slow is not None and not slow(again):
                return False
            with self._count:
                if not still(again):
                    return False
                self._unloading.add(name)
            try:
                self.instances.stop(again.recipe, park=True, told=self.note)
            finally:
                with self._count:
                    self._unloading.discard(name)
        self.idle.forget(name)
        self._forget_listed()
        self.counts.unloaded()  # beside the journal's line
        return True

    def _due(self, status: InstanceStatus) -> bool:
        name = status.recipe.name
        limit = self.idle.limit(status.recipe)
        return due(status, self._busy.get(name, 0), self.idle.idle_for(status), limit)
