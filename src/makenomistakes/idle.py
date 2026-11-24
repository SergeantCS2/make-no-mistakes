"""The router's idle unload: how long each loaded instance has gone without a reply
in flight, and whether that is long enough to unload it.

The timer counts from the moment its last reply finished, or, for a container the router
has served nothing from yet, from the first time the router saw that container running.
A new container (a restart, a load) starts a new timer, whatever the old one said.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from threading import Event

from makenomistakes.instance_types import InstanceStatus
from makenomistakes.journal import Lasting
from makenomistakes.recipes import Recipe
from makenomistakes.supervisor import Health


class IdleClock:
    """Thread-safe: the request threads note finished replies, the sweep reads them."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._finished: dict[str, float] = {}
        self._seen: dict[str, tuple[str | None, float]] = {}  # name -> (container, since)
        # name -> (a client's keep_alive in seconds (0: never), the recipe's (idle, keep) it
        # was asked over): a saved change of either drops it
        self._limits: dict[str, tuple[int, tuple[int, str]]] = {}

    def finished(self, name: str) -> None:
        """Its last request in flight just ended."""
        with self._lock:
            self._finished[name] = self._clock()

    def idle_for(self, status: InstanceStatus) -> float:
        """Seconds since this container last finished a reply (or was first seen)."""
        name, now = status.recipe.name, self._clock()
        with self._lock:
            container, since = self._seen.get(name, (None, now))
            if name not in self._seen or container != status.started_at:
                since = now  # a container the timer has not seen: it starts now
                self._seen[name] = (status.started_at, since)
            return now - max(since, self._finished.get(name, since))

    def forget_keep(self, name: str) -> None:
        """The user saved its `idle` or `keep` (`mnm pin`, `unpin`, `set`): a client's
        keep_alive gives way, even when the saved values did not change."""
        with self._lock:
            self._limits.pop(name, None)

    def forget(self, name: str) -> None:
        """It was unloaded: the next container starts a new timer."""
        with self._lock:
            self._seen.pop(name, None)
            self._finished.pop(name, None)

    # --- a client's `keep_alive` (Ollama's API): held here, never saved -------------------

    def keep(self, recipe: Recipe, seconds: int) -> None:
        """A client asked to keep it loaded `seconds` after its reply (negative: no idle
        unload at all). It holds until the router restarts or the user saves another `idle`
        or `keep` (`mnm pin`, `unpin`, `set idle=`): the user's saved word is the last."""
        with self._lock:
            self._limits[recipe.name] = (0 if seconds < 0 else seconds, _saved(recipe))

    def limit(self, recipe: Recipe) -> int:
        """Its idle limit in seconds, 0 for none: a client's `keep_alive`, else the recipe's."""
        with self._lock:
            held = self._limits.get(recipe.name)
            if held is None:
                return recipe.idle
            if held[1] != _saved(recipe):  # the user changed it since: theirs wins
                del self._limits[recipe.name]
                return recipe.idle
            return held[0]

    def expires_at(self, status: InstanceStatus, wall: Callable[[], float] = time.time) -> float:
        """When the idle unload would take it (seconds since the epoch, `inf` for never)."""
        limit = self.limit(status.recipe)
        if status.recipe.pinned or limit == 0:
            return math.inf
        return wall() + max(limit - self.idle_for(status), 0.0)


def _saved(recipe: Recipe) -> tuple[int, str]:
    return recipe.idle, recipe.keep


def due(status: InstanceStatus, busy: int, idle_for: float, limit: int | None = None) -> bool:
    """Whether the router unloads it now: loaded and ready, not pinned, an idle limit set
    (`limit`, else the recipe's), nothing in flight, and idle at least that long."""
    recipe = status.recipe
    limit = recipe.idle if limit is None else limit
    return (
        status.running
        and status.health is Health.READY
        and not recipe.pinned
        and limit > 0
        and busy == 0
        and idle_for >= limit
    )


SWEEP_SECONDS = 30.0  # how often the router looks; an idle limit is met within this much


def sweep_forever(
    unload: Callable[[], list[str]],
    report: Callable[[str], None],
    stopped: Event,
    every: float = SWEEP_SECONDS,
) -> None:
    """Run `unload` every `every` seconds until `stopped` is set. A sweep that fails (Docker
    away, a recipe unreadable) is reported and the next one tries again: the thread never
    dies of it. A failure that lasts is said once (`journal.Lasting`: a recipe a newer mnm wrote
    would fill the journal with the same line every 30 s)."""
    said = Lasting(report)
    while not stopped.wait(every):
        try:
            for name in unload():
                report(f"Unloaded {name}: idle past its limit (the next request loads it).")
        except Exception as exc:
            said.failed(f"The idle sweep failed and tries again every {every:.0f} s: {exc}")
        else:
            said.worked()
